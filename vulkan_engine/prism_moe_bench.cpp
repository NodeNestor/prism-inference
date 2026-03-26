// Prism MoE Benchmark — Clean, isolated test of MoE pipeline
// No other tests that might crash/TDR the GPU.

#define VK_USE_PLATFORM_WIN32_KHR
#include "deps/volk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <fstream>
#include <chrono>

static VkPipeline loadPipeline(VkDevice device, VkPipelineLayout layout, const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) return VK_NULL_HANDLE;
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint32_t> code(sz / 4);
    f.read((char*)code.data(), sz);
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sz; smi.pCode = code.data();
    VkShaderModule mod;
    if (vkCreateShaderModule(device, &smi, nullptr, &mod) != VK_SUCCESS) return VK_NULL_HANDLE;
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod; cpci.stage.pName = "main"; cpci.layout = layout;
    VkPipeline pipe;
    VkResult r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe);
    vkDestroyShaderModule(device, mod, nullptr);
    return (r == VK_SUCCESS) ? pipe : VK_NULL_HANDLE;
}
static uint32_t findMem(VkPhysicalDevice phys, uint32_t tf, VkMemoryPropertyFlags p) {
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((tf & (1 << i)) && (mp.memoryTypes[i].propertyFlags & p) == p) return i;
    return 0;
}
static void createBuf(VkDevice dev, VkPhysicalDevice phys, VkDeviceSize size,
                       VkBufferUsageFlags usage, VkMemoryPropertyFlags mp,
                       VkBuffer& buf, VkDeviceMemory& mem) {
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = size; bi.usage = usage;
    vkCreateBuffer(dev, &bi, nullptr, &buf);
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, buf, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size; ai.memoryTypeIndex = findMem(phys, mr.memoryTypeBits, mp);
    vkAllocateMemory(dev, &ai, nullptr, &mem); vkBindBufferMemory(dev, buf, mem, 0);
}
static void addBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}

int main() {
    printf("=== Prism MoE Benchmark (clean) ===\n\n");

    volkInitialize();
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &ai;
    VkInstance instance; vkCreateInstance(&ici, nullptr, &instance); volkLoadInstance(instance);
    uint32_t gc = 0; vkEnumeratePhysicalDevices(instance, &gc, nullptr);
    std::vector<VkPhysicalDevice> gpus(gc); vkEnumeratePhysicalDevices(instance, &gc, gpus.data());
    VkPhysicalDevice phys = gpus[0];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(phys, &props);
    printf("GPU: %s\n\n", props.deviceName);

    VkPhysicalDeviceCooperativeVectorFeaturesNV cv = {(VkStructureType)1000553000};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR}; cm.pNext = &cv;
    VkPhysicalDeviceShaderFloat16Int8Features f16 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES}; f16.pNext = &cm;
    VkPhysicalDevice16BitStorageFeatures s16 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES}; s16.pNext = &f16;
    VkPhysicalDeviceFeatures2 feat2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; feat2.pNext = &s16;
    vkGetPhysicalDeviceFeatures2(phys, &feat2);

    uint32_t qc = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &qc, nullptr);
    std::vector<VkQueueFamilyProperties> qp(qc); vkGetPhysicalDeviceQueueFamilyProperties(phys, &qc, qp.data());
    uint32_t qf = 0;
    for (uint32_t i = 0; i < qc; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = qf; qi.queueCount = 1; qi.pQueuePriorities = &prio;
    const char* exts[] = {"VK_KHR_cooperative_matrix", "VK_NV_cooperative_vector"};
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qi;
    dci.pNext = &feat2; dci.enabledExtensionCount = 2; dci.ppEnabledExtensionNames = exts;
    VkDevice device; vkCreateDevice(phys, &dci, nullptr, &device); volkLoadDevice(device);
    VkQueue queue; vkGetDeviceQueue(device, qf, 0, &queue);

    VkCommandPool cmdPool;
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = qf; cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device, &cpi, nullptr, &cmdPool);
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cbai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cmdPool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &cbai, &cmd);
    VkFence fence; VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(device, &fi, nullptr, &fence);
    VkQueryPool qpool;
    VkQueryPoolCreateInfo qpci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP; qpci.queryCount = 8;
    vkCreateQueryPool(device, &qpci, nullptr, &qpool);

    // 3 bindings: weights (0), features (1), aux/int buffer (2)
    VkDescriptorSetLayoutBinding binds[3] = {};
    binds[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    binds[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    binds[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 3; dli.pBindings = binds;
    VkDescriptorSetLayout descLayout; vkCreateDescriptorSetLayout(device, &dli, nullptr, &descLayout);
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &descLayout;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pipeLayout; vkCreatePipelineLayout(device, &pli, nullptr, &pipeLayout);
    VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps;
    VkDescriptorPool descPool; vkCreateDescriptorPool(device, &dpci, nullptr, &descPool);
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = descPool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &descLayout;
    VkDescriptorSet descSet; vkAllocateDescriptorSets(device, &dsai, &descSet);

    // Load shaders
    VkPipeline pRouter = loadPipeline(device, pipeLayout, "shaders/moe_router.spv");
    VkPipeline pExpertSkip = loadPipeline(device, pipeLayout, "shaders/moe_expert_skip.spv");
    VkPipeline pAttn = loadPipeline(device, pipeLayout, "shaders/split2_qkv_attn_d256.spv");
    VkPipeline pStdFFN = loadPipeline(device, pipeLayout, "shaders/attention_windowed_d256.spv");

    printf("Router: %s\n", pRouter ? "OK" : "FAIL");
    printf("Expert (skip): %s\n", pExpertSkip ? "OK" : "FAIL");
    printf("Attention d256: %s\n", pAttn ? "OK" : "FAIL");
    printf("Standard d256: %s\n\n", pStdFFN ? "OK" : "FAIL");

    int n_tokens = 120 * 68;  // 8160
    int dim = 256;

    // Buffers
    VkBuffer wBuf, fBuf, auxBuf;
    VkDeviceMemory wMem, fMem, auxMem;
    createBuf(device, phys, 10*1024*1024,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, wBuf, wMem);
    createBuf(device, phys, 128*1024*1024,  // 128MB for full pipeline at 540x960
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, fBuf, fMem);
    // Aux buffer for assignments + counts (int32)
    createBuf(device, phys, (n_tokens + 64) * 4,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, auxBuf, auxMem);

    VkDescriptorBufferInfo wbi = {wBuf,0,VK_WHOLE_SIZE};
    VkDescriptorBufferInfo fbi = {fBuf,0,VK_WHOLE_SIZE};
    VkDescriptorBufferInfo abi = {auxBuf,0,VK_WHOLE_SIZE};
    VkWriteDescriptorSet wr[3] = {};
    wr[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,0,descSet,0,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,0,&wbi,0};
    wr[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,0,descSet,1,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,0,&fbi,0};
    wr[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,0,descSet,2,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,0,&abi,0};
    vkUpdateDescriptorSets(device, 3, wr, 0, nullptr);

    // Weight offsets
    int w = 0;
    auto wa = [&](int n) { int o = w; w += n; return o; };
    int qkv_w = wa(3*256*256), qkv_b = wa(3*256);
    int out_w = wa(256*256), out_b = wa(256);
    int ffn_w1 = wa(1024*256), ffn_b1 = wa(1024), ffn_w2 = wa(256*1024), ffn_b2 = wa(256);
    int rtr_w = wa(4*256), rtr_b = wa(4);
    int exp_stride = 256*256 + 256 + 256*256 + 256;
    int e0_w1 = wa(256*256), e0_b1 = wa(256), e0_w2 = wa(256*256), e0_b2 = wa(256);
    for (int e = 1; e < 16; e++) { wa(256*256); wa(256); wa(256*256); wa(256); }

    // Feature offsets
    int f_in = 0, f_attn = n_tokens * 256, f_out = f_attn + n_tokens * 256;
    // Aux: assignments at 0, counts at n_tokens
    int a_assign = 0, a_count = n_tokens;

    int n_wg = (n_tokens + 255) / 256;
    int total_win = ((120+7)/8) * ((68+7)/8); // 135
    int warmup = 10, loops = 100;

    auto bench = [&](const char* name, auto record_fn) {
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd, &cbi);
        vkCmdResetQueryPool(cmd, qpool, 0, 8);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descSet, 0, nullptr);
        int ts = 0;
        auto stamp = [&]() { vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, ts++); };
        stamp();
        record_fn(cmd, pipeLayout);
        stamp();
        vkEndCommandBuffer(cmd);

        for (int i = 0; i < warmup; i++) {
            VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cmd;
            vkQueueSubmit(queue,1,&si,fence); vkWaitForFences(device,1,&fence,VK_TRUE,UINT64_MAX); vkResetFences(device,1,&fence);
        }
        double total = 0;
        for (int i = 0; i < loops; i++) {
            VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cmd;
            vkQueueSubmit(queue,1,&si,fence); vkWaitForFences(device,1,&fence,VK_TRUE,UINT64_MAX); vkResetFences(device,1,&fence);
            uint64_t t[8];
            vkGetQueryPoolResults(device,qpool,0,ts,sizeof(t),t,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT);
            total += (t[ts-1]-t[0]) * props.limits.timestampPeriod / 1e6;
        }
        printf("  %-45s %7.3f ms\n", name, total / loops);
    };

    // Test 1: Standard d256 monolithic (baseline)
    printf("--- Standard d256 (monolithic, 4 blocks) ---\n");
    bench("d256 monolithic 4blk", [&](VkCommandBuffer c, VkPipelineLayout pl) {
        int ci = f_in, co = f_out;
        for (int b = 0; b < 4; b++) {
            vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pStdFFN);
            int32_t pc[] = {n_tokens, 256, 8, 32, 120, 68, 8,
                qkv_w, qkv_b, out_w, out_b, ffn_w1, ffn_b1, ffn_w2, ffn_b2, ci, co};
            vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
            vkCmdDispatch(c, total_win, 1, 1); addBarrier(c);
            int tmp=ci; ci=co; co=tmp;
        }
    });

    // Test 2: MoE with early-skip experts (4 experts, 4 blocks)
    if (pRouter && pExpertSkip && pAttn) {
        printf("\n--- MoE 4-expert early-skip (2-split, 4 blocks) ---\n");
        bench("MoE-skip 4exp x 4blk", [&](VkCommandBuffer c, VkPipelineLayout pl) {
            int ci = f_in, co = f_out;
            for (int b = 0; b < 4; b++) {
                // Pass 1: Attention
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pAttn);
                int32_t pa[] = {n_tokens, 256, 8, 32, 120, 68, 8,
                    qkv_w, qkv_b, out_w, out_b, ci, f_attn};
                vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pa), pa);
                vkCmdDispatch(c, total_win, 1, 1); addBarrier(c);

                // Router
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pRouter);
                int32_t pr[] = {n_tokens, 256, 4, rtr_w, rtr_b, f_attn, a_assign, a_count};
                vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pr), pr);
                vkCmdDispatch(c, n_wg, 1, 1); addBarrier(c);

                // 4 expert dispatches (each skips non-assigned tokens)
                for (int e = 0; e < 4; e++) {
                    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pExpertSkip);
                    int eoff = e * exp_stride;
                    int32_t pe[] = {n_tokens, 256, e,
                        e0_w1+eoff, e0_b1+eoff, e0_w2+eoff, e0_b2+eoff,
                        f_attn, co, a_assign};
                    vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pe), pe);
                    vkCmdDispatch(c, n_wg, 1, 1); addBarrier(c);
                }

                int tmp=ci; ci=co; co=tmp;
            }
        });
    }

    // Test scaling: more experts, more blocks
    printf("\n--- MoE SCALING ---\n");
    auto run_moe = [&](int n_exp, int nblk) {
        char name[80];
        snprintf(name, 80, "MoE %dexp x %dblk", n_exp, nblk);
        bench(name, [&](VkCommandBuffer c, VkPipelineLayout pl) {
            int ci = f_in, co = f_out;
            for (int b = 0; b < nblk; b++) {
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pAttn);
                int32_t pa[] = {n_tokens, 256, 8, 32, 120, 68, 8,
                    qkv_w, qkv_b, out_w, out_b, ci, f_attn};
                vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pa), pa);
                vkCmdDispatch(c, total_win, 1, 1); addBarrier(c);

                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pRouter);
                int32_t pr[] = {n_tokens, 256, n_exp, rtr_w, rtr_b, f_attn, a_assign, a_count};
                vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pr), pr);
                vkCmdDispatch(c, n_wg, 1, 1); addBarrier(c);

                for (int e = 0; e < n_exp; e++) {
                    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pExpertSkip);
                    int eoff = e * exp_stride;
                    int32_t pe[] = {n_tokens, 256, e,
                        e0_w1+eoff, e0_b1+eoff, e0_w2+eoff, e0_b2+eoff,
                        f_attn, co, a_assign};
                    vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pe), pe);
                    vkCmdDispatch(c, n_wg, 1, 1); addBarrier(c);
                }
                int tmp=ci; ci=co; co=tmp;
            }
        });
    };

    // 4 experts
    run_moe(4, 4);
    run_moe(4, 8);
    run_moe(4, 12);
    run_moe(4, 16);
    run_moe(4, 24);
    // 8 experts (need more expert weights)
    run_moe(8, 4);
    run_moe(8, 8);
    run_moe(8, 12);
    run_moe(8, 16);
    // 16 experts
    run_moe(16, 4);
    run_moe(16, 8);

    // Standard d256 comparison points
    printf("\n--- Standard d256 comparison ---\n");
    for (int nblk : {4, 8, 12, 16, 24}) {
        char name[80];
        snprintf(name, 80, "Standard d256 %dblk", nblk);
        bench(name, [&](VkCommandBuffer c, VkPipelineLayout pl) {
            int ci = f_in, co = f_out;
            for (int b = 0; b < nblk; b++) {
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pStdFFN);
                int32_t pc[] = {n_tokens, 256, 8, 32, 120, 68, 8,
                    qkv_w, qkv_b, out_w, out_b, ffn_w1, ffn_b1, ffn_w2, ffn_b2, ci, co};
                vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
                vkCmdDispatch(c, total_win, 1, 1); addBarrier(c);
                int tmp=ci; ci=co; co=tmp;
            }
        });
    }

    // === END-TO-END TESTS (encoder + transformer + decoder) ===
    printf("\n--- END-TO-END PIPELINE ---\n\n");

    VkPipeline pEncIC = loadPipeline(device, pipeLayout, "shaders/conv3x3_coopvec_9ch.spv");
    VkPipeline pEnc1 = loadPipeline(device, pipeLayout, "shaders/strided_conv_coopvec_32ch.spv");
    VkPipeline pEnc2Split = loadPipeline(device, pipeLayout, "shaders/strided_conv_split_64ch.spv");
    VkPipeline pEnc3Split = loadPipeline(device, pipeLayout, "shaders/strided_conv_split_128ch.spv");
    VkPipeline pProjUp = loadPipeline(device, pipeLayout, "shaders/pw_project_128to256.spv");
    VkPipeline pProjDn = loadPipeline(device, pipeLayout, "shaders/pw_project_256to128.spv");
    VkPipeline pDec3 = loadPipeline(device, pipeLayout, "shaders/dec3_fused.spv");
    VkPipeline pDec2 = loadPipeline(device, pipeLayout, "shaders/dec2_fused.spv");
    VkPipeline pDec1 = loadPipeline(device, pipeLayout, "shaders/dec1_fused.spv");
    VkPipeline pOut12 = loadPipeline(device, pipeLayout, "shaders/pw_conv_coopvec_32to12.spv");
    VkPipeline pPS = loadPipeline(device, pipeLayout, "shaders/pixelshuffle_sigmoid.spv");

    bool e2e_ok = pEncIC && pEnc1 && pEnc2Split && pEnc3Split && pProjUp && pProjDn &&
                  pDec3 && pDec2 && pDec1 && pOut12 && pPS;
    printf("  End-to-end shaders: %s\n\n", e2e_ok ? "ALL OK" : "MISSING");

    if (e2e_ok) {
        // Encoder weights
        int enc_ic_w = wa(32*9*9), enc_ic_b = wa(32);
        int enc1_w = wa(64*32*9), enc1_b = wa(64);
        int enc2_w = wa(128*64*9), enc2_b = wa(128);
        int enc3_w = wa(128*128*9), enc3_b = wa(128);
        int proj_up_w = wa(256*128), proj_up_b = wa(256);
        int proj_dn_w = wa(128*256), proj_dn_b = wa(128);
        // Decoder weights
        int d3_pw_w = wa(128*256), d3_pw_b = wa(128);
        int d2_pw_w = wa(64*192), d2_pw_b = wa(64);
        int d1_pw_w = wa(32*96), d1_pw_b = wa(32);
        int out_pw_w = wa(12*32), out_pw_b = wa(12);

        // Feature offsets (full pipeline)
        struct { int w, h; } r0={960,540}, r1={480,270}, r2={240,135}, r3={120,68}, rD={1920,1080};
        int fp = 0;
        auto fa = [&](int ch, int w, int h) { int o = fp; fp += ch * w * h; return o; };
        int fe_in = fa(9,r0.w,r0.h);
        int fe_e0 = fa(32,r0.w,r0.h);
        int fe_e1 = fa(64,r1.w,r1.h);
        int fe_e2 = fa(128,r2.w,r2.h);
        int fe_e3 = fa(128,r3.w,r3.h);
        int fe_d256_in = fa(256,r3.w,r3.h);
        int fe_d256_out = fa(256,r3.w,r3.h);
        int fe_d256_attn = fa(256,r3.w,r3.h);
        int fe_d3_out = fa(128,r2.w,r2.h);
        int fe_d2_out = fa(64,r1.w,r1.h);
        int fe_d1_out = fa(32,r0.w,r0.h);
        int fe_out12 = fa(12,r0.w,r0.h);
        int fe_disp = fa(3,rD.w,rD.h);

        auto run_e2e = [&](int n_exp, int nblk) {
            char name[80];
            int ffn_per_blk = n_exp * 131584;
            int attn_per_blk = 197376 + 65792 + n_exp * 256 + n_exp;
            int total_trans = nblk * (attn_per_blk + ffn_per_blk);
            int active_per_tok = nblk * (263168 + 131584);
            snprintf(name, 80, "E2E MoE%d×%dblk (%dM tot, %dM act)",
                n_exp, nblk, total_trans/1000000, active_per_tok/1000000);

            bench(name, [&](VkCommandBuffer c, VkPipelineLayout pl) {
                // Encoder (split)
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pEncIC);
                int32_t p0[] = {9,32,r0.w,r0.h,enc_ic_w,enc_ic_b,fe_in,fe_e0,1};
                vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p0),p0);
                vkCmdDispatch(c,(r0.w*r0.h+255)/256,1,1); addBarrier(c);

                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pEnc1);
                int32_t p1[] = {32,64,r0.w,r0.h,r1.w,r1.h,enc1_w,enc1_b,fe_e0,fe_e1};
                vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p1),p1);
                vkCmdDispatch(c,(r1.w*r1.h+255)/256,1,1); addBarrier(c);

                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pEnc2Split);
                int32_t p2[] = {64,128,r1.w,r1.h,r2.w,r2.h,enc2_w,enc2_b,fe_e1,fe_e2};
                vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p2),p2);
                vkCmdDispatch(c,(r2.w*r2.h+255)/256,1,1); addBarrier(c);

                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pEnc3Split);
                int32_t p3[] = {128,128,r2.w,r2.h,r3.w,r3.h,enc3_w,enc3_b,fe_e2,fe_e3};
                vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p3),p3);
                vkCmdDispatch(c,(r3.w*r3.h+255)/256,1,1); addBarrier(c);

                // Project 128→256
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pProjUp);
                int32_t pu[] = {n_tokens, proj_up_w, proj_up_b, fe_e3, fe_d256_in};
                vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pu),pu);
                vkCmdDispatch(c,n_wg,1,1); addBarrier(c);

                // Transformer (MoE)
                int ci = fe_d256_in, co = fe_d256_out;
                for (int b = 0; b < nblk; b++) {
                    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pAttn);
                    int32_t pa[] = {n_tokens, 256, 8, 32, 120, 68, 8,
                        qkv_w, qkv_b, out_w, out_b, ci, fe_d256_attn};
                    vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pa),pa);
                    vkCmdDispatch(c, total_win, 1, 1); addBarrier(c);

                    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pRouter);
                    int32_t pr[] = {n_tokens, 256, n_exp, rtr_w, rtr_b, fe_d256_attn, a_assign, a_count};
                    vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pr),pr);
                    vkCmdDispatch(c, n_wg, 1, 1); addBarrier(c);

                    for (int e = 0; e < n_exp; e++) {
                        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pExpertSkip);
                        int eoff = e * exp_stride;
                        int32_t pe[] = {n_tokens, 256, e,
                            e0_w1+eoff, e0_b1+eoff, e0_w2+eoff, e0_b2+eoff,
                            fe_d256_attn, co, a_assign};
                        vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pe),pe);
                        vkCmdDispatch(c, n_wg, 1, 1); addBarrier(c);
                    }
                    int tmp=ci; ci=co; co=tmp;
                }

                // Project 256→128
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pProjDn);
                int32_t pd[] = {n_tokens, proj_dn_w, proj_dn_b, ci, fe_e3};
                vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pd),pd);
                vkCmdDispatch(c,n_wg,1,1); addBarrier(c);

                // Decoder (fused)
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pDec3);
                int32_t dd3[] = {r2.w,r2.h,r3.w,r3.h,128,d3_pw_w,d3_pw_b,fe_e3,fe_e2,fe_d3_out};
                vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(dd3),dd3);
                vkCmdDispatch(c,(r2.w*r2.h+255)/256,1,1); addBarrier(c);

                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pDec2);
                int32_t dd2[] = {r1.w,r1.h,r2.w,r2.h,128,64,d2_pw_w,d2_pw_b,fe_d3_out,fe_e1,fe_d2_out};
                vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(dd2),dd2);
                vkCmdDispatch(c,(r1.w*r1.h+255)/256,1,1); addBarrier(c);

                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pDec1);
                int32_t dd1[] = {r0.w,r0.h,r1.w,r1.h,64,32,d1_pw_w,d1_pw_b,fe_d2_out,fe_e0,fe_d1_out};
                vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(dd1),dd1);
                vkCmdDispatch(c,(r0.w*r0.h+255)/256,1,1); addBarrier(c);

                // Output
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pOut12);
                int32_t po[] = {32,12,r0.w,r0.h,out_pw_w,out_pw_b,fe_d1_out,fe_out12,0};
                vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(po),po);
                vkCmdDispatch(c,(r0.w*r0.h+255)/256,1,1); addBarrier(c);

                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pPS);
                int32_t ps[] = {r0.w,r0.h,rD.w,rD.h,2,fe_out12,fe_disp};
                vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(ps),ps);
                vkCmdDispatch(c,(rD.w+15)/16,(rD.h+15)/16,1);
            });
        };

        // How it works: each token gets routed to 1 of N experts.
        // The router is a tiny 256→N matmul (argmax picks expert).
        // Each expert FFN dispatch runs on ALL tokens but skips non-assigned ones.
        // So each token only computes 1 expert FFN (256→256→256 = 131K active params).
        // Total params = N_experts × 131K per block. Active = 131K per block.

        printf("  How MoE works:\n");
        printf("    Router: 256→N matmul picks 1 expert per token\n");
        printf("    Each expert dispatch: runs all tokens, skips non-assigned\n");
        printf("    Active params/token/block: 263K attn + 131K expert = 394K\n\n");

        run_e2e(4, 8);
        run_e2e(4, 16);
        run_e2e(4, 22);
        run_e2e(8, 8);
        run_e2e(8, 16);
        run_e2e(8, 22);
        run_e2e(16, 8);
        run_e2e(16, 16);

        // Standard d256 E2E for comparison
        printf("\n  Standard d256 E2E comparison:\n");
        for (int nblk : {8, 12, 16}) {
            char name[80];
            snprintf(name, 80, "E2E Standard d256 %dblk (%dM)", nblk, nblk * 789 / 1000);
            bench(name, [&](VkCommandBuffer c, VkPipelineLayout pl) {
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pEncIC);
                int32_t p0[] = {9,32,r0.w,r0.h,enc_ic_w,enc_ic_b,fe_in,fe_e0,1};
                vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p0),p0);
                vkCmdDispatch(c,(r0.w*r0.h+255)/256,1,1); addBarrier(c);
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pEnc1);
                int32_t p1[] = {32,64,r0.w,r0.h,r1.w,r1.h,enc1_w,enc1_b,fe_e0,fe_e1};
                vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p1),p1);
                vkCmdDispatch(c,(r1.w*r1.h+255)/256,1,1); addBarrier(c);
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pEnc2Split);
                int32_t p2[] = {64,128,r1.w,r1.h,r2.w,r2.h,enc2_w,enc2_b,fe_e1,fe_e2};
                vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p2),p2);
                vkCmdDispatch(c,(r2.w*r2.h+255)/256,1,1); addBarrier(c);
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pEnc3Split);
                int32_t p3[] = {128,128,r2.w,r2.h,r3.w,r3.h,enc3_w,enc3_b,fe_e2,fe_e3};
                vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p3),p3);
                vkCmdDispatch(c,(r3.w*r3.h+255)/256,1,1); addBarrier(c);

                int ci = fe_e3, co = fe_d256_out;
                for (int b = 0; b < nblk; b++) {
                    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pStdFFN);
                    int32_t pc[] = {n_tokens, 256, 8, 32, 120, 68, 8,
                        qkv_w, qkv_b, out_w, out_b, ffn_w1, ffn_b1, ffn_w2, ffn_b2, ci, co};
                    vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(pc),pc);
                    vkCmdDispatch(c, total_win, 1, 1); addBarrier(c);
                    int tmp=ci; ci=co; co=tmp;
                }

                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pDec3);
                int32_t dd3[] = {r2.w,r2.h,r3.w,r3.h,128,d3_pw_w,d3_pw_b,ci,fe_e2,fe_d3_out};
                vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(dd3),dd3);
                vkCmdDispatch(c,(r2.w*r2.h+255)/256,1,1); addBarrier(c);
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pDec2);
                int32_t dd2[] = {r1.w,r1.h,r2.w,r2.h,128,64,d2_pw_w,d2_pw_b,fe_d3_out,fe_e1,fe_d2_out};
                vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(dd2),dd2);
                vkCmdDispatch(c,(r1.w*r1.h+255)/256,1,1); addBarrier(c);
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pDec1);
                int32_t dd1[] = {r0.w,r0.h,r1.w,r1.h,64,32,d1_pw_w,d1_pw_b,fe_d2_out,fe_e0,fe_d1_out};
                vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(dd1),dd1);
                vkCmdDispatch(c,(r0.w*r0.h+255)/256,1,1); addBarrier(c);
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pOut12);
                int32_t po[] = {32,12,r0.w,r0.h,out_pw_w,out_pw_b,fe_d1_out,fe_out12,0};
                vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(po),po);
                vkCmdDispatch(c,(r0.w*r0.h+255)/256,1,1); addBarrier(c);
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pPS);
                int32_t ps[] = {r0.w,r0.h,rD.w,rD.h,2,fe_out12,fe_disp};
                vkCmdPushConstants(c,pl,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(ps),ps);
                vkCmdDispatch(c,(rD.w+15)/16,(rD.h+15)/16,1);
            });
        }
    }

    printf("\n--- PARAMS vs TIME COMPARISON ---\n\n");
    printf("  Fixed cost (enc+dec+out): ~2.1ms\n");
    printf("  60fps budget: 16.67ms - 2.1ms = 14.6ms for transformer\n\n");
    printf("  Config                    Per-blk  @60fps-blks  FFN/blk   Total params\n");
    printf("  Standard d256             ~0.95ms  15 blocks    525K      ~11.8M\n");
    printf("  MoE 4exp d256             ~0.65ms  22 blocks    2.1M      ~49M\n");
    printf("  MoE 8exp d256             ~???ms   ?? blocks    4.2M      ~???M\n");
    printf("  MoE 16exp d256            ~???ms   ?? blocks    8.4M      ~???M\n");

    // Cleanup
    vkDeviceWaitIdle(device);
    vkDestroyQueryPool(device, qpool, nullptr);
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, cmdPool, nullptr);
    vkDestroyBuffer(device, wBuf, nullptr); vkFreeMemory(device, wMem, nullptr);
    vkDestroyBuffer(device, fBuf, nullptr); vkFreeMemory(device, fMem, nullptr);
    vkDestroyBuffer(device, auxBuf, nullptr); vkFreeMemory(device, auxMem, nullptr);
    auto dp = [&](VkPipeline p) { if (p) vkDestroyPipeline(device,p,nullptr); };
    dp(pRouter); dp(pExpertSkip); dp(pAttn); dp(pStdFFN);
    vkDestroyPipelineLayout(device,pipeLayout,nullptr);
    vkDestroyDescriptorPool(device,descPool,nullptr);
    vkDestroyDescriptorSetLayout(device,descLayout,nullptr);
    vkDestroyDevice(device,nullptr); vkDestroyInstance(instance,nullptr);
    return 0;
}
