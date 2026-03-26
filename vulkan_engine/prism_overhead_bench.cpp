// Prism Overhead Benchmark — Find and fix the 170x gap
//
// Tests isolate individual bottlenecks:
//   1. FFN-only (1 tok/thread) — tensor core ceiling at dim=128
//   2. FFN-only (4 tok/thread) — weight caching effect
//   3. Original monolithic windowed (8x8) — current baseline
//   4. Split pipeline (QKV + Attn + OutFFN) — occupancy fix test
//   5. 16x16 windowed (256 threads/WG) — larger workgroups
//   6. 3-dispatch linear attention — global attention overhead
//
// All run at 8160 tokens (68x120), dim=128, 4 heads, FFN=512

#define VK_USE_PLATFORM_WIN32_KHR
#include "deps/volk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <fstream>
#include <chrono>

struct GenericPush { int32_t data[32]; };  // 128 bytes max push constant

struct Resolution { int w, h; int pixels() const { return w * h; } };
struct WeightAlloc { int offset, count; };

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
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(phys, &mp);
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
    printf("================================================================\n");
    printf("  Prism Overhead Analysis Benchmark\n");
    printf("  Isolating the 170x gap from theoretical peak\n");
    printf("================================================================\n\n");

    // Vulkan init (abbreviated)
    volkInitialize();
    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &ai;
    VkInstance instance; vkCreateInstance(&ici, nullptr, &instance); volkLoadInstance(instance);

    uint32_t gc = 0; vkEnumeratePhysicalDevices(instance, &gc, nullptr);
    std::vector<VkPhysicalDevice> gpus(gc); vkEnumeratePhysicalDevices(instance, &gc, gpus.data());
    VkPhysicalDevice phys = gpus[0];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(phys, &props);
    printf("GPU: %s\n\n", props.deviceName);

    // Feature chain
    VkPhysicalDeviceCooperativeVectorFeaturesNV cv = {(VkStructureType)1000553000};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    cm.pNext = &cv;
    VkPhysicalDeviceShaderFloat16Int8Features f16 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    f16.pNext = &cm;
    VkPhysicalDevice16BitStorageFeatures s16 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    s16.pNext = &f16;
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

    const int MTS = 32;
    VkQueryPool qpool;
    VkQueryPoolCreateInfo qpci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP; qpci.queryCount = MTS;
    vkCreateQueryPool(device, &qpci, nullptr, &qpool);

    // Descriptor set
    VkDescriptorSetLayoutBinding binds[2] = {};
    binds[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    binds[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 2; dli.pBindings = binds;
    VkDescriptorSetLayout descLayout; vkCreateDescriptorSetLayout(device, &dli, nullptr, &descLayout);
    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &descLayout;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pipeLayout; vkCreatePipelineLayout(device, &pli, nullptr, &pipeLayout);
    VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps;
    VkDescriptorPool descPool; vkCreateDescriptorPool(device, &dpci, nullptr, &descPool);
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = descPool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &descLayout;
    VkDescriptorSet descSet; vkAllocateDescriptorSets(device, &dsai, &descSet);

    // Load pipelines
    printf("Loading shaders...\n");
    #define LOAD(v, f) VkPipeline v = loadPipeline(device, pipeLayout, "shaders/" f); \
        printf("  %-40s %s\n", f, v ? "OK" : "FAIL");

    LOAD(pFFN1,    "bench_ffn_1tok.spv");
    LOAD(pFFN4,    "bench_ffn_4tok.spv");
    LOAD(pWin8,    "attention_windowed.spv");
    LOAD(pQKV,     "split_qkv_project.spv");
    LOAD(pAttn,    "split_windowed_attn.spv");
    LOAD(pOutFFN,  "split_out_ffn.spv");
    LOAD(pWin16,   "attention_windowed_16x16.spv");
    LOAD(pLinKV,   "linear_attn_kv.spv");
    LOAD(pLinRed,  "linear_attn_reduce.spv");
    LOAD(pLinQFFN, "linear_attn_qffn.spv");
    LOAD(pWinFast, "attention_windowed_fast.spv");
    LOAD(pSplit2A,  "split2_qkv_attn.spv");
    LOAD(pSplit2F,  "split2_ffn_cv256.spv");
    LOAD(pFFNSplit,"bench_ffn_split.spv");
    LOAD(pFFNcv256,"bench_ffn_cv256.spv");
    LOAD(pFFNW1,   "bench_ffn_w1only.spv");
    LOAD(pFFNW2,   "bench_ffn_w2only.spv");
    #undef LOAD

    // Constants
    Resolution r3 = {120, 68};
    int n_tokens = r3.pixels();  // 8160
    int dim = 128;

    // Allocate buffers: weights (2MB) + features (16MB, plenty of room)
    int total_weights = 2 * 1024 * 1024;  // 2M fp16 = 4MB, enough for any config
    int total_feat = 8 * 1024 * 1024;     // 8M fp16 = 16MB

    VkBuffer wBuf, fBuf; VkDeviceMemory wMem, fMem;
    createBuf(device, phys, (VkDeviceSize)total_weights * 2,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, wBuf, wMem);
    createBuf(device, phys, (VkDeviceSize)total_feat * 2,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, fBuf, fMem);

    VkDescriptorBufferInfo wbi = {wBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo fbi = {fBuf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wr[2] = {};
    wr[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, descSet, 0, 0, 1,
             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &wbi, 0};
    wr[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, 0, descSet, 1, 0, 1,
             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &fbi, 0};
    vkUpdateDescriptorSets(device, 2, wr, 0, nullptr);

    // Weight offsets (all tests share the same weight layout conceptually)
    // Just use sequential offsets, doesn't matter what the actual values are
    int w = 0;
    auto wa = [&](int n) { int o = w; w += n; return o; };

    int qkv_w = wa(3*128*128), qkv_b = wa(3*128);
    int out_w = wa(128*128), out_b = wa(128);
    int ffn_w1 = wa(512*128), ffn_b1 = wa(512);
    int ffn_w2 = wa(128*512), ffn_b2 = wa(128);
    // For separate K,V,Q weights (linear attention)
    int k_w = wa(128*128), k_b = wa(128);
    int v_w = wa(128*128), v_b = wa(128);
    int q_w = wa(128*128), q_b = wa(128);

    // Feature offsets
    int f_in = 0;
    int f_out = n_tokens * 128;
    int f_q = f_out + n_tokens * 128;
    int f_k = f_q + n_tokens * 128;
    int f_v = f_k + n_tokens * 128;
    int f_attn = f_v + n_tokens * 128;
    int f_phiK = f_attn + n_tokens * 128;
    int f_V = f_phiK + n_tokens * 128;
    int f_S = f_V + n_tokens * 128;
    int f_Z = f_S + 4 * 32 * 32;

    int warmup = 10, loops = 100;

    // Helper: run a benchmark
    auto bench = [&](const char* name, auto record_fn) {
        // Record
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd, &cbi);
        vkCmdResetQueryPool(cmd, qpool, 0, MTS);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descSet, 0, nullptr);
        int ts = 0;
        auto stamp = [&]() { vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, qpool, ts++); };

        stamp();
        record_fn(cmd, pipeLayout, stamp);
        stamp();

        vkEndCommandBuffer(cmd);

        // Warmup
        for (int i = 0; i < warmup; i++) {
            VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
            vkQueueSubmit(queue, 1, &si, fence); vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX); vkResetFences(device, 1, &fence);
        }

        // Benchmark
        double total = 0, min_ms = 1e9;
        for (int i = 0; i < loops; i++) {
            VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
            vkQueueSubmit(queue, 1, &si, fence); vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX); vkResetFences(device, 1, &fence);
            uint64_t t[MTS];
            vkGetQueryPoolResults(device, qpool, 0, ts, sizeof(t), t, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
            double ms = (t[ts-1] - t[0]) * props.limits.timestampPeriod / 1e6;
            total += ms;
            if (ms < min_ms) min_ms = ms;
        }
        double avg = total / loops;
        printf("  %-35s  avg: %7.3f ms  min: %7.3f ms\n", name, avg, min_ms);
        return avg;
    };

    printf("\n--- ISOLATED TESTS (single dispatch, no full pipeline) ---\n\n");

    int n_wg32 = (n_tokens + 255) / 256;

    // Test 1: FFN only, 1 tok/thread
    double t_ffn1 = bench("FFN-only (1 tok/thread, 32 WGs)", [&](VkCommandBuffer c, VkPipelineLayout pl, auto& stamp) {
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pFFN1);
        int32_t pc[] = {n_tokens, dim, ffn_w1, ffn_b1, ffn_w2, ffn_b2, f_in, f_out};
        vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(c, n_wg32, 1, 1);
    });

    // Test 2: FFN only, 4 tok/thread
    double t_ffn4 = bench("FFN-only (4 tok/thread, 8 WGs)", [&](VkCommandBuffer c, VkPipelineLayout pl, auto& stamp) {
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pFFN4);
        int32_t pc[] = {n_tokens, dim, ffn_w1, ffn_b1, ffn_w2, ffn_b2, f_in, f_out};
        vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(c, (n_tokens + 1023) / 1024, 1, 1);
    });

    // Test 3: QKV projection only
    double t_qkv = bench("QKV-only (3 matmuls, 32 WGs)", [&](VkCommandBuffer c, VkPipelineLayout pl, auto& stamp) {
        if (!pQKV) return;
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pQKV);
        int32_t pc[] = {n_tokens, dim, qkv_w, qkv_b, f_in, f_q, f_k, f_v};
        vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(c, n_wg32, 1, 1);
    });

    // Test 4: Out+FFN only
    double t_outffn = bench("Out+FFN-only (4 matmuls, 32 WGs)", [&](VkCommandBuffer c, VkPipelineLayout pl, auto& stamp) {
        if (!pOutFFN) return;
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pOutFFN);
        int32_t pc[] = {n_tokens, dim, out_w, out_b, ffn_w1, ffn_b1, ffn_w2, ffn_b2, f_attn, f_in, f_out};
        vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(c, n_wg32, 1, 1);
    });

    // Test 5: Barrier-only (measure pure barrier cost)
    bench("Barrier-only (4 barriers, no compute)", [&](VkCommandBuffer c, VkPipelineLayout pl, auto& stamp) {
        addBarrier(c); addBarrier(c); addBarrier(c); addBarrier(c);
    });

    // Test: FFN split into 4x coopvec<128>
    double t_ffnsplit = bench("FFN split 4x cv128 (32 WGs)", [&](VkCommandBuffer c, VkPipelineLayout pl, auto& stamp) {
        if (!pFFNSplit) return;
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pFFNSplit);
        int32_t pc[] = {n_tokens, dim, ffn_w1, ffn_b1, ffn_w2, ffn_b2, f_in, f_out};
        vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(c, n_wg32, 1, 1);
    });

    // Test: FFN with 2x coopvec<256>
    double t_ffncv256 = bench("FFN 2x cv256 (32 WGs)", [&](VkCommandBuffer c, VkPipelineLayout pl, auto& stamp) {
        if (!pFFNcv256) return;
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pFFNcv256);
        int32_t pc[] = {n_tokens, dim, ffn_w1, ffn_b1, ffn_w2, ffn_b2, f_in, f_out};
        vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(c, n_wg32, 1, 1);
    });

    // Test: FFN W1 only (isolate W1 vs W2)
    int f_hidden512 = f_out + n_tokens * 128;  // temp for 512-dim hidden
    bench("FFN W1-only 128->512 (32 WGs)", [&](VkCommandBuffer c, VkPipelineLayout pl, auto& stamp) {
        if (!pFFNW1) return;
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pFFNW1);
        int32_t pc[] = {n_tokens, dim, 512, ffn_w1, ffn_b1, f_in, f_hidden512};
        vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(c, n_wg32, 1, 1);
    });

    // Test: FFN W2 only
    bench("FFN W2-only 512->128 (32 WGs)", [&](VkCommandBuffer c, VkPipelineLayout pl, auto& stamp) {
        if (!pFFNW2) return;
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pFFNW2);
        int32_t pc[] = {n_tokens, dim, 512, ffn_w2, ffn_b2, f_hidden512, f_in, f_out};
        vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
        vkCmdDispatch(c, n_wg32, 1, 1);
    });

    // Test: FFN 2-dispatch (W1 + W2 with barrier)
    bench("FFN 2-dispatch W1+W2 (32 WGs each)", [&](VkCommandBuffer c, VkPipelineLayout pl, auto& stamp) {
        if (!pFFNW1 || !pFFNW2) return;
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pFFNW1);
        int32_t pc1[] = {n_tokens, dim, 512, ffn_w1, ffn_b1, f_in, f_hidden512};
        vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc1), pc1);
        vkCmdDispatch(c, n_wg32, 1, 1);
        addBarrier(c);
        vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pFFNW2);
        int32_t pc2[] = {n_tokens, dim, 512, ffn_w2, ffn_b2, f_hidden512, f_in, f_out};
        vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), pc2);
        vkCmdDispatch(c, n_wg32, 1, 1);
    });

    printf("\n--- FULL TRANSFORMER BLOCK TESTS (4 blocks each) ---\n\n");

    int ws8 = 8, wx8 = (r3.w + ws8 - 1) / ws8, wy8 = (r3.h + ws8 - 1) / ws8;
    int total_win8 = wx8 * wy8;

    // Test 6: Original windowed 8x8 (monolithic, 4 blocks)
    bench("Original windowed 8x8 (4 blocks)", [&](VkCommandBuffer c, VkPipelineLayout pl, auto& stamp) {
        int cur_in = f_in, cur_out = f_out;
        for (int b = 0; b < 4; b++) {
            vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pWin8);
            int32_t pc[] = {n_tokens, 128, 4, 32, r3.w, r3.h, 8,
                qkv_w, qkv_b, out_w, out_b, ffn_w1, ffn_b1, ffn_w2, ffn_b2,
                cur_in, cur_out};
            vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
            vkCmdDispatch(c, total_win8, 1, 1);
            addBarrier(c);
            int tmp = cur_in; cur_in = cur_out; cur_out = tmp;
        }
    });

    // Test 7: Split pipeline (QKV + Attn + OutFFN, 4 blocks)
    if (pQKV && pAttn && pOutFFN) {
        bench("Split pipeline (3 disp/block, 4 blk)", [&](VkCommandBuffer c, VkPipelineLayout pl, auto& stamp) {
            int cur_in = f_in, cur_out = f_out;
            for (int b = 0; b < 4; b++) {
                // QKV
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pQKV);
                int32_t pc1[] = {n_tokens, dim, qkv_w, qkv_b, cur_in, f_q, f_k, f_v};
                vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc1), pc1);
                vkCmdDispatch(c, n_wg32, 1, 1);
                addBarrier(c);

                // Windowed Attention
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pAttn);
                int32_t pc2[] = {n_tokens, 128, 4, 32, r3.w, r3.h, 8, f_q, f_k, f_v, f_attn};
                vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), pc2);
                vkCmdDispatch(c, total_win8, 1, 1);
                addBarrier(c);

                // Out + FFN
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pOutFFN);
                int32_t pc3[] = {n_tokens, dim, out_w, out_b, ffn_w1, ffn_b1, ffn_w2, ffn_b2,
                    f_attn, cur_in, cur_out};
                vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc3), pc3);
                vkCmdDispatch(c, n_wg32, 1, 1);
                addBarrier(c);

                int tmp = cur_in; cur_in = cur_out; cur_out = tmp;
            }
        });
    }

    // Test 8: 16x16 windowed (4 blocks)
    if (pWin16) {
        int ws16 = 16, wx16 = (r3.w + ws16 - 1) / ws16, wy16 = (r3.h + ws16 - 1) / ws16;
        int total_win16 = wx16 * wy16;
        bench("Windowed 16x16 (1 disp/block, 4 blk)", [&](VkCommandBuffer c, VkPipelineLayout pl, auto& stamp) {
            int cur_in = f_in, cur_out = f_out;
            for (int b = 0; b < 4; b++) {
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pWin16);
                int32_t pc[] = {n_tokens, 128, 4, 32, r3.w, r3.h, 16,
                    qkv_w, qkv_b, out_w, out_b, ffn_w1, ffn_b1, ffn_w2, ffn_b2,
                    cur_in, cur_out};
                vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
                vkCmdDispatch(c, total_win16, 1, 1);
                addBarrier(c);
                int tmp = cur_in; cur_in = cur_out; cur_out = tmp;
            }
        });
    }

    // Test 9: Linear attention 3-dispatch (4 blocks)
    if (pLinKV && pLinRed && pLinQFFN) {
        bench("Linear attn 3-disp (4 blocks)", [&](VkCommandBuffer c, VkPipelineLayout pl, auto& stamp) {
            int cur_in = f_in, cur_out = f_out;
            for (int b = 0; b < 4; b++) {
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pLinKV);
                int32_t pc1[] = {n_tokens, dim, k_w, k_b, v_w, v_b, cur_in, f_phiK, f_V};
                vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc1), pc1);
                vkCmdDispatch(c, n_wg32, 1, 1);
                addBarrier(c);

                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pLinRed);
                int32_t pc2[] = {n_tokens, 128, 4, 32, f_phiK, f_V, f_S, f_Z};
                vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), pc2);
                vkCmdDispatch(c, 128, 1, 1);
                addBarrier(c);

                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pLinQFFN);
                int32_t pc3[] = {n_tokens, 128, 4, 32, q_w, q_b, out_w, out_b,
                    ffn_w1, ffn_b1, ffn_w2, ffn_b2, cur_in, cur_out, f_S, f_Z};
                vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc3), pc3);
                vkCmdDispatch(c, n_wg32, 1, 1);
                addBarrier(c);

                int tmp = cur_in; cur_in = cur_out; cur_out = tmp;
            }
        });
    }

    // Test: OPTIMIZED windowed 8x8 with split FFN (4 blocks)
    if (pWinFast) {
        bench("FAST windowed 8x8 (4 blocks)", [&](VkCommandBuffer c, VkPipelineLayout pl, auto& stamp) {
            int cur_in = f_in, cur_out = f_out;
            for (int b = 0; b < 4; b++) {
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pWinFast);
                int32_t pc[] = {n_tokens, 128, 4, 32, r3.w, r3.h, 8,
                    qkv_w, qkv_b, out_w, out_b, ffn_w1, ffn_b1, ffn_w2, ffn_b2,
                    cur_in, cur_out};
                vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
                vkCmdDispatch(c, total_win8, 1, 1);
                addBarrier(c);
                int tmp = cur_in; cur_in = cur_out; cur_out = tmp;
            }
        });
    }

    // Test: 2-split (QKV+Attn | FFN-cv256) — 4 blocks
    if (pSplit2A && pSplit2F) {
        bench("2-SPLIT qkv+attn|ffn-cv256 (4 blk)", [&](VkCommandBuffer c, VkPipelineLayout pl, auto& stamp) {
            int cur_in = f_in, cur_out = f_out;
            for (int b = 0; b < 4; b++) {
                // Pass 1: QKV + Attention + OutProj (writes to f_attn)
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pSplit2A);
                int32_t pc1[] = {n_tokens, 128, 4, 32, r3.w, r3.h, 8,
                    qkv_w, qkv_b, out_w, out_b, cur_in, f_attn};
                vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc1), pc1);
                vkCmdDispatch(c, total_win8, 1, 1);
                addBarrier(c);

                // Pass 2: FFN with split coopvec<256> (reads f_attn, writes cur_out)
                vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, pSplit2F);
                int32_t pc2[] = {n_tokens, dim, ffn_w1, ffn_b1, ffn_w2, ffn_b2, f_attn, cur_out};
                vkCmdPushConstants(c, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), pc2);
                vkCmdDispatch(c, n_wg32, 1, 1);
                addBarrier(c);

                int tmp = cur_in; cur_in = cur_out; cur_out = tmp;
            }
        });
    }

    // Summary
    printf("\n--- ANALYSIS ---\n\n");
    double theoretical = 3.3 / 380.0;  // 3.3 GFLOP / 380 TOPS = ms per block
    printf("  Theoretical per block:     %7.4f ms  (3.3 GFLOP at 380 TOPS)\n", theoretical);
    printf("  FFN-only 1tok (per block): %7.3f ms  (tensor core ceiling)\n", t_ffn1);
    printf("  FFN-only 4tok (per block): %7.3f ms  (with weight caching)\n", t_ffn4);
    printf("  QKV-only (per block):      %7.3f ms\n", t_qkv);
    printf("  Out+FFN (per block):       %7.3f ms\n", t_outffn);
    printf("\n");
    printf("  Tensor core utilization (FFN-1tok): %.1f%%\n", theoretical / t_ffn1 * 100);
    printf("  Weight caching benefit (4tok/1tok): %.2fx\n", t_ffn1 / t_ffn4);
    printf("\n  These numbers tell us WHERE the gap is:\n");
    printf("  - If FFN-only is fast: overhead is in attention/barriers/dispatch\n");
    printf("  - If FFN-only is slow: coopVecMatMulAddNV itself is underperforming\n");
    printf("  - If 4tok >> 1tok: we're bandwidth-bound on weight loading\n");

    // Cleanup
    vkDeviceWaitIdle(device);
    vkDestroyQueryPool(device, qpool, nullptr);
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, cmdPool, nullptr);
    vkDestroyBuffer(device, wBuf, nullptr); vkFreeMemory(device, wMem, nullptr);
    vkDestroyBuffer(device, fBuf, nullptr); vkFreeMemory(device, fMem, nullptr);
    auto dp = [&](VkPipeline p) { if (p) vkDestroyPipeline(device, p, nullptr); };
    dp(pFFN1); dp(pFFN4); dp(pWin8); dp(pQKV); dp(pAttn); dp(pOutFFN);
    dp(pWin16); dp(pLinKV); dp(pLinRed); dp(pLinQFFN);
    dp(pWinFast); dp(pSplit2A); dp(pSplit2F); dp(pFFNSplit); dp(pFFNcv256); dp(pFFNW1); dp(pFFNW2);
    vkDestroyPipelineLayout(device, pipeLayout, nullptr);
    vkDestroyDescriptorPool(device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
