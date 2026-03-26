// Prism Linear Attention Benchmark
//
// Compares windowed attention vs linear attention at various scales.
// Linear attention uses 3 dispatches per block (KV project, reduce, Q+FFN)
// instead of 1 monolithic dispatch, giving better occupancy and GLOBAL attention.
//
// Linear attention replaces O(N^2) windowed softmax with O(N*d) global linear:
//   S_h = Σ φ(K_h)^T ⊗ V_h   (tiny 32x32 state matrix per head)
//   output_h = φ(Q_h) @ S_h / (φ(Q_h) · z_h)
//
// Configs tested:
//   Windowed: 4b/512, 8b/512, 4b/1024, 8b/1024
//   Linear:   4b/512, 8b/512, 12b/512, 16b/512, 4b/1024, 8b/1024, 12b/1024

#define VK_USE_PLATFORM_WIN32_KHR
#include "deps/volk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <fstream>
#include <chrono>

// Push constants
struct Conv3x3Push {
    int32_t in_channels, out_channels, width, height;
    int32_t weight_offset, bias_offset, input_offset, output_offset;
    int32_t relu;
};
struct StridedConvPush {
    int32_t in_channels, out_channels, in_width, in_height;
    int32_t out_width, out_height;
    int32_t weight_offset, bias_offset, input_offset, output_offset;
};
struct NNUpsamplePush {
    int32_t channels, in_width, in_height, out_width, out_height;
    int32_t input_offset, output_offset;
};
struct ConcatPush {
    int32_t ch_a, ch_b, width, height;
    int32_t offset_a, offset_b, output_offset;
};
struct PWConvPush {
    int32_t in_channels, out_channels, width, height;
    int32_t weight_offset, bias_offset, input_offset, output_offset;
    int32_t relu;
};
struct TransformerPush {
    int32_t n_tokens, dim, n_heads, head_dim;
    int32_t spatial_w, spatial_h, window_size;
    int32_t qkv_w_offset, qkv_b_offset;
    int32_t out_w_offset, out_b_offset;
    int32_t ffn_w1_offset, ffn_b1_offset;
    int32_t ffn_w2_offset, ffn_b2_offset;
    int32_t input_offset, output_offset;
};
struct KVPush {
    int32_t n_tokens, dim;
    int32_t w_k_offset, b_k_offset, w_v_offset, b_v_offset;
    int32_t input_offset, k_output_offset, v_output_offset;
};
struct FusedReducePush {
    int32_t n_tokens, dim, n_heads, head_dim;
    int32_t w_k_offset, b_k_offset, w_v_offset, b_v_offset;
    int32_t input_offset, s_offset, z_offset;
};
struct ReducePush {
    int32_t n_tokens, dim, n_heads, head_dim;
    int32_t k_offset, v_offset, s_offset, z_offset;
};
struct QFFNPush {
    int32_t n_tokens, dim, n_heads, head_dim;
    int32_t w_q_offset, b_q_offset;
    int32_t w_o_offset, b_o_offset;
    int32_t ffn_w1_offset, ffn_b1_offset;
    int32_t ffn_w2_offset, ffn_b2_offset;
    int32_t input_offset, output_offset;
    int32_t s_offset, z_offset;
};
struct PixelShufflePush {
    int32_t render_width, render_height, display_width, display_height;
    int32_t scale, input_offset, output_offset;
};

struct Resolution { int w, h; int pixels() const { return w * h; } };
struct WeightAlloc { int offset, count; };

// Helpers
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

struct ModelConfig {
    const char* name;
    const char* tag;
    int n_blocks;
    int dim;
    int n_heads;
    int head_dim;
    int ffn_hidden;
    int attn_mode;      // 0 = windowed, 1 = linear (3 dispatch), 2 = flash linear (2 dispatch fused)
};

struct BlockWeights {
    // For windowed: QKV fused
    WeightAlloc qkv_w, qkv_b, out_w, out_b;
    // For linear: separate K, V, Q
    WeightAlloc k_w, k_b, v_w, v_b, q_w, q_b;
    // Shared
    WeightAlloc ffn_w1, ffn_b1, ffn_w2, ffn_b2;
};

int main(int argc, char** argv) {
    printf("================================================================\n");
    printf("  Prism Linear Attention Benchmark\n");
    printf("  Windowed vs Linear attention comparison\n");
    printf("================================================================\n\n");

    int gpu_id = argc > 1 ? atoi(argv[1]) : 0;
    int warmup = 10, loops = 50;

    ModelConfig configs[] = {
        // Windowed baselines
        {"Win 4b/512",        "win-4b-512",    4, 128, 4, 32,  512, 0},
        {"Win 8b/512",        "win-8b-512",    8, 128, 4, 32,  512, 0},
        {"Win 4b/1024",       "win-4b-1024",   4, 128, 4, 32, 1024, 0},
        {"Win 8b/1024",       "win-8b-1024",   8, 128, 4, 32, 1024, 0},
        // Linear attention (3 dispatch — for reference)
        {"Lin3 4b/512",       "lin3-4b-512",   4, 128, 4, 32,  512, 1},
        {"Lin3 8b/512",       "lin3-8b-512",   8, 128, 4, 32,  512, 1},
        // Flash linear attention (2 dispatch — fused reduce)
        {"Flash 4b/512",      "fla-4b-512",    4, 128, 4, 32,  512, 2},
        {"Flash 8b/512",      "fla-8b-512",    8, 128, 4, 32,  512, 2},
        {"Flash 12b/512",     "fla-12b-512",  12, 128, 4, 32,  512, 2},
        {"Flash 16b/512",     "fla-16b-512",  16, 128, 4, 32,  512, 2},
        {"Flash 4b/1024",     "fla-4b-1024",   4, 128, 4, 32, 1024, 2},
        {"Flash 8b/1024",     "fla-8b-1024",   8, 128, 4, 32, 1024, 2},
        {"Flash 12b/1024",    "fla-12b-1024", 12, 128, 4, 32, 1024, 2},
        {"Flash 16b/1024",    "fla-16b-1024", 16, 128, 4, 32, 1024, 2},
        // Flash v2: fused reduce+QFFN (2 dispatches total: KV + fused)
        {"Fla2 4b/512",       "fl2-4b-512",    4, 128, 4, 32,  512, 3},
        {"Fla2 8b/512",       "fl2-8b-512",    8, 128, 4, 32,  512, 3},
        {"Fla2 12b/512",      "fl2-12b-512",  12, 128, 4, 32,  512, 3},
        {"Fla2 16b/512",      "fl2-16b-512",  16, 128, 4, 32,  512, 3},
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    // Vulkan init
    if (volkInitialize() != VK_SUCCESS) { printf("FATAL: No Vulkan\n"); return 1; }
    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo instInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instInfo.pApplicationInfo = &appInfo;
    VkInstance instance;
    vkCreateInstance(&instInfo, nullptr, &instance);
    volkLoadInstance(instance);

    uint32_t gpuCount = 0;
    vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr);
    std::vector<VkPhysicalDevice> gpus(gpuCount);
    vkEnumeratePhysicalDevices(instance, &gpuCount, gpus.data());
    if (gpu_id >= (int)gpuCount) gpu_id = 0;
    VkPhysicalDevice physical = gpus[gpu_id];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical, &props);
    printf("GPU: %s\n", props.deviceName);

    // Features
    VkPhysicalDeviceCooperativeVectorFeaturesNV coopVecFeat = {(VkStructureType)1000553000};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMat = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    coopMat.pNext = &coopVecFeat;
    VkPhysicalDeviceShaderFloat16Int8Features f16 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    f16.pNext = &coopMat;
    VkPhysicalDeviceSubgroupProperties sgprops = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    sgprops.pNext = &f16;
    VkPhysicalDevice16BitStorageFeatures s16 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    s16.pNext = &sgprops;
    VkPhysicalDeviceFeatures2 features2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.pNext = &s16;
    vkGetPhysicalDeviceFeatures2(physical, &features2);
    printf("Subgroup size: %d, operations: 0x%x\n", sgprops.subgroupSize, sgprops.supportedOperations);

    // Queue + device
    uint32_t qc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &qc, nullptr);
    std::vector<VkQueueFamilyProperties> qp(qc);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &qc, qp.data());
    uint32_t qf = 0;
    for (uint32_t i = 0; i < qc; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = qf; qi.queueCount = 1; qi.pQueuePriorities = &prio;
    const char* exts[] = {"VK_KHR_cooperative_matrix", "VK_NV_cooperative_vector"};
    VkDeviceCreateInfo di = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    di.pNext = &features2; di.enabledExtensionCount = 2; di.ppEnabledExtensionNames = exts;
    VkDevice device;
    vkCreateDevice(physical, &di, nullptr, &device);
    volkLoadDevice(device);
    VkQueue queue; vkGetDeviceQueue(device, qf, 0, &queue);

    // Cmd pool, buffer, fence, queries
    VkCommandPool cmdPool;
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = qf; cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device, &cpi, nullptr, &cmdPool);
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = cmdPool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &cai, &cmd);
    VkFence fence; VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(device, &fi, nullptr, &fence);
    const int MAX_TS = 64;
    VkQueryPool queryPool;
    VkQueryPoolCreateInfo qpi_ = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi_.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi_.queryCount = MAX_TS;
    vkCreateQueryPool(device, &qpi_, nullptr, &queryPool);

    // Descriptors + layout
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

    // Load pipelines
    printf("\nLoading shaders...\n");
    const char* sd = "shaders/";
    char path[512];
    #define LOAD(var, file) \
        snprintf(path, 512, "%s%s", sd, file); \
        VkPipeline var = loadPipeline(device, pipeLayout, path); \
        printf("  %-45s %s\n", file, var ? "OK" : "FAIL");

    LOAD(pipeInputConv,   "conv3x3_coopvec_9ch.spv");
    LOAD(pipeEnc1,        "strided_conv_coopvec_32ch.spv");
    LOAD(pipeEnc2,        "strided_conv_coopvec_64ch.spv");
    LOAD(pipeEnc3,        "strided_conv_coopvec_128ch.spv");
    // Windowed attention variants
    LOAD(pipeWinH512,     "attention_windowed.spv");
    LOAD(pipeWinH1024,    "attention_windowed_h1024.spv");
    // Linear attention (3-dispatch)
    LOAD(pipeLinKV,       "linear_attn_kv.spv");
    LOAD(pipeLinReduce,   "linear_attn_reduce.spv");
    LOAD(pipeLinQFFN512,  "linear_attn_qffn.spv");
    LOAD(pipeLinQFFN1024, "linear_attn_qffn_h1024.spv");
    // Flash linear attention (fused reduce — old approach)
    LOAD(pipeFusedReduce, "linear_attn_fused_reduce.spv");
    // Flash v2: fused reduce+QFFN (2-dispatch: KV + this)
    LOAD(pipeFusedQFFN512, "linear_attn_qffn_fused.spv");
    // Decoder
    LOAD(pipeNNUp,        "nn_upsample.spv");
    LOAD(pipeCat,         "concat_skip.spv");
    LOAD(pipePW256_128,   "pw_conv_coopvec_256to128.spv");
    LOAD(pipePW192_64,    "pw_conv_coopvec_192to64.spv");
    LOAD(pipePW96_32,     "pw_conv_coopvec_96to32.spv");
    LOAD(pipePW32_12,     "pw_conv_coopvec_32to12.spv");
    LOAD(pipePS,          "pixelshuffle_sigmoid.spv");
    #undef LOAD

    if (!pipeInputConv || !pipeEnc1 || !pipeEnc2 || !pipeEnc3 ||
        !pipeWinH512 || !pipeLinQFFN512 ||
        !pipeNNUp || !pipeCat || !pipePW256_128 || !pipePW192_64 ||
        !pipePW96_32 || !pipePW32_12 || !pipePS) {
        printf("FATAL: Missing critical shaders\n"); return 1;
    }

    Resolution r0={960,540}, r1={480,270}, r2={240,135}, r3={120,68}, rD={1920,1080};

    // Results
    struct Result { double enc_ms, trans_ms, dec_ms, out_ms, total_ms; int params; };
    std::vector<Result> results(n_configs);

    for (int c = 0; c < n_configs; c++) {
        auto& cfg = configs[c];
        printf("\n  [%s] %s ...\n", cfg.tag, cfg.name);

        // Check pipeline availability
        VkPipeline winPipe = VK_NULL_HANDLE, linQFFN = VK_NULL_HANDLE;
        if (cfg.attn_mode == 0) {
            winPipe = (cfg.ffn_hidden == 512) ? pipeWinH512 :
                      (cfg.ffn_hidden == 1024) ? pipeWinH1024 : VK_NULL_HANDLE;
            if (!winPipe) { printf("    SKIP (no shader)\n"); results[c].total_ms = -1; continue; }
        } else if (cfg.attn_mode == 3) {
            // Flash v2: fused reduce+QFFN
            linQFFN = (cfg.ffn_hidden == 512) ? pipeFusedQFFN512 : VK_NULL_HANDLE;
            if (!linQFFN) { printf("    SKIP (no fused shader)\n"); results[c].total_ms = -1; continue; }
        } else {
            linQFFN = (cfg.ffn_hidden == 512) ? pipeLinQFFN512 :
                      (cfg.ffn_hidden == 1024) ? pipeLinQFFN1024 : VK_NULL_HANDLE;
            if (!linQFFN) { printf("    SKIP (no shader)\n"); results[c].total_ms = -1; continue; }
        }

        // Weight allocation
        int wcur = 0;
        auto wa = [&](int n) -> WeightAlloc { WeightAlloc w = {wcur, n}; wcur += n; return w; };

        // Encoder
        auto w_ic_w = wa(32*9*9); auto w_ic_b = wa(32);
        auto w_e1_w = wa(64*32*9); auto w_e1_b = wa(64);
        auto w_e2_w = wa(128*64*9); auto w_e2_b = wa(128);
        auto w_e3_w = wa(128*128*9); auto w_e3_b = wa(128);

        // Transformer
        std::vector<BlockWeights> bw(cfg.n_blocks);
        for (int i = 0; i < cfg.n_blocks; i++) {
            if (cfg.attn_mode == 0) {
                bw[i].qkv_w = wa(3*128*128); bw[i].qkv_b = wa(3*128);
            } else {
                bw[i].k_w = wa(128*128); bw[i].k_b = wa(128);
                bw[i].v_w = wa(128*128); bw[i].v_b = wa(128);
                bw[i].q_w = wa(128*128); bw[i].q_b = wa(128);
            }
            bw[i].out_w = wa(128*128); bw[i].out_b = wa(128);
            bw[i].ffn_w1 = wa(cfg.ffn_hidden*128); bw[i].ffn_b1 = wa(cfg.ffn_hidden);
            bw[i].ffn_w2 = wa(128*cfg.ffn_hidden); bw[i].ffn_b2 = wa(128);
        }

        // Decoder
        auto w_d3w = wa(128*256); auto w_d3b = wa(128);
        auto w_d2w = wa(64*192); auto w_d2b = wa(64);
        auto w_d1w = wa(32*96); auto w_d1b = wa(32);
        auto w_ow = wa(12*32); auto w_ob = wa(12);

        int total_weights = wcur;
        results[c].params = total_weights;

        // Feature allocation
        int fcur = 0;
        auto fa = [&](int ch, Resolution r) -> int { int o = fcur; fcur += ch*r.pixels(); return o; };

        int f_inp = fa(9,r0), f_e0 = fa(32,r0), f_e1 = fa(64,r1), f_e2 = fa(128,r2);
        int f_e3 = fa(128,r3), f_tp = fa(128,r3);
        // Linear attention extra buffers
        int f_phiK = 0, f_V = 0, f_S = 0, f_Z = 0;
        if (cfg.attn_mode == 1 || cfg.attn_mode == 3) {
            // Modes 1 and 3 need intermediate K,V buffers
            f_phiK = fa(128,r3);
            f_V    = fa(128,r3);
        }
        if (cfg.attn_mode >= 1) {
            // S: n_heads * head_dim * head_dim, z: n_heads * head_dim
            int s_size = cfg.n_heads * cfg.head_dim * cfg.head_dim;
            int z_size = cfg.n_heads * cfg.head_dim;
            f_S = fcur; fcur += s_size;
            f_Z = fcur; fcur += z_size;
        }
        // Decoder buffers
        int f_d3u = fa(128,r2), f_d3c = fa(256,r2), f_d3o = fa(128,r2);
        int f_d2u = fa(128,r1), f_d2c = fa(192,r1), f_d2o = fa(64,r1);
        int f_d1u = fa(64,r0), f_d1c = fa(96,r0), f_d1o = fa(32,r0);
        int f_o12 = fa(12,r0), f_disp = fa(3,rD);
        int total_feat = fcur;

        // GPU buffers
        VkBuffer wBuf, fBuf; VkDeviceMemory wMem, fMem;
        createBuf(device, physical, (VkDeviceSize)total_weights*2,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, wBuf, wMem);
        createBuf(device, physical, (VkDeviceSize)total_feat*2,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, fBuf, fMem);

        VkDescriptorBufferInfo wbi={wBuf,0,VK_WHOLE_SIZE}, fbi={fBuf,0,VK_WHOLE_SIZE};
        VkWriteDescriptorSet wr[3] = {};
        wr[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,0,descSet,0,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,0,&wbi,0};
        wr[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,0,descSet,1,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,0,&fbi,0};
        wr[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,0,descSet,2,0,1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,0,&fbi,0};
        vkUpdateDescriptorSets(device, 3, wr, 0, nullptr);

        // Record command buffer
        int ts = 0;
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd, &cbi);
        vkCmdResetQueryPool(cmd, queryPool, 0, MAX_TS);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descSet, 0, nullptr);
        auto stamp = [&]() { if (ts < MAX_TS) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, ts++); };

        stamp(); // 0: start

        // === ENCODER ===
        #define ENC_DISPATCH(pipe, push_type, ...) { \
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe); \
            push_type p = __VA_ARGS__; \
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p); }

        { vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeInputConv);
          Conv3x3Push p={9,32,r0.w,r0.h,w_ic_w.offset,w_ic_b.offset,f_inp,f_e0,1};
          vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p);
          vkCmdDispatch(cmd,(r0.pixels()+255)/256,1,1); addBarrier(cmd); }
        { vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeEnc1);
          StridedConvPush p={32,64,r0.w,r0.h,r1.w,r1.h,w_e1_w.offset,w_e1_b.offset,f_e0,f_e1};
          vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p);
          vkCmdDispatch(cmd,(r1.pixels()+255)/256,1,1); addBarrier(cmd); }
        { vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeEnc2);
          StridedConvPush p={64,128,r1.w,r1.h,r2.w,r2.h,w_e2_w.offset,w_e2_b.offset,f_e1,f_e2};
          vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p);
          vkCmdDispatch(cmd,(r2.pixels()+255)/256,1,1); addBarrier(cmd); }
        { vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeEnc3);
          StridedConvPush p={128,128,r2.w,r2.h,r3.w,r3.h,w_e3_w.offset,w_e3_b.offset,f_e2,f_e3};
          vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p);
          vkCmdDispatch(cmd,(r3.pixels()+255)/256,1,1); addBarrier(cmd); }

        stamp(); // 1: after encoder

        // === TRANSFORMER ===
        int n_tokens = r3.pixels();
        if (cfg.attn_mode == 0) {
            // Windowed attention (monolithic dispatch per block)
            int ws = 8, wx = (r3.w+ws-1)/ws, wy = (r3.h+ws-1)/ws;
            int cur_in = f_e3, cur_out = f_tp;
            for (int b = 0; b < cfg.n_blocks; b++) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, winPipe);
                TransformerPush p = {};
                p.n_tokens=n_tokens; p.dim=128; p.n_heads=4; p.head_dim=32;
                p.spatial_w=r3.w; p.spatial_h=r3.h; p.window_size=ws;
                p.qkv_w_offset=bw[b].qkv_w.offset; p.qkv_b_offset=bw[b].qkv_b.offset;
                p.out_w_offset=bw[b].out_w.offset; p.out_b_offset=bw[b].out_b.offset;
                p.ffn_w1_offset=bw[b].ffn_w1.offset; p.ffn_b1_offset=bw[b].ffn_b1.offset;
                p.ffn_w2_offset=bw[b].ffn_w2.offset; p.ffn_b2_offset=bw[b].ffn_b2.offset;
                p.input_offset=cur_in; p.output_offset=cur_out;
                vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p);
                vkCmdDispatch(cmd, wx*wy, 1, 1);
                addBarrier(cmd);
                int tmp=cur_in; cur_in=cur_out; cur_out=tmp;
            }
            // If odd blocks, result in f_tp; even blocks in f_e3
            if (cfg.n_blocks % 2 != 0) f_e3 = f_tp;
        } else if (cfg.attn_mode == 1) {
            // Linear attention — 3 dispatches per block
            int cur_in = f_e3, cur_out = f_tp;
            int reduce_wgs = cfg.n_heads * cfg.head_dim;
            for (int b = 0; b < cfg.n_blocks; b++) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLinKV);
                KVPush kp = {n_tokens, 128, bw[b].k_w.offset, bw[b].k_b.offset,
                    bw[b].v_w.offset, bw[b].v_b.offset, cur_in, f_phiK, f_V};
                vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(kp),&kp);
                vkCmdDispatch(cmd, (n_tokens+255)/256, 1, 1); addBarrier(cmd);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLinReduce);
                ReducePush rp = {n_tokens, 128, 4, 32, f_phiK, f_V, f_S, f_Z};
                vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(rp),&rp);
                vkCmdDispatch(cmd, reduce_wgs, 1, 1); addBarrier(cmd);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, linQFFN);
                QFFNPush qp = {n_tokens, 128, 4, 32, bw[b].q_w.offset, bw[b].q_b.offset,
                    bw[b].out_w.offset, bw[b].out_b.offset,
                    bw[b].ffn_w1.offset, bw[b].ffn_b1.offset,
                    bw[b].ffn_w2.offset, bw[b].ffn_b2.offset,
                    cur_in, cur_out, f_S, f_Z};
                vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(qp),&qp);
                vkCmdDispatch(cmd, (n_tokens+255)/256, 1, 1); addBarrier(cmd);
                int tmp=cur_in; cur_in=cur_out; cur_out=tmp;
            }
            if (cfg.n_blocks % 2 != 0) f_e3 = f_tp;
        } else {
            // Flash linear attention — 2 dispatches per block (fused KV+reduce)
            int cur_in = f_e3, cur_out = f_tp;
            int reduce_wgs = cfg.n_heads * cfg.head_dim; // 128
            for (int b = 0; b < cfg.n_blocks; b++) {
                // Dispatch 1: Fused KV projection + S reduction
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeFusedReduce);
                FusedReducePush fp = {n_tokens, 128, 4, 32,
                    bw[b].k_w.offset, bw[b].k_b.offset,
                    bw[b].v_w.offset, bw[b].v_b.offset,
                    cur_in, f_S, f_Z};
                vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(fp),&fp);
                vkCmdDispatch(cmd, reduce_wgs, 1, 1); addBarrier(cmd);

                // Dispatch 2: Q + attention output + FFN
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, linQFFN);
                QFFNPush qp = {n_tokens, 128, 4, 32, bw[b].q_w.offset, bw[b].q_b.offset,
                    bw[b].out_w.offset, bw[b].out_b.offset,
                    bw[b].ffn_w1.offset, bw[b].ffn_b1.offset,
                    bw[b].ffn_w2.offset, bw[b].ffn_b2.offset,
                    cur_in, cur_out, f_S, f_Z};
                vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(qp),&qp);
                vkCmdDispatch(cmd, (n_tokens+255)/256, 1, 1); addBarrier(cmd);
                int tmp=cur_in; cur_in=cur_out; cur_out=tmp;
            }
            if (cfg.n_blocks % 2 != 0) f_e3 = f_tp;
        }

        if (cfg.attn_mode == 3) {
            // Flash v2: KV dispatch + fused reduce+QFFN dispatch (2 per block)
            int cur_in = f_e3, cur_out = f_tp;
            int n_wgs = (n_tokens + 255) / 256;  // ~32 workgroups

            for (int b = 0; b < cfg.n_blocks; b++) {
                // Dispatch 1: KV project (same as mode 1)
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLinKV);
                KVPush kp = {n_tokens, 128, bw[b].k_w.offset, bw[b].k_b.offset,
                    bw[b].v_w.offset, bw[b].v_b.offset, cur_in, f_phiK, f_V};
                vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(kp),&kp);
                vkCmdDispatch(cmd, n_wgs, 1, 1); addBarrier(cmd);

                // Dispatch 2: Fused reduce + Q + FFN
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, linQFFN);
                QFFNPush qp = {n_tokens, 128, 4, 32, bw[b].q_w.offset, bw[b].q_b.offset,
                    bw[b].out_w.offset, bw[b].out_b.offset,
                    bw[b].ffn_w1.offset, bw[b].ffn_b1.offset,
                    bw[b].ffn_w2.offset, bw[b].ffn_b2.offset,
                    cur_in, cur_out, f_phiK, f_V};  // k_offset=f_phiK, v_offset=f_V via s_offset/z_offset
                vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(qp),&qp);
                vkCmdDispatch(cmd, n_wgs, 1, 1); addBarrier(cmd);

                int tmp=cur_in; cur_in=cur_out; cur_out=tmp;
            }
            if (cfg.n_blocks % 2 != 0) f_e3 = f_tp;
        }

        stamp(); // 2: after transformer

        // === DECODER ===
        // dec3
        { vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeNNUp);
          NNUpsamplePush p={128,r3.w,r3.h,r2.w,r2.h,f_e3,f_d3u};
          vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p);
          vkCmdDispatch(cmd,(128*r3.pixels()+255)/256,1,1); addBarrier(cmd); }
        { vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeCat);
          ConcatPush p={128,128,r2.w,r2.h,f_d3u,f_e2,f_d3c};
          vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p);
          vkCmdDispatch(cmd,(256*r2.pixels()+255)/256,1,1); addBarrier(cmd); }
        { vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipePW256_128);
          PWConvPush p={256,128,r2.w,r2.h,w_d3w.offset,w_d3b.offset,f_d3c,f_d3o,1};
          vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p);
          vkCmdDispatch(cmd,(r2.pixels()+255)/256,1,1); addBarrier(cmd); }
        // dec2
        { vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeNNUp);
          NNUpsamplePush p={128,r2.w,r2.h,r1.w,r1.h,f_d3o,f_d2u};
          vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p);
          vkCmdDispatch(cmd,(128*r2.pixels()+255)/256,1,1); addBarrier(cmd); }
        { vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeCat);
          ConcatPush p={128,64,r1.w,r1.h,f_d2u,f_e1,f_d2c};
          vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p);
          vkCmdDispatch(cmd,(192*r1.pixels()+255)/256,1,1); addBarrier(cmd); }
        { vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipePW192_64);
          PWConvPush p={192,64,r1.w,r1.h,w_d2w.offset,w_d2b.offset,f_d2c,f_d2o,1};
          vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p);
          vkCmdDispatch(cmd,(r1.pixels()+255)/256,1,1); addBarrier(cmd); }
        // dec1
        { vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeNNUp);
          NNUpsamplePush p={64,r1.w,r1.h,r0.w,r0.h,f_d2o,f_d1u};
          vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p);
          vkCmdDispatch(cmd,(64*r1.pixels()+255)/256,1,1); addBarrier(cmd); }
        { vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipeCat);
          ConcatPush p={64,32,r0.w,r0.h,f_d1u,f_e0,f_d1c};
          vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p);
          vkCmdDispatch(cmd,(96*r0.pixels()+255)/256,1,1); addBarrier(cmd); }
        { vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipePW96_32);
          PWConvPush p={96,32,r0.w,r0.h,w_d1w.offset,w_d1b.offset,f_d1c,f_d1o,1};
          vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p);
          vkCmdDispatch(cmd,(r0.pixels()+255)/256,1,1); addBarrier(cmd); }

        stamp(); // 3: after decoder

        // === OUTPUT ===
        { vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipePW32_12);
          PWConvPush p={32,12,r0.w,r0.h,w_ow.offset,w_ob.offset,f_d1o,f_o12,0};
          vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p);
          vkCmdDispatch(cmd,(r0.pixels()+255)/256,1,1); addBarrier(cmd); }
        { vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,pipePS);
          PixelShufflePush p={r0.w,r0.h,rD.w,rD.h,2,f_o12,f_disp};
          vkCmdPushConstants(cmd,pipeLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(p),&p);
          vkCmdDispatch(cmd,(rD.w+15)/16,(rD.h+15)/16,1); addBarrier(cmd); }

        stamp(); // 4: end

        vkEndCommandBuffer(cmd);

        // Warmup + benchmark
        for (int i = 0; i < warmup; i++) {
            VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cmd;
            vkQueueSubmit(queue,1,&si,fence); vkWaitForFences(device,1,&fence,VK_TRUE,UINT64_MAX); vkResetFences(device,1,&fence);
        }
        double enc_t=0, trans_t=0, dec_t=0, out_t=0, tot_t=0;
        for (int i = 0; i < loops; i++) {
            VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount=1; si.pCommandBuffers=&cmd;
            vkQueueSubmit(queue,1,&si,fence); vkWaitForFences(device,1,&fence,VK_TRUE,UINT64_MAX); vkResetFences(device,1,&fence);
            uint64_t t[MAX_TS];
            vkGetQueryPoolResults(device,queryPool,0,ts,sizeof(t),t,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT);
            double ns = props.limits.timestampPeriod;
            enc_t += (t[1]-t[0])*ns/1e6; trans_t += (t[2]-t[1])*ns/1e6;
            dec_t += (t[3]-t[2])*ns/1e6; out_t += (t[4]-t[3])*ns/1e6;
            tot_t += (t[4]-t[0])*ns/1e6;
        }
        results[c] = {enc_t/loops, trans_t/loops, dec_t/loops, out_t/loops, tot_t/loops, total_weights};
        printf("    Enc: %.3fms  Trans: %.3fms  Dec: %.3fms  Out: %.3fms  TOTAL: %.3fms (%.0f FPS)\n",
               results[c].enc_ms, results[c].trans_ms, results[c].dec_ms, results[c].out_ms,
               results[c].total_ms, 1000.0/results[c].total_ms);

        vkDeviceWaitIdle(device);
        vkDestroyBuffer(device,wBuf,nullptr); vkFreeMemory(device,wMem,nullptr);
        vkDestroyBuffer(device,fBuf,nullptr); vkFreeMemory(device,fMem,nullptr);
    }

    // === FINAL TABLE ===
    printf("\n\n================================================================\n");
    printf("  RESULTS — Windowed vs Linear Attention\n");
    printf("  GPU: %s | 540x960→1080x1920\n", props.deviceName);
    printf("================================================================\n\n");

    printf("  %-18s %7s %4s %3s | %7s %7s %7s | %7s %5s\n",
           "Config", "Params", "FFN", "Blk", "Enc", "Trans", "Dec", "Total", "FPS");
    printf("  %-18s %7s %4s %3s | %7s %7s %7s | %7s %5s\n",
           "------------------", "-------", "----", "---", "-------", "-------", "-------", "-------", "-----");

    for (int c = 0; c < n_configs; c++) {
        auto& cfg = configs[c]; auto& r = results[c];
        if (r.total_ms < 0) continue;
        printf("  %-18s %6.1fK %4d %3d | %6.2fms %6.2fms %6.2fms | %6.2fms %5.0f\n",
               cfg.tag, r.params/1000.0, cfg.ffn_hidden, cfg.n_blocks,
               r.enc_ms, r.trans_ms, r.dec_ms, r.total_ms, 1000.0/r.total_ms);
    }

    // Compare windowed vs flash linear at same config
    printf("\n  Windowed vs Flash Linear (same blocks/FFN):\n");
    for (int c = 0; c < n_configs; c++) {
        if (configs[c].attn_mode != 0 || results[c].total_ms < 0) continue;
        for (int l = 0; l < n_configs; l++) {
            if (configs[l].attn_mode != 2 || results[l].total_ms < 0) continue;
            if (configs[l].n_blocks == configs[c].n_blocks && configs[l].ffn_hidden == configs[c].ffn_hidden) {
                double speedup = results[c].trans_ms / results[l].trans_ms;
                printf("  %db/FFN%d: Win %.2fms vs Flash %.2fms (trans: %.2fx %s)\n",
                       configs[c].n_blocks, configs[c].ffn_hidden,
                       results[c].total_ms, results[l].total_ms,
                       speedup, speedup > 1.0 ? "FASTER" : "slower");
            }
        }
    }

    printf("\n  Budget (16.67ms = 60 FPS):\n");
    for (int c = 0; c < n_configs; c++) {
        if (results[c].total_ms < 0) continue;
        bool ok = results[c].total_ms < 16.67;
        printf("  %-18s %6.2fms  %s\n", configs[c].tag, results[c].total_ms,
               ok ? "OK 60fps" : "OVER");
    }

    printf("\n================================================================\n");

    // Cleanup
    vkDeviceWaitIdle(device);
    vkDestroyQueryPool(device,queryPool,nullptr); vkDestroyFence(device,fence,nullptr);
    vkDestroyCommandPool(device,cmdPool,nullptr);
    auto dp = [&](VkPipeline p) { if (p) vkDestroyPipeline(device,p,nullptr); };
    dp(pipeInputConv); dp(pipeEnc1); dp(pipeEnc2); dp(pipeEnc3);
    dp(pipeWinH512); dp(pipeWinH1024);
    dp(pipeLinKV); dp(pipeLinReduce); dp(pipeFusedReduce); dp(pipeFusedQFFN512); dp(pipeLinQFFN512); dp(pipeLinQFFN1024);
    dp(pipeNNUp); dp(pipeCat); dp(pipePW256_128); dp(pipePW192_64); dp(pipePW96_32);
    dp(pipePW32_12); dp(pipePS);
    vkDestroyPipelineLayout(device,pipeLayout,nullptr);
    vkDestroyDescriptorPool(device,descPool,nullptr);
    vkDestroyDescriptorSetLayout(device,descLayout,nullptr);
    vkDestroyDevice(device,nullptr); vkDestroyInstance(instance,nullptr);
    return 0;
}
