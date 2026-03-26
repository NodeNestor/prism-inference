// Prism Scaling Benchmark — Compare model configurations
//
// Tests 6 configurations with identical encoder/decoder but different transformer scaling:
//
//   Config A "baseline":      4 blocks, dim=128, FFN hidden=512   (~1.0M transformer)
//   Config B "deep":          8 blocks, dim=128, FFN hidden=512   (~2.1M transformer)
//   Config C "wide-ffn":      4 blocks, dim=128, FFN hidden=1024  (~1.6M transformer)
//   Config D "deep+wide":     8 blocks, dim=128, FFN hidden=1024  (~3.2M transformer)
//   Config E "super-wide":    4 blocks, dim=128, FFN hidden=2048  (~2.9M transformer)
//   Config F "monster":       8 blocks, dim=128, FFN hidden=2048  (~5.8M transformer)
//
// All configs share the same U-Net encoder/decoder (~1.1M params).
// Encoder:  conv3x3 9->32, strided 32->64, 64->128, 128->128
// Decoder:  PW 256->128, 192->64, 96->32, 32->12 + PixelShuffle
// Resolution: 540x960 -> 1080x1920 (2x upscale via PixelShuffle)

#define VK_USE_PLATFORM_WIN32_KHR
#include "deps/volk.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <fstream>
#include <chrono>

// ============================================================================
// Push constant structs (must match shader layouts)
// ============================================================================

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
    int32_t channels;
    int32_t in_width, in_height;
    int32_t out_width, out_height;
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

struct PixelShufflePush {
    int32_t render_width, render_height, display_width, display_height;
    int32_t scale, input_offset, output_offset;
};

// ============================================================================
// Resolution
// ============================================================================

struct Resolution {
    int w, h;
    int pixels() const { return w * h; }
};

// ============================================================================
// Weight allocation
// ============================================================================

struct WeightAlloc {
    int offset, count;
    WeightAlloc() : offset(0), count(0) {}
    WeightAlloc(int off, int cnt) : offset(off), count(cnt) {}
};

// ============================================================================
// Model configuration
// ============================================================================

struct ModelConfig {
    const char* name;
    const char* tag;
    int n_blocks;       // transformer blocks
    int dim;            // transformer dim (always 128 for now)
    int n_heads;        // attention heads
    int head_dim;       // per-head dim
    int ffn_hidden;     // FFN hidden size
    int window_size;    // attention window
};

// ============================================================================
// Per-block transformer weights
// ============================================================================

struct TransformerBlockWeights {
    WeightAlloc qkv_w, qkv_b, out_w, out_b;
    WeightAlloc ffn_w1, ffn_b1, ffn_w2, ffn_b2;
};

// ============================================================================
// Helpers
// ============================================================================

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
    cpci.stage.module = mod;
    cpci.stage.pName = "main";
    cpci.layout = layout;

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

static void createBuf(VkDevice device, VkPhysicalDevice phys,
                       VkDeviceSize size, VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags memProps,
                       VkBuffer& buf, VkDeviceMemory& mem) {
    VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size; bi.usage = usage;
    vkCreateBuffer(device, &bi, nullptr, &buf);
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(device, buf, &mr);
    VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = findMem(phys, mr.memoryTypeBits, memProps);
    vkAllocateMemory(device, &ai, nullptr, &mem);
    vkBindBufferMemory(device, buf, mem, 0);
}

static void addBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
}

// ============================================================================
// Print architecture diagram for a config
// ============================================================================

static void printArchitecture(const ModelConfig& cfg, int encoder_params, int decoder_params) {
    int dim = cfg.dim;
    int hid = cfg.ffn_hidden;
    int attn_per_block = 3*dim*dim + 3*dim + dim*dim + dim;
    int ffn_per_block = hid*dim + hid + dim*hid + dim;
    int per_block = attn_per_block + ffn_per_block;
    int transformer_params = per_block * cfg.n_blocks;
    int total = encoder_params + transformer_params + decoder_params;

    printf("\n");
    printf("  +============================================================+\n");
    printf("  |  %-56s  |\n", cfg.name);
    printf("  |  tag: %-50s  |\n", cfg.tag);
    printf("  +============================================================+\n");
    printf("  |                                                            |\n");
    printf("  |  INPUT: 540x960 RGB+Depth+MV (9ch padded)                 |\n");
    printf("  |    |                                                       |\n");
    printf("  |  ENCODER (shared across all configs):                      |\n");
    printf("  |    | conv3x3 9->32   @ 540x960   coopvec<81>              |\n");
    printf("  |    | stride2 32->64  @ 270x480   coopvec<288>             |\n");
    printf("  |    | stride2 64->128 @ 135x240   coopvec<576>             |\n");
    printf("  |    | stride2 128->128@ 68x120    coopvec<1152>            |\n");
    printf("  |    |                                         [%6.1fK]     |\n", encoder_params/1000.0);
    printf("  |    v                                                       |\n");
    printf("  |  TRANSFORMER @ 68x120 (%d tokens, 8x8 windows):          |\n", 68*120);
    printf("  |    | dim=%-3d  heads=%-2d  head_dim=%-2d                      |\n", dim, cfg.n_heads, cfg.head_dim);
    printf("  |    | FFN hidden=%-4d  (%.1fx expansion)                    |\n", hid, (float)hid/dim);
    printf("  |    | blocks=%-2d                                             |\n", cfg.n_blocks);
    printf("  |    |                                                       |\n");
    printf("  |    | Per block:                                            |\n");
    printf("  |    |   QKV proj: %dx%d -> %d  [%6d params]              |\n", dim, dim, 3*dim, 3*dim*dim + 3*dim);
    printf("  |    |   Attention: %d tokens x %d heads (windowed)         |\n", 64, cfg.n_heads);
    printf("  |    |   Out proj: %dx%d                [%6d params]       |\n", dim, dim, dim*dim + dim);
    printf("  |    |   FFN: %d->%d->%d (GELU)       [%6d params]       |\n", dim, hid, dim, hid*dim + hid + dim*hid + dim);
    printf("  |    |   Total per block:              [%6d params]       |\n", per_block);
    printf("  |    |                                                       |\n");
    printf("  |    | x%d blocks                          [%7.1fK total]  |\n", cfg.n_blocks, transformer_params/1000.0);
    printf("  |    v                                                       |\n");
    printf("  |  DECODER (shared across all configs):                      |\n");
    printf("  |    | up 2x + cat(enc2) -> PW 256->128 @ 135x240           |\n");
    printf("  |    | up 2x + cat(enc1) -> PW 192->64  @ 270x480           |\n");
    printf("  |    | up 2x + cat(enc0) -> PW 96->32   @ 540x960           |\n");
    printf("  |    |                                         [%6.1fK]     |\n", decoder_params/1000.0);
    printf("  |    v                                                       |\n");
    printf("  |  OUTPUT: PW 32->12 + PixelShuffle(2x) + Sigmoid           |\n");
    printf("  |    -> 1080x1920 RGB                                        |\n");
    printf("  |                                                            |\n");
    printf("  |  TOTAL: %d params (%.2f MB fp16)                      |\n", total, total * 2.0 / 1024 / 1024);
    printf("  +============================================================+\n");
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    printf("================================================================\n");
    printf("  Prism Scaling Benchmark\n");
    printf("  Compare 6 transformer configurations\n");
    printf("  Same encoder/decoder, different transformer scaling\n");
    printf("================================================================\n\n");

    int gpu_id = argc > 1 ? atoi(argv[1]) : 0;
    int warmup = 10;
    int loops = 50;

    // ---- Define configs ----
    // Grouped by scaling strategy:
    //   - Block scaling (depth): same FFN, more blocks
    //   - FFN scaling (width): same blocks, wider FFN
    //   - Combined: both wider and deeper
    //   - Extreme: push the transformer to see where it breaks
    ModelConfig configs[] = {
        // --- Baselines ---
        {"Baseline",              "baseline",    4, 128, 4, 32,  512, 8},

        // --- Depth scaling (more blocks, same FFN) ---
        {"Deep 8blk",            "deep-8",       8, 128, 4, 32,  512, 8},
        {"Deep 12blk",           "deep-12",     12, 128, 4, 32,  512, 8},
        {"Deep 16blk",           "deep-16",     16, 128, 4, 32,  512, 8},

        // --- Width scaling (wider FFN, same blocks) ---
        {"Wide FFN 1024",        "wide-1024",    4, 128, 4, 32, 1024, 8},
        {"Wide FFN 2048",        "wide-2048",    4, 128, 4, 32, 2048, 8},

        // --- Combined depth + width ---
        {"8blk + FFN 1024",      "8b+1024",      8, 128, 4, 32, 1024, 8},
        {"8blk + FFN 2048",      "8b+2048",      8, 128, 4, 32, 2048, 8},
        {"12blk + FFN 1024",     "12b+1024",    12, 128, 4, 32, 1024, 8},
        {"12blk + FFN 2048",     "12b+2048",    12, 128, 4, 32, 2048, 8},

        // --- Extreme (pushing limits) ---
        {"16blk + FFN 2048",     "16b+2048",    16, 128, 4, 32, 2048, 8},
        {"24blk + FFN 2048",     "24b+2048",    24, 128, 4, 32, 2048, 8},
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    // ---- Vulkan init ----
    if (volkInitialize() != VK_SUCCESS) { printf("FATAL: No Vulkan\n"); return 1; }

    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.apiVersion = VK_API_VERSION_1_3;
    appInfo.pApplicationName = "PrismScalingBench";
    VkInstanceCreateInfo instInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instInfo.pApplicationInfo = &appInfo;
    VkInstance instance;
    if (vkCreateInstance(&instInfo, nullptr, &instance) != VK_SUCCESS) {
        printf("FATAL: vkCreateInstance failed\n"); return 1;
    }
    volkLoadInstance(instance);

    uint32_t gpuCount = 0;
    vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr);
    if (gpuCount == 0) { printf("FATAL: No GPUs\n"); return 1; }
    std::vector<VkPhysicalDevice> gpus(gpuCount);
    vkEnumeratePhysicalDevices(instance, &gpuCount, gpus.data());

    if (gpu_id >= (int)gpuCount) gpu_id = 0;
    VkPhysicalDevice physical = gpus[gpu_id];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical, &props);
    printf("GPU: %s\n", props.deviceName);

    // Feature chain
    VkPhysicalDeviceCooperativeVectorFeaturesNV coopVecFeat = {
        (VkStructureType)1000553000};
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMat = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    coopMat.pNext = &coopVecFeat;
    VkPhysicalDeviceShaderFloat16Int8Features f16feat = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    f16feat.pNext = &coopMat;
    VkPhysicalDevice16BitStorageFeatures s16feat = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    s16feat.pNext = &f16feat;
    VkPhysicalDeviceFeatures2 features2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.pNext = &s16feat;
    vkGetPhysicalDeviceFeatures2(physical, &features2);

    if (!f16feat.shaderFloat16 || !s16feat.storageBuffer16BitAccess) {
        printf("FATAL: GPU lacks FP16 support\n"); return 1;
    }

    // Queue
    uint32_t qc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &qc, nullptr);
    std::vector<VkQueueFamilyProperties> qp(qc);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &qc, qp.data());
    uint32_t qf = 0;
    for (uint32_t i = 0; i < qc; i++) {
        if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
    }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = qf; qi.queueCount = 1; qi.pQueuePriorities = &prio;

    const char* exts[] = {"VK_KHR_cooperative_matrix", "VK_NV_cooperative_vector"};
    VkDeviceCreateInfo di = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    di.pNext = &features2;
    di.enabledExtensionCount = 2; di.ppEnabledExtensionNames = exts;

    VkDevice device;
    if (vkCreateDevice(physical, &di, nullptr, &device) != VK_SUCCESS) {
        printf("FATAL: vkCreateDevice failed\n"); return 1;
    }
    volkLoadDevice(device);
    VkQueue queue;
    vkGetDeviceQueue(device, qf, 0, &queue);

    // Command pool, buffer, fence
    VkCommandPool cmdPool;
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = qf; cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device, &cpi, nullptr, &cmdPool);

    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = cmdPool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &cai, &cmd);

    VkFence fence;
    VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(device, &fi, nullptr, &fence);

    const int MAX_TS = 64;
    VkQueryPool queryPool;
    VkQueryPoolCreateInfo qpi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount = MAX_TS;
    vkCreateQueryPool(device, &qpi, nullptr, &queryPool);

    // Descriptor set + pipeline layout
    VkDescriptorSetLayoutBinding bindings[3] = {};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 3; dli.pBindings = bindings;
    VkDescriptorSetLayout descLayout;
    vkCreateDescriptorSetLayout(device, &dli, nullptr, &descLayout);

    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &descLayout;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pipeLayout;
    vkCreatePipelineLayout(device, &pli, nullptr, &pipeLayout);

    VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps;
    VkDescriptorPool descPool;
    vkCreateDescriptorPool(device, &dpci, nullptr, &descPool);

    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = descPool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &descLayout;
    VkDescriptorSet descSet;
    vkAllocateDescriptorSets(device, &dsai, &descSet);

    // ---- Load shader pipelines ----
    printf("\nLoading shaders...\n");
    const char* sd = "shaders/";
    char path[512];

    #define LOAD(var, file) \
        snprintf(path, 512, "%s%s", sd, file); \
        VkPipeline var = loadPipeline(device, pipeLayout, path); \
        printf("  %-45s %s\n", file, var ? "OK" : "FAIL");

    // Encoder (shared)
    LOAD(pipeInputConv,   "conv3x3_coopvec_9ch.spv");
    LOAD(pipeEnc1,        "strided_conv_coopvec_32ch.spv");
    LOAD(pipeEnc2,        "strided_conv_coopvec_64ch.spv");
    LOAD(pipeEnc3,        "strided_conv_coopvec_128ch.spv");

    // Transformer variants (one per FFN hidden size)
    LOAD(pipeTransH512,   "attention_windowed.spv");
    LOAD(pipeTransH1024,  "attention_windowed_h1024.spv");
    LOAD(pipeTransH2048,  "attention_windowed_h2048.spv");

    // Decoder (shared)
    LOAD(pipeNNUpsample,  "nn_upsample.spv");
    LOAD(pipeConcatSkip,  "concat_skip.spv");
    LOAD(pipePW256to128,  "pw_conv_coopvec_256to128.spv");
    LOAD(pipePW192to64,   "pw_conv_coopvec_192to64.spv");
    LOAD(pipePW96to32,    "pw_conv_coopvec_96to32.spv");
    LOAD(pipePW32to12,    "pw_conv_coopvec_32to12.spv");
    LOAD(pipePixelShuffle,"pixelshuffle_sigmoid.spv");
    #undef LOAD

    // Validate critical pipelines
    if (!pipeInputConv || !pipeEnc1 || !pipeEnc2 || !pipeEnc3 ||
        !pipeTransH512 || !pipeNNUpsample || !pipeConcatSkip ||
        !pipePW256to128 || !pipePW192to64 || !pipePW96to32 ||
        !pipePW32to12 || !pipePixelShuffle) {
        printf("\nFATAL: Missing critical shader(s)\n");
        return 1;
    }

    // Map FFN hidden size to pipeline
    auto getTransformerPipe = [&](int ffn_hidden) -> VkPipeline {
        if (ffn_hidden == 512)  return pipeTransH512;
        if (ffn_hidden == 1024) return pipeTransH1024;
        if (ffn_hidden == 2048) return pipeTransH2048;
        return VK_NULL_HANDLE;
    };

    // Resolution plan (same for all configs)
    Resolution r0 = {960, 540};
    Resolution r1 = {480, 270};
    Resolution r2 = {240, 135};
    Resolution r3 = {120, 68};
    Resolution rD = {1920, 1080};

    // ================================================================
    // Compute encoder/decoder param counts (shared, constant)
    // ================================================================

    int enc_params = (32*9*9 + 32) + (64*32*9 + 64) + (128*64*9 + 128) + (128*128*9 + 128);
    int dec_params = (128*256 + 128) + (64*192 + 64) + (32*96 + 32) + (12*32 + 12);

    // ================================================================
    // Print all architectures first
    // ================================================================

    printf("\n================================================================\n");
    printf("  MODEL ARCHITECTURES\n");
    printf("================================================================\n");
    for (int c = 0; c < n_configs; c++) {
        printArchitecture(configs[c], enc_params, dec_params);
    }

    // ================================================================
    // Print parameter comparison table
    // ================================================================

    printf("\n================================================================\n");
    printf("  PARAMETER COMPARISON\n");
    printf("================================================================\n\n");
    printf("  %-25s %6s %6s %8s %8s %8s\n",
           "Config", "Blocks", "FFN", "Trans.", "Enc+Dec", "TOTAL");
    printf("  %-25s %6s %6s %8s %8s %8s\n",
           "-------------------------", "------", "------", "--------", "--------", "--------");

    for (int c = 0; c < n_configs; c++) {
        auto& cfg = configs[c];
        int attn_per = 3*cfg.dim*cfg.dim + 3*cfg.dim + cfg.dim*cfg.dim + cfg.dim;
        int ffn_per = cfg.ffn_hidden*cfg.dim + cfg.ffn_hidden + cfg.dim*cfg.ffn_hidden + cfg.dim;
        int trans_total = (attn_per + ffn_per) * cfg.n_blocks;
        int total = enc_params + trans_total + dec_params;
        printf("  %-25s %6d %6d %7.1fK %7.1fK %7.1fK\n",
               cfg.tag, cfg.n_blocks, cfg.ffn_hidden,
               trans_total/1000.0, (enc_params+dec_params)/1000.0, total/1000.0);
    }

    // ================================================================
    // Run benchmarks for each config
    // ================================================================

    printf("\n================================================================\n");
    printf("  BENCHMARKING (warmup=%d, loops=%d)\n", warmup, loops);
    printf("================================================================\n");

    struct BenchResult {
        double encoder_ms, transformer_ms, decoder_ms, output_ms, total_ms;
        int total_params;
    };
    BenchResult results[12];

    for (int c = 0; c < n_configs; c++) {
        auto& cfg = configs[c];
        VkPipeline transPipe = getTransformerPipe(cfg.ffn_hidden);

        if (!transPipe) {
            printf("\n  [%s] SKIPPED — no shader for FFN hidden=%d\n", cfg.tag, cfg.ffn_hidden);
            results[c] = {-1, -1, -1, -1, -1, 0};
            continue;
        }

        printf("\n  Running: %s (%s)...\n", cfg.name, cfg.tag);

        // --- Weight allocation ---
        int wcur = 0;
        auto walloc = [&](int n) -> WeightAlloc {
            WeightAlloc w(wcur, n); wcur += n; return w;
        };

        // Encoder weights
        WeightAlloc w_input_w = walloc(32*9*9);   WeightAlloc w_input_b = walloc(32);
        WeightAlloc w_enc1_w  = walloc(64*32*9);   WeightAlloc w_enc1_b  = walloc(64);
        WeightAlloc w_enc2_w  = walloc(128*64*9);   WeightAlloc w_enc2_b  = walloc(128);
        WeightAlloc w_enc3_w  = walloc(128*128*9);  WeightAlloc w_enc3_b  = walloc(128);

        // Transformer weights (variable per config)
        std::vector<TransformerBlockWeights> tw(cfg.n_blocks);
        for (int i = 0; i < cfg.n_blocks; i++) {
            tw[i].qkv_w  = walloc(3 * cfg.dim * cfg.dim);
            tw[i].qkv_b  = walloc(3 * cfg.dim);
            tw[i].out_w  = walloc(cfg.dim * cfg.dim);
            tw[i].out_b  = walloc(cfg.dim);
            tw[i].ffn_w1 = walloc(cfg.ffn_hidden * cfg.dim);
            tw[i].ffn_b1 = walloc(cfg.ffn_hidden);
            tw[i].ffn_w2 = walloc(cfg.dim * cfg.ffn_hidden);
            tw[i].ffn_b2 = walloc(cfg.dim);
        }

        // Decoder weights
        WeightAlloc w_d3_w = walloc(128*256); WeightAlloc w_d3_b = walloc(128);
        WeightAlloc w_d2_w = walloc(64*192);  WeightAlloc w_d2_b = walloc(64);
        WeightAlloc w_d1_w = walloc(32*96);   WeightAlloc w_d1_b = walloc(32);
        WeightAlloc w_out_w = walloc(12*32);  WeightAlloc w_out_b = walloc(12);

        int total_weights = wcur;
        results[c].total_params = total_weights;

        // --- Feature buffer allocation ---
        int fcur = 0;
        auto falloc = [&](int ch, Resolution r) -> int {
            int off = fcur; fcur += ch * r.pixels(); return off;
        };

        int f_input  = falloc(9, r0);
        int f_e0     = falloc(32, r0);
        int f_e1     = falloc(64, r1);
        int f_e2     = falloc(128, r2);
        int f_e3     = falloc(128, r3);
        int f_tping  = falloc(128, r3);
        int f_d3_up  = falloc(128, r2);
        int f_d3_cat = falloc(256, r2);
        int f_d3_out = falloc(128, r2);
        int f_d2_up  = falloc(128, r1);
        int f_d2_cat = falloc(192, r1);
        int f_d2_out = falloc(64, r1);
        int f_d1_up  = falloc(64, r0);
        int f_d1_cat = falloc(96, r0);
        int f_d1_out = falloc(32, r0);
        int f_out12  = falloc(12, r0);
        int f_disp   = falloc(3, rD);
        int total_feat = fcur;

        // --- Allocate GPU buffers ---
        VkBuffer weightBuf, featBuf;
        VkDeviceMemory weightMem, featMem;
        VkDeviceSize wb = (VkDeviceSize)total_weights * 2;
        VkDeviceSize fb = (VkDeviceSize)total_feat * 2;

        createBuf(device, physical, wb,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, weightBuf, weightMem);
        createBuf(device, physical, fb,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, featBuf, featMem);

        // Update descriptors
        VkDescriptorBufferInfo wbi = {weightBuf, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo fbi = {featBuf, 0, VK_WHOLE_SIZE};
        VkDescriptorBufferInfo obi = {featBuf, 0, VK_WHOLE_SIZE};
        VkWriteDescriptorSet writes[3] = {};
        writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 0, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &wbi, nullptr};
        writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 1, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &fbi, nullptr};
        writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 2, 0, 1,
                     VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &obi, nullptr};
        vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

        // --- Record command buffer ---
        int ts_idx = 0;
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        vkBeginCommandBuffer(cmd, &cbi);
        vkCmdResetQueryPool(cmd, queryPool, 0, MAX_TS);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeLayout, 0, 1, &descSet, 0, nullptr);

        auto writeTS = [&]() {
            if (ts_idx < MAX_TS)
                vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, ts_idx++);
        };

        // TS 0: start
        writeTS();

        // === ENCODER ===
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeInputConv);
            Conv3x3Push p = {}; p.in_channels=9; p.out_channels=32;
            p.width=r0.w; p.height=r0.h;
            p.weight_offset=w_input_w.offset; p.bias_offset=w_input_b.offset;
            p.input_offset=f_input; p.output_offset=f_e0; p.relu=1;
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
            vkCmdDispatch(cmd, (r0.pixels()+255)/256, 1, 1);
            addBarrier(cmd);
        }
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeEnc1);
            StridedConvPush p = {}; p.in_channels=32; p.out_channels=64;
            p.in_width=r0.w; p.in_height=r0.h; p.out_width=r1.w; p.out_height=r1.h;
            p.weight_offset=w_enc1_w.offset; p.bias_offset=w_enc1_b.offset;
            p.input_offset=f_e0; p.output_offset=f_e1;
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
            vkCmdDispatch(cmd, (r1.pixels()+255)/256, 1, 1);
            addBarrier(cmd);
        }
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeEnc2);
            StridedConvPush p = {}; p.in_channels=64; p.out_channels=128;
            p.in_width=r1.w; p.in_height=r1.h; p.out_width=r2.w; p.out_height=r2.h;
            p.weight_offset=w_enc2_w.offset; p.bias_offset=w_enc2_b.offset;
            p.input_offset=f_e1; p.output_offset=f_e2;
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
            vkCmdDispatch(cmd, (r2.pixels()+255)/256, 1, 1);
            addBarrier(cmd);
        }
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeEnc3);
            StridedConvPush p = {}; p.in_channels=128; p.out_channels=128;
            p.in_width=r2.w; p.in_height=r2.h; p.out_width=r3.w; p.out_height=r3.h;
            p.weight_offset=w_enc3_w.offset; p.bias_offset=w_enc3_b.offset;
            p.input_offset=f_e2; p.output_offset=f_e3;
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
            vkCmdDispatch(cmd, (r3.pixels()+255)/256, 1, 1);
            addBarrier(cmd);
        }

        // TS 1: after encoder
        writeTS();

        // === TRANSFORMER ===
        {
            int ws = cfg.window_size;
            int wx = (r3.w + ws - 1) / ws;
            int wy = (r3.h + ws - 1) / ws;
            int total_windows = wx * wy;
            int cur_in = f_e3, cur_out = f_tping;

            for (int blk = 0; blk < cfg.n_blocks; blk++) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, transPipe);
                TransformerPush p = {};
                p.n_tokens = r3.pixels();
                p.dim = cfg.dim; p.n_heads = cfg.n_heads; p.head_dim = cfg.head_dim;
                p.spatial_w = r3.w; p.spatial_h = r3.h; p.window_size = ws;
                p.qkv_w_offset  = tw[blk].qkv_w.offset;
                p.qkv_b_offset  = tw[blk].qkv_b.offset;
                p.out_w_offset  = tw[blk].out_w.offset;
                p.out_b_offset  = tw[blk].out_b.offset;
                p.ffn_w1_offset = tw[blk].ffn_w1.offset;
                p.ffn_b1_offset = tw[blk].ffn_b1.offset;
                p.ffn_w2_offset = tw[blk].ffn_w2.offset;
                p.ffn_b2_offset = tw[blk].ffn_b2.offset;
                p.input_offset  = cur_in;
                p.output_offset = cur_out;
                vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
                vkCmdDispatch(cmd, total_windows, 1, 1);
                addBarrier(cmd);

                int tmp = cur_in; cur_in = cur_out; cur_out = tmp;
            }
            // If odd number of blocks, final result is in f_tping — need to adjust
            // For even blocks, result is back in f_e3. For odd, it's in f_tping.
            // Decoder reads from f_e3, so if odd blocks we need to swap.
            if (cfg.n_blocks % 2 != 0) {
                // Result is in f_tping, but decoder expects f_e3.
                // Just point the decoder upsample at f_tping instead.
                // We'll handle this below.
                f_e3 = f_tping; // temporary override for this config
            }
        }

        // TS 2: after transformer
        writeTS();

        // === DECODER ===
        // dec3: up(128, r3->r2) + cat(enc2=128) = 256 -> PW 256->128
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeNNUpsample);
            NNUpsamplePush up = {}; up.channels=128;
            up.in_width=r3.w; up.in_height=r3.h; up.out_width=r2.w; up.out_height=r2.h;
            up.input_offset=f_e3; up.output_offset=f_d3_up;
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(up), &up);
            vkCmdDispatch(cmd, (128*r3.pixels()+255)/256, 1, 1);
            addBarrier(cmd);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeConcatSkip);
            ConcatPush cp = {}; cp.ch_a=128; cp.ch_b=128; cp.width=r2.w; cp.height=r2.h;
            cp.offset_a=f_d3_up; cp.offset_b=f_e2; cp.output_offset=f_d3_cat;
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cp), &cp);
            vkCmdDispatch(cmd, (256*r2.pixels()+255)/256, 1, 1);
            addBarrier(cmd);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipePW256to128);
            PWConvPush pw = {}; pw.in_channels=256; pw.out_channels=128;
            pw.width=r2.w; pw.height=r2.h;
            pw.weight_offset=w_d3_w.offset; pw.bias_offset=w_d3_b.offset;
            pw.input_offset=f_d3_cat; pw.output_offset=f_d3_out; pw.relu=1;
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pw), &pw);
            vkCmdDispatch(cmd, (r2.pixels()+255)/256, 1, 1);
            addBarrier(cmd);
        }
        // dec2: up(128, r2->r1) + cat(enc1=64) = 192 -> PW 192->64
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeNNUpsample);
            NNUpsamplePush up = {}; up.channels=128;
            up.in_width=r2.w; up.in_height=r2.h; up.out_width=r1.w; up.out_height=r1.h;
            up.input_offset=f_d3_out; up.output_offset=f_d2_up;
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(up), &up);
            vkCmdDispatch(cmd, (128*r2.pixels()+255)/256, 1, 1);
            addBarrier(cmd);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeConcatSkip);
            ConcatPush cp = {}; cp.ch_a=128; cp.ch_b=64; cp.width=r1.w; cp.height=r1.h;
            cp.offset_a=f_d2_up; cp.offset_b=f_e1; cp.output_offset=f_d2_cat;
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cp), &cp);
            vkCmdDispatch(cmd, (192*r1.pixels()+255)/256, 1, 1);
            addBarrier(cmd);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipePW192to64);
            PWConvPush pw = {}; pw.in_channels=192; pw.out_channels=64;
            pw.width=r1.w; pw.height=r1.h;
            pw.weight_offset=w_d2_w.offset; pw.bias_offset=w_d2_b.offset;
            pw.input_offset=f_d2_cat; pw.output_offset=f_d2_out; pw.relu=1;
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pw), &pw);
            vkCmdDispatch(cmd, (r1.pixels()+255)/256, 1, 1);
            addBarrier(cmd);
        }
        // dec1: up(64, r1->r0) + cat(enc0=32) = 96 -> PW 96->32
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeNNUpsample);
            NNUpsamplePush up = {}; up.channels=64;
            up.in_width=r1.w; up.in_height=r1.h; up.out_width=r0.w; up.out_height=r0.h;
            up.input_offset=f_d2_out; up.output_offset=f_d1_up;
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(up), &up);
            vkCmdDispatch(cmd, (64*r1.pixels()+255)/256, 1, 1);
            addBarrier(cmd);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeConcatSkip);
            ConcatPush cp = {}; cp.ch_a=64; cp.ch_b=32; cp.width=r0.w; cp.height=r0.h;
            cp.offset_a=f_d1_up; cp.offset_b=f_e0; cp.output_offset=f_d1_cat;
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cp), &cp);
            vkCmdDispatch(cmd, (96*r0.pixels()+255)/256, 1, 1);
            addBarrier(cmd);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipePW96to32);
            PWConvPush pw = {}; pw.in_channels=96; pw.out_channels=32;
            pw.width=r0.w; pw.height=r0.h;
            pw.weight_offset=w_d1_w.offset; pw.bias_offset=w_d1_b.offset;
            pw.input_offset=f_d1_cat; pw.output_offset=f_d1_out; pw.relu=1;
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pw), &pw);
            vkCmdDispatch(cmd, (r0.pixels()+255)/256, 1, 1);
            addBarrier(cmd);
        }

        // TS 3: after decoder
        writeTS();

        // === OUTPUT ===
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipePW32to12);
            PWConvPush pw = {}; pw.in_channels=32; pw.out_channels=12;
            pw.width=r0.w; pw.height=r0.h;
            pw.weight_offset=w_out_w.offset; pw.bias_offset=w_out_b.offset;
            pw.input_offset=f_d1_out; pw.output_offset=f_out12; pw.relu=0;
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pw), &pw);
            vkCmdDispatch(cmd, (r0.pixels()+255)/256, 1, 1);
            addBarrier(cmd);
        }
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipePixelShuffle);
            PixelShufflePush p = {}; p.render_width=r0.w; p.render_height=r0.h;
            p.display_width=rD.w; p.display_height=rD.h;
            p.scale=2; p.input_offset=f_out12; p.output_offset=f_disp;
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
            vkCmdDispatch(cmd, (rD.w+15)/16, (rD.h+15)/16, 1);
            addBarrier(cmd);
        }

        // TS 4: end
        writeTS();

        vkEndCommandBuffer(cmd);

        // --- Warmup ---
        for (int i = 0; i < warmup; i++) {
            VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
            vkQueueSubmit(queue, 1, &si, fence);
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
            vkResetFences(device, 1, &fence);
        }

        // --- Benchmark ---
        double enc_total = 0, trans_total = 0, dec_total = 0, out_total = 0, frame_total = 0;

        for (int i = 0; i < loops; i++) {
            VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
            vkQueueSubmit(queue, 1, &si, fence);
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
            vkResetFences(device, 1, &fence);

            uint64_t ts[MAX_TS];
            vkGetQueryPoolResults(device, queryPool, 0, ts_idx, sizeof(ts), ts,
                                  sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);

            double ns = props.limits.timestampPeriod;
            double enc_ms   = (ts[1] - ts[0]) * ns / 1e6;
            double trans_ms = (ts[2] - ts[1]) * ns / 1e6;
            double dec_ms   = (ts[3] - ts[2]) * ns / 1e6;
            double out_ms   = (ts[4] - ts[3]) * ns / 1e6;
            double total_ms = (ts[4] - ts[0]) * ns / 1e6;

            enc_total   += enc_ms;
            trans_total += trans_ms;
            dec_total   += dec_ms;
            out_total   += out_ms;
            frame_total += total_ms;
        }

        results[c].encoder_ms     = enc_total / loops;
        results[c].transformer_ms = trans_total / loops;
        results[c].decoder_ms     = dec_total / loops;
        results[c].output_ms      = out_total / loops;
        results[c].total_ms       = frame_total / loops;

        printf("    Encoder:     %6.3f ms\n", results[c].encoder_ms);
        printf("    Transformer: %6.3f ms\n", results[c].transformer_ms);
        printf("    Decoder:     %6.3f ms\n", results[c].decoder_ms);
        printf("    Output:      %6.3f ms\n", results[c].output_ms);
        printf("    TOTAL:       %6.3f ms  (%.0f FPS)\n",
               results[c].total_ms, 1000.0 / results[c].total_ms);

        // Cleanup config buffers
        vkDeviceWaitIdle(device);
        vkDestroyBuffer(device, weightBuf, nullptr); vkFreeMemory(device, weightMem, nullptr);
        vkDestroyBuffer(device, featBuf, nullptr); vkFreeMemory(device, featMem, nullptr);

        // Restore f_e3 if we overrode it
        f_e3 = -1; // will be recalculated next iteration
    }

    // ================================================================
    // FINAL COMPARISON TABLE
    // ================================================================

    printf("\n\n================================================================\n");
    printf("  FINAL RESULTS — Prism Scaling Benchmark\n");
    printf("  GPU: %s\n", props.deviceName);
    printf("  540x960 -> 1080x1920 | warmup=%d | loops=%d\n", warmup, loops);
    printf("================================================================\n\n");

    printf("  %-16s %6s %4s %7s | %7s %7s %7s %7s | %7s %5s\n",
           "Config", "Params", "FFN", "Blocks",
           "Enc", "Trans", "Dec", "Out",
           "Total", "FPS");
    printf("  %-16s %6s %4s %7s | %7s %7s %7s %7s | %7s %5s\n",
           "----------------", "------", "----", "-------",
           "-------", "-------", "-------", "-------",
           "-------", "-----");

    for (int c = 0; c < n_configs; c++) {
        auto& cfg = configs[c];
        auto& r = results[c];
        if (r.total_ms < 0) {
            printf("  %-16s %5.1fK %4d %7d | %43s\n",
                   cfg.tag, r.total_params/1000.0, cfg.ffn_hidden, cfg.n_blocks, "SKIPPED");
            continue;
        }
        printf("  %-16s %5.1fK %4d %7d | %6.2fms %6.2fms %6.2fms %6.2fms | %6.2fms %5.0f\n",
               cfg.tag, r.total_params/1000.0, cfg.ffn_hidden, cfg.n_blocks,
               r.encoder_ms, r.transformer_ms, r.decoder_ms, r.output_ms,
               r.total_ms, 1000.0 / r.total_ms);
    }

    // Speedup relative to baseline
    if (results[0].total_ms > 0) {
        printf("\n  Relative to baseline:\n");
        for (int c = 1; c < n_configs; c++) {
            if (results[c].total_ms < 0) continue;
            double slowdown = results[c].total_ms / results[0].total_ms;
            double param_ratio = (double)results[c].total_params / results[0].total_params;
            printf("  %-16s  %.2fx params  ->  %.2fx slower  (transformer: %.2fx)\n",
                   configs[c].tag, param_ratio, slowdown,
                   results[c].transformer_ms / results[0].transformer_ms);
        }
    }

    // Budget analysis
    printf("\n  Budget analysis (16.67ms = 60 FPS target):\n");
    for (int c = 0; c < n_configs; c++) {
        if (results[c].total_ms < 0) continue;
        double remaining = 16.67 - results[c].total_ms;
        printf("  %-16s  %6.2fms  %s  %s\n",
               configs[c].tag, results[c].total_ms,
               remaining > 0 ? "UNDER BUDGET" : "OVER BUDGET ",
               remaining > 0 ? "(OK for 60fps)" : "(need lower res or fewer blocks)");
    }

    printf("\n================================================================\n");

    // Cleanup
    vkDeviceWaitIdle(device);
    vkDestroyQueryPool(device, queryPool, nullptr);
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, cmdPool, nullptr);

    auto dp = [&](VkPipeline p) { if (p) vkDestroyPipeline(device, p, nullptr); };
    dp(pipeInputConv); dp(pipeEnc1); dp(pipeEnc2); dp(pipeEnc3);
    dp(pipeTransH512); dp(pipeTransH1024); dp(pipeTransH2048);
    dp(pipeNNUpsample); dp(pipeConcatSkip);
    dp(pipePW256to128); dp(pipePW192to64); dp(pipePW96to32);
    dp(pipePW32to12); dp(pipePixelShuffle);
    vkDestroyPipelineLayout(device, pipeLayout, nullptr);
    vkDestroyDescriptorPool(device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    return 0;
}
