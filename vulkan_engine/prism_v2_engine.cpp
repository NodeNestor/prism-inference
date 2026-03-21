// Prism V2 Full Pipeline Benchmark
// U-Net encoder (conv) + Transformer bottleneck + U-Net decoder + PixelShuffle
// All dispatches in ONE command buffer with pipeline barriers.
// Cooperative vectors (tensor cores) for all matmuls.
//
// Architecture (balanced preset, ~2.5M params):
//   Input: 6ch (color3+depth1+mv2) padded to 9ch at 540x960
//   Encoder:
//     input_conv: Conv3x3 9->32, ReLU          @ 540x960  (e0)
//     enc1: StridedConv 32->64 + DSCBlock(64)   @ 270x480  (e1)
//     enc2: StridedConv 64->128 + DSCBlock(128)  @ 135x240  (e2)
//     enc3: StridedConv 128->128 + DSCBlock(128)  @ 68x120   (e3)
//   Transformer: 4 windowed attention blocks     @ 68x120, dim=128
//   Decoder (with skip connections):
//     dec3: TransConv 128->128 + cat(e2) + PW 256->128 + DSCBlock(128) @ 135x240
//     dec2: TransConv 128->64  + cat(e1) + PW 128->64  + DSCBlock(64)  @ 270x480
//     dec1: TransConv 64->32   + cat(e0) + PW 64->32   + DSCBlock(32)  @ 540x960
//   Output: Conv3x3 32->12 + PixelShuffle(2) + Sigmoid -> 1080x1920

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

struct FusedDWPWPush {
    int32_t channels, width, height;
    int32_t dw_weight_offset, pw_weight_offset, pw_bias_offset;
    int32_t input_offset, output_offset;
    int32_t relu, residual_offset;
};

struct TransConvPush {
    int32_t in_channels, out_channels, in_width, in_height;
    int32_t out_width, out_height;
    int32_t weight_offset, bias_offset, input_offset, output_offset;
};

struct ConcatPush {
    int32_t ch_a, ch_b, width, height;
    int32_t offset_a, offset_b, output_offset;
};

struct PWConvPush {
    int32_t in_channels, out_channels, width, height;
    int32_t weight_offset, bias_offset, input_offset, output_offset;
    int32_t relu, residual_offset;
};

// Transformer push constants (matches attention_windowed.comp.glsl)
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
// Helper: load SPV and create pipeline
// ============================================================================

static VkPipeline loadPipeline(VkDevice device, VkPipelineLayout layout, const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) { printf("  WARN: Cannot open %s\n", path); return VK_NULL_HANDLE; }
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint32_t> code(sz / 4);
    f.read((char*)code.data(), sz);

    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sz; smi.pCode = code.data();
    VkShaderModule mod;
    if (vkCreateShaderModule(device, &smi, nullptr, &mod) != VK_SUCCESS) {
        printf("  WARN: Shader module creation failed for %s\n", path);
        return VK_NULL_HANDLE;
    }

    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = mod;
    cpci.stage.pName = "main";
    cpci.layout = layout;

    VkPipeline pipe;
    VkResult r = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe);
    vkDestroyShaderModule(device, mod, nullptr);

    if (r != VK_SUCCESS) {
        printf("  WARN: Pipeline creation failed for %s (VkResult=%d)\n", path, r);
        return VK_NULL_HANDLE;
    }
    return pipe;
}

// ============================================================================
// Helpers
// ============================================================================

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
// Resolution levels
// ============================================================================

struct Resolution {
    int w, h;
    int pixels() const { return w * h; }
};

// ============================================================================
// Weight offset tracker
// ============================================================================

struct WeightAlloc {
    int offset;   // fp16 element offset
    int count;    // number of fp16 elements

    WeightAlloc() : offset(0), count(0) {}
    WeightAlloc(int off, int cnt) : offset(off), count(cnt) {}
};

static int g_weight_cursor = 0;

static WeightAlloc allocWeight(int count) {
    WeightAlloc w(g_weight_cursor, count);
    g_weight_cursor += count;
    return w;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    printf("================================================================\n");
    printf("  Prism V2 Full Pipeline Benchmark\n");
    printf("  U-Net + Windowed Transformer + Cooperative Vectors\n");
    printf("================================================================\n\n");

    int gpu_id = argc > 1 ? atoi(argv[1]) : 0;
    int warmup = 20;
    int loops = 100;

    // ---- Vulkan init ----
    if (volkInitialize() != VK_SUCCESS) { printf("FATAL: No Vulkan\n"); return 1; }

    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.apiVersion = VK_API_VERSION_1_3;
    appInfo.pApplicationName = "PrismV2Bench";
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
    printf("GPU %d: %s\n", gpu_id, props.deviceName);
    printf("  Timestamp period: %.1f ns\n", props.limits.timestampPeriod);

    // Feature chain for cooperative vector + fp16
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMat = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    VkPhysicalDeviceShaderFloat16Int8Features f16feat = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    f16feat.pNext = &coopMat;
    VkPhysicalDevice16BitStorageFeatures s16feat = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    s16feat.pNext = &f16feat;
    VkPhysicalDeviceFeatures2 features2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.pNext = &s16feat;
    vkGetPhysicalDeviceFeatures2(physical, &features2);

    printf("  FP16 arithmetic: %s\n", f16feat.shaderFloat16 ? "YES" : "NO");
    printf("  16-bit storage: %s\n", s16feat.storageBuffer16BitAccess ? "YES" : "NO");

    // Queue
    uint32_t qc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &qc, nullptr);
    std::vector<VkQueueFamilyProperties> qp(qc);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &qc, qp.data());
    uint32_t qf = 0;
    for (uint32_t i = 0; i < qc; i++) {
        if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
    }

    // Device
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

    // Timestamp query pool — many queries for per-stage timing
    const int MAX_TIMESTAMPS = 64;
    VkQueryPool queryPool;
    VkQueryPoolCreateInfo qpi = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount = MAX_TIMESTAMPS;
    vkCreateQueryPool(device, &qpi, nullptr, &queryPool);

    // ---- Descriptor set + pipeline layout ----
    VkDescriptorSetLayoutBinding bindings[3] = {};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    bindings[2] = {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 3; dli.pBindings = bindings;
    VkDescriptorSetLayout descLayout;
    vkCreateDescriptorSetLayout(device, &dli, nullptr, &descLayout);

    // Push constant size: max of all push structs
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
    printf("\nLoading shader pipelines...\n");
    const char* sd = "shaders/";
    char path[512];

    #define LOAD_PIPE(name, file) \
        snprintf(path, 512, "%s%s", sd, file); \
        VkPipeline name = loadPipeline(device, pipeLayout, path); \
        printf("  %-30s %s\n", file, name ? "OK" : "FAILED");

    LOAD_PIPE(pipeInputConvV2,   "input_conv_v2.spv");
    LOAD_PIPE(pipeInputConv,     "input_conv_coopvec.spv");   // fallback 6->64
    LOAD_PIPE(pipeStridedConv,   "strided_conv.spv");
    LOAD_PIPE(pipeFusedDWPW64,   "fused_dw_pw.spv");
    LOAD_PIPE(pipeFusedDWPW128,  "fused_dw_pw_128.spv");
    LOAD_PIPE(pipeFusedDWPW32,   "fused_dw_pw_32.spv");
    LOAD_PIPE(pipeTransformer,   "attention_windowed.spv");
    LOAD_PIPE(pipeTransConv,     "transpose_conv.spv");
    LOAD_PIPE(pipeConcatSkip,    "concat_skip.spv");
    LOAD_PIPE(pipeConv3x3,       "conv3x3.spv");
    LOAD_PIPE(pipePWConv,        "pointwise_conv.spv");       // naive PW (64ch cap)
    LOAD_PIPE(pipePWConvGeneric, "pw_conv_generic.spv");      // generic PW any channels
    LOAD_PIPE(pipePWConvCoopvec, "pointwise_conv_coopvec.spv");
    LOAD_PIPE(pipePW64to12,      "pw_conv_64to12_coopvec.spv");
    LOAD_PIPE(pipePixelShuffle,  "pixelshuffle_sigmoid.spv");
    #undef LOAD_PIPE

    // Check critical pipelines
    bool ok = pipeStridedConv && pipeTransformer && pipeTransConv &&
              pipeConcatSkip && pipeConv3x3 && (pipePWConvGeneric || pipePWConv) && pipePixelShuffle;
    if (!ok) {
        printf("\nFATAL: Missing critical pipeline(s). Cannot continue.\n");
        return 1;
    }

    // Determine which cooperative vector pipelines are available
    bool hasCoopInputV2 = pipeInputConvV2 != VK_NULL_HANDLE;
    bool hasFused64 = pipeFusedDWPW64 != VK_NULL_HANDLE;
    bool hasFused128 = pipeFusedDWPW128 != VK_NULL_HANDLE;
    bool hasFused32 = pipeFusedDWPW32 != VK_NULL_HANDLE;

    printf("\nCooperative vector status:\n");
    printf("  Input conv V2 (9->32): %s\n", hasCoopInputV2 ? "TENSOR CORE" : "fallback to naive");
    printf("  Fused DW+PW 32ch:      %s\n", hasFused32 ? "TENSOR CORE" : "fallback to naive");
    printf("  Fused DW+PW 64ch:      %s\n", hasFused64 ? "TENSOR CORE" : "fallback to naive");
    printf("  Fused DW+PW 128ch:     %s\n", hasFused128 ? "TENSOR CORE" : "fallback to naive");
    printf("  Transformer:           TENSOR CORE (always)\n");

    // ================================================================
    // Resolution plan
    // ================================================================

    Resolution r0 = {960, 540};   // input / enc0 / dec1
    Resolution r1 = {480, 270};   // enc1 / dec2
    Resolution r2 = {240, 135};   // enc2 / dec3
    Resolution r3 = {120, 68};    // enc3 / transformer bottleneck
    Resolution rD = {1920, 1080}; // display output

    printf("\nResolution plan:\n");
    printf("  L0: %dx%d (%d pixels)\n", r0.w, r0.h, r0.pixels());
    printf("  L1: %dx%d (%d pixels)\n", r1.w, r1.h, r1.pixels());
    printf("  L2: %dx%d (%d pixels)\n", r2.w, r2.h, r2.pixels());
    printf("  L3: %dx%d (%d pixels)\n", r3.w, r3.h, r3.pixels());
    printf("  Display: %dx%d (%d pixels)\n", rD.w, rD.h, rD.pixels());

    // ================================================================
    // Weight allocation — track offsets for every layer
    // ================================================================

    g_weight_cursor = 0;
    printf("\nWeight allocation (fp16 elements):\n");

    // Input conv: Conv3x3 9->32 (pad input 6ch to 9ch with zeros for neat kernel size)
    // Actually, we'll use the real 6ch input. Weight shape: [32, 6, 3, 3] = 32*6*9 = 1728
    // But the V2 shader expects 9ch input -> weight [32, 9*9] = [32, 81] = 2592
    // Let's use 6 channels properly: [32, 6*9] = [32, 54] = 1728 weights + 32 bias
    // If using input_conv_v2 (9ch), pad to 9: [32, 81] = 2592 + 32
    // Simpler: use 6ch with input_conv_coopvec (already works for 6->64, but we need 6->32)
    // Actually input_conv_coopvec has hardcoded coopvecNV<64> output.
    // Let's use conv3x3.spv (naive) for input conv 6->32. It's flexible.
    int inp_in_ch = 6;
    int inp_out_ch = 32;
    WeightAlloc w_input_conv = allocWeight(inp_out_ch * inp_in_ch * 9);  // 1728
    WeightAlloc w_input_bias = allocWeight(inp_out_ch);                   // 32
    printf("  input_conv (6->32, 3x3):  w=%d b=%d\n", w_input_conv.count, w_input_bias.count);

    // Encoder stages
    // enc1: StridedConv 32->64 + DSCBlock(64)
    int enc1_in = 32, enc1_out = 64;
    WeightAlloc w_enc1_conv = allocWeight(enc1_out * enc1_in * 9);  // 18432
    WeightAlloc w_enc1_bias = allocWeight(enc1_out);                 // 64
    // DSC block: DW 3x3 (64*9=576) + PW 1x1 (64*64=4096 + 64 bias)
    WeightAlloc w_enc1_dw = allocWeight(enc1_out * 9);               // 576
    WeightAlloc w_enc1_pw = allocWeight(enc1_out * enc1_out);        // 4096
    WeightAlloc w_enc1_pw_b = allocWeight(enc1_out);                 // 64
    printf("  enc1 (32->64, stride2 + DSC): conv=%d dsc=%d\n",
           w_enc1_conv.count, w_enc1_dw.count + w_enc1_pw.count + w_enc1_pw_b.count);

    // enc2: StridedConv 64->128 + DSCBlock(128)
    int enc2_in = 64, enc2_out = 128;
    WeightAlloc w_enc2_conv = allocWeight(enc2_out * enc2_in * 9);   // 73728
    WeightAlloc w_enc2_bias = allocWeight(enc2_out);                  // 128
    WeightAlloc w_enc2_dw = allocWeight(enc2_out * 9);                // 1152
    WeightAlloc w_enc2_pw = allocWeight(enc2_out * enc2_out);         // 16384
    WeightAlloc w_enc2_pw_b = allocWeight(enc2_out);                  // 128
    printf("  enc2 (64->128, stride2 + DSC): conv=%d dsc=%d\n",
           w_enc2_conv.count, w_enc2_dw.count + w_enc2_pw.count + w_enc2_pw_b.count);

    // enc3: StridedConv 128->128 + DSCBlock(128)
    int enc3_in = 128, enc3_out = 128;
    WeightAlloc w_enc3_conv = allocWeight(enc3_out * enc3_in * 9);   // 147456
    WeightAlloc w_enc3_bias = allocWeight(enc3_out);                  // 128
    WeightAlloc w_enc3_dw = allocWeight(enc3_out * 9);                // 1152
    WeightAlloc w_enc3_pw = allocWeight(enc3_out * enc3_out);         // 16384
    WeightAlloc w_enc3_pw_b = allocWeight(enc3_out);                  // 128
    printf("  enc3 (128->128, stride2 + DSC): conv=%d dsc=%d\n",
           w_enc3_conv.count, w_enc3_dw.count + w_enc3_pw.count + w_enc3_pw_b.count);

    // Transformer: 4 blocks, each has:
    //   QKV: [3*128, 128] + [3*128] = 49152 + 384
    //   Out proj: [128, 128] + [128] = 16384 + 128
    //   FFN W1: [512, 128] + [512] = 65536 + 512
    //   FFN W2: [128, 512] + [128] = 65536 + 128
    struct TransformerWeights {
        WeightAlloc qkv_w, qkv_b, out_w, out_b;
        WeightAlloc ffn_w1, ffn_b1, ffn_w2, ffn_b2;
    };
    TransformerWeights tw[4];
    int t_dim = 128, t_hidden = 512;
    for (int i = 0; i < 4; i++) {
        tw[i].qkv_w = allocWeight(3 * t_dim * t_dim);    // 49152
        tw[i].qkv_b = allocWeight(3 * t_dim);             // 384
        tw[i].out_w = allocWeight(t_dim * t_dim);          // 16384
        tw[i].out_b = allocWeight(t_dim);                  // 128
        tw[i].ffn_w1 = allocWeight(t_hidden * t_dim);      // 65536
        tw[i].ffn_b1 = allocWeight(t_hidden);              // 512
        tw[i].ffn_w2 = allocWeight(t_dim * t_hidden);      // 65536
        tw[i].ffn_b2 = allocWeight(t_dim);                 // 128
    }
    printf("  transformer (4 blocks): %d per block\n",
           3*t_dim*t_dim + 3*t_dim + t_dim*t_dim + t_dim +
           t_hidden*t_dim + t_hidden + t_dim*t_hidden + t_dim);

    // Decoder stages
    // dec3: TransConv 128->128, k=4, s=2 + cat(e2,128ch) + PW 256->128 + DSC(128)
    // TransConv weights: [128, 128, 4, 4] = 262144
    WeightAlloc w_dec3_tconv = allocWeight(128 * 128 * 16);    // 262144
    WeightAlloc w_dec3_tconv_b = allocWeight(128);              // 128
    WeightAlloc w_dec3_reduce = allocWeight(256 * 128);         // PW 256->128: 32768
    WeightAlloc w_dec3_reduce_b = allocWeight(128);             // 128
    WeightAlloc w_dec3_dw = allocWeight(128 * 9);               // 1152
    WeightAlloc w_dec3_pw = allocWeight(128 * 128);             // 16384
    WeightAlloc w_dec3_pw_b = allocWeight(128);                 // 128
    printf("  dec3 (tconv 128->128 + skip + DSC): %d\n",
           w_dec3_tconv.count + w_dec3_tconv_b.count + w_dec3_reduce.count +
           w_dec3_reduce_b.count + w_dec3_dw.count + w_dec3_pw.count + w_dec3_pw_b.count);

    // dec2: TransConv 128->64, k=4, s=2 + cat(e1,64ch) + PW 128->64 + DSC(64)
    WeightAlloc w_dec2_tconv = allocWeight(128 * 64 * 16);     // 131072
    WeightAlloc w_dec2_tconv_b = allocWeight(64);               // 64
    WeightAlloc w_dec2_reduce = allocWeight(128 * 64);          // PW 128->64: 8192
    WeightAlloc w_dec2_reduce_b = allocWeight(64);              // 64
    WeightAlloc w_dec2_dw = allocWeight(64 * 9);                // 576
    WeightAlloc w_dec2_pw = allocWeight(64 * 64);               // 4096
    WeightAlloc w_dec2_pw_b = allocWeight(64);                  // 64
    printf("  dec2 (tconv 128->64 + skip + DSC): %d\n",
           w_dec2_tconv.count + w_dec2_tconv_b.count + w_dec2_reduce.count +
           w_dec2_reduce_b.count + w_dec2_dw.count + w_dec2_pw.count + w_dec2_pw_b.count);

    // dec1: TransConv 64->32, k=4, s=2 + cat(e0,32ch) + PW 64->32 + DSC(32)
    WeightAlloc w_dec1_tconv = allocWeight(64 * 32 * 16);      // 32768
    WeightAlloc w_dec1_tconv_b = allocWeight(32);               // 32
    WeightAlloc w_dec1_reduce = allocWeight(64 * 32);           // PW 64->32: 2048
    WeightAlloc w_dec1_reduce_b = allocWeight(32);              // 32
    WeightAlloc w_dec1_dw = allocWeight(32 * 9);                // 288
    WeightAlloc w_dec1_pw = allocWeight(32 * 32);               // 1024
    WeightAlloc w_dec1_pw_b = allocWeight(32);                  // 32
    printf("  dec1 (tconv 64->32 + skip + DSC): %d\n",
           w_dec1_tconv.count + w_dec1_tconv_b.count + w_dec1_reduce.count +
           w_dec1_reduce_b.count + w_dec1_dw.count + w_dec1_pw.count + w_dec1_pw_b.count);

    // Output conv: Conv3x3 32->12, no ReLU
    WeightAlloc w_output_conv = allocWeight(12 * 32 * 9);  // 3456
    WeightAlloc w_output_bias = allocWeight(12);             // 12
    printf("  output_conv (32->12, 3x3): w=%d b=%d\n", w_output_conv.count, w_output_bias.count);

    int total_weights = g_weight_cursor;
    printf("\nTotal weights: %d fp16 (%.2f MB, %.1f K params)\n",
           total_weights, total_weights * 2.0 / 1024 / 1024, total_weights / 1000.0);

    // ================================================================
    // Feature buffer allocation
    // We need space for all intermediate feature maps simultaneously
    // because skip connections reference earlier encoder outputs.
    // ================================================================

    // Named feature regions (fp16 element offsets into feature buffer)
    // Each region: channels * pixels fp16 elements
    int feat_cursor = 0;
    auto allocFeat = [&](int ch, Resolution r, const char* name) -> int {
        int off = feat_cursor;
        int size = ch * r.pixels();
        feat_cursor += size;
        printf("  %-20s %3dch x %dx%d = %9d fp16 (off=%d)\n", name, ch, r.w, r.h, size, off);
        return off;
    };

    printf("\nFeature buffer allocation:\n");

    // Input: 6 channels at r0
    int f_input     = allocFeat(6, r0, "input");

    // Encoder skip connections (kept alive for decoder)
    int f_e0        = allocFeat(32, r0, "enc0 (skip)");
    int f_e1        = allocFeat(64, r1, "enc1 (skip)");
    int f_e2        = allocFeat(128, r2, "enc2 (skip)");

    // Encoder e3 / transformer working space at r3
    // Transformer does in-place (ping-pong between two regions)
    int f_e3        = allocFeat(128, r3, "enc3/transformer A");
    int f_t_ping    = allocFeat(128, r3, "transformer B");

    // Decoder working buffers (reusable after each stage)
    // dec3 needs: transconv output (128ch @ r2) + concat (256ch @ r2) + reduce+DSC output (128ch @ r2)
    int f_dec_tmp1  = allocFeat(128, r2, "dec tmp1 (128@r2)");
    int f_dec_cat   = allocFeat(256, r2, "dec cat (256@r2)");
    // dec2: transconv (64@r1) + concat (128@r1) + reduce+DSC (64@r1)
    // We can reuse dec_tmp1 area (it's at r2 = 135*240 = 32400 pixels * 128ch = 4.1M fp16)
    // dec2 cat = 128ch * 129600 = 16.6M fp16 — doesn't fit in r2 space
    // Better: allocate separate regions for each decoder stage
    int f_dec2_tmp  = allocFeat(64, r1, "dec2 tmp (64@r1)");
    int f_dec2_cat  = allocFeat(128, r1, "dec2 cat (128@r1)");
    int f_dec1_tmp  = allocFeat(32, r0, "dec1 tmp (32@r0)");
    int f_dec1_cat  = allocFeat(64, r0, "dec1 cat (64@r0)");

    // Decoder output regions (result of each decoder stage)
    int f_dec3_out  = allocFeat(128, r2, "dec3 out (128@r2)");
    int f_dec2_out  = allocFeat(64, r1, "dec2 out (64@r1)");
    int f_dec1_out  = allocFeat(32, r0, "dec1 out (32@r0)");

    // Output conv result: 12ch at r0
    int f_output_12 = allocFeat(12, r0, "output 12ch@r0");

    // Pixelshuffle output: 3ch at display res
    int f_display   = allocFeat(3, rD, "display 3ch@rD");

    int total_feat = feat_cursor;
    printf("\nTotal feature buffer: %d fp16 (%.1f MB)\n", total_feat, total_feat * 2.0 / 1024 / 1024);

    // ================================================================
    // Allocate GPU buffers
    // ================================================================

    VkBuffer weightBuf, featBuf;
    VkDeviceMemory weightMem, featMem;

    VkDeviceSize weightBytes = (VkDeviceSize)total_weights * 2;
    VkDeviceSize featBytes = (VkDeviceSize)total_feat * 2;

    printf("\nAllocating GPU buffers...\n");
    printf("  Weights: %.1f MB\n", weightBytes / 1024.0 / 1024.0);
    printf("  Features: %.1f MB\n", featBytes / 1024.0 / 1024.0);

    createBuf(device, physical, weightBytes,
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, weightBuf, weightMem);
    createBuf(device, physical, featBytes,
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, featBuf, featMem);

    // Update descriptor set
    VkDescriptorBufferInfo wbi = {weightBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo fbi = {featBuf, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo obi = {featBuf, 0, VK_WHOLE_SIZE};  // output binding = same feature buf
    VkWriteDescriptorSet writes[3] = {};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 0, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &wbi, nullptr};
    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 1, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &fbi, nullptr};
    writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 2, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &obi, nullptr};
    vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

    // ================================================================
    // Record command buffer — THE FULL PIPELINE
    // ================================================================

    printf("\nRecording command buffer...\n");
    int ts_idx = 0;  // timestamp counter
    int dispatch_count = 0;

    const char* stage_names[MAX_TIMESTAMPS];
    memset(stage_names, 0, sizeof(stage_names));

    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo cbi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &cbi);

    vkCmdResetQueryPool(cmd, queryPool, 0, MAX_TIMESTAMPS);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeLayout, 0, 1, &descSet, 0, nullptr);

    auto writeTS = [&](const char* name) {
        if (ts_idx < MAX_TIMESTAMPS) {
            stage_names[ts_idx] = name;
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, ts_idx++);
        }
    };

    writeTS("pipeline_start");

    // ============================================================
    // ENCODER
    // ============================================================

    // --- Input Conv: Conv3x3 6->32 + ReLU @ r0 ---
    // Use naive conv3x3 (works for any channel count)
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeConv3x3);
        Conv3x3Push p = {};
        p.in_channels = 6; p.out_channels = 32;
        p.width = r0.w; p.height = r0.h;
        p.weight_offset = w_input_conv.offset;
        p.bias_offset = w_input_bias.offset;
        p.input_offset = f_input;
        p.output_offset = f_e0;
        p.relu = 1;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        uint32_t gx = (r0.w + 15) / 16;
        uint32_t gy = (r0.h + 15) / 16;
        vkCmdDispatch(cmd, gx, gy, 32);
        addBarrier(cmd);
        dispatch_count++;
    }
    writeTS("input_conv");

    // --- enc1: StridedConv 32->64 @ r1 ---
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeStridedConv);
        StridedConvPush p = {};
        p.in_channels = 32; p.out_channels = 64;
        p.in_width = r0.w; p.in_height = r0.h;
        p.out_width = r1.w; p.out_height = r1.h;
        p.weight_offset = w_enc1_conv.offset;
        p.bias_offset = w_enc1_bias.offset;
        p.input_offset = f_e0;
        p.output_offset = f_e1;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        uint32_t groups = (r1.pixels() + 255) / 256;
        vkCmdDispatch(cmd, groups, 1, 1);
        addBarrier(cmd);
        dispatch_count++;
    }

    // enc1 DSCBlock(64): fused DW+PW at r1
    if (hasFused64) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeFusedDWPW64);
        FusedDWPWPush p = {};
        p.channels = 64; p.width = r1.w; p.height = r1.h;
        p.dw_weight_offset = w_enc1_dw.offset;
        p.pw_weight_offset = w_enc1_pw.offset;
        p.pw_bias_offset = w_enc1_pw_b.offset;
        p.input_offset = f_e1;
        p.output_offset = f_e1;  // in-place with residual
        p.relu = 1;
        p.residual_offset = f_e1;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        uint32_t groups = (r1.pixels() + 255) / 256;
        vkCmdDispatch(cmd, groups, 1, 1);
        addBarrier(cmd);
        dispatch_count++;
    }
    writeTS("enc1");

    // --- enc2: StridedConv 64->128 @ r2 ---
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeStridedConv);
        StridedConvPush p = {};
        p.in_channels = 64; p.out_channels = 128;
        p.in_width = r1.w; p.in_height = r1.h;
        p.out_width = r2.w; p.out_height = r2.h;
        p.weight_offset = w_enc2_conv.offset;
        p.bias_offset = w_enc2_bias.offset;
        p.input_offset = f_e1;
        p.output_offset = f_e2;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        uint32_t groups = (r2.pixels() + 255) / 256;
        vkCmdDispatch(cmd, groups, 1, 1);
        addBarrier(cmd);
        dispatch_count++;
    }

    // enc2 DSCBlock(128) at r2
    if (hasFused128) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeFusedDWPW128);
        FusedDWPWPush p = {};
        p.channels = 128; p.width = r2.w; p.height = r2.h;
        p.dw_weight_offset = w_enc2_dw.offset;
        p.pw_weight_offset = w_enc2_pw.offset;
        p.pw_bias_offset = w_enc2_pw_b.offset;
        p.input_offset = f_e2;
        p.output_offset = f_e2;
        p.relu = 1;
        p.residual_offset = f_e2;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        uint32_t groups = (r2.pixels() + 255) / 256;
        vkCmdDispatch(cmd, groups, 1, 1);
        addBarrier(cmd);
        dispatch_count++;
    }
    writeTS("enc2");

    // --- enc3: StridedConv 128->128 @ r3 ---
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeStridedConv);
        StridedConvPush p = {};
        p.in_channels = 128; p.out_channels = 128;
        p.in_width = r2.w; p.in_height = r2.h;
        p.out_width = r3.w; p.out_height = r3.h;
        p.weight_offset = w_enc3_conv.offset;
        p.bias_offset = w_enc3_bias.offset;
        p.input_offset = f_e2;
        p.output_offset = f_e3;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        uint32_t groups = (r3.pixels() + 255) / 256;
        vkCmdDispatch(cmd, groups, 1, 1);
        addBarrier(cmd);
        dispatch_count++;
    }

    // enc3 DSCBlock(128) at r3
    if (hasFused128) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeFusedDWPW128);
        FusedDWPWPush p = {};
        p.channels = 128; p.width = r3.w; p.height = r3.h;
        p.dw_weight_offset = w_enc3_dw.offset;
        p.pw_weight_offset = w_enc3_pw.offset;
        p.pw_bias_offset = w_enc3_pw_b.offset;
        p.input_offset = f_e3;
        p.output_offset = f_e3;
        p.relu = 1;
        p.residual_offset = f_e3;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        uint32_t groups = (r3.pixels() + 255) / 256;
        vkCmdDispatch(cmd, groups, 1, 1);
        addBarrier(cmd);
        dispatch_count++;
    }
    writeTS("enc3");

    // ============================================================
    // TRANSFORMER (4 blocks of windowed attention @ r3, dim=128)
    // ============================================================

    {
        int window_size = 8;
        int windows_x = (r3.w + window_size - 1) / window_size;  // 15
        int windows_y = (r3.h + window_size - 1) / window_size;  // 9 (ceil(68/8)=9)
        int total_windows = windows_x * windows_y;                // 135

        // Ping-pong between f_e3 and f_t_ping
        int cur_in = f_e3;
        int cur_out = f_t_ping;

        for (int blk = 0; blk < 4; blk++) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeTransformer);
            TransformerPush p = {};
            p.n_tokens = r3.pixels();  // 8160
            p.dim = 128;
            p.n_heads = 4;
            p.head_dim = 32;
            p.spatial_w = r3.w;        // 120
            p.spatial_h = r3.h;        // 68
            p.window_size = window_size;
            p.qkv_w_offset = tw[blk].qkv_w.offset;
            p.qkv_b_offset = tw[blk].qkv_b.offset;
            p.out_w_offset = tw[blk].out_w.offset;
            p.out_b_offset = tw[blk].out_b.offset;
            p.ffn_w1_offset = tw[blk].ffn_w1.offset;
            p.ffn_b1_offset = tw[blk].ffn_b1.offset;
            p.ffn_w2_offset = tw[blk].ffn_w2.offset;
            p.ffn_b2_offset = tw[blk].ffn_b2.offset;
            p.input_offset = cur_in;
            p.output_offset = cur_out;
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);

            // Each workgroup = one 8x8 window of 64 tokens
            vkCmdDispatch(cmd, total_windows, 1, 1);
            addBarrier(cmd);
            dispatch_count++;

            // Swap ping-pong
            int tmp = cur_in; cur_in = cur_out; cur_out = tmp;
        }

        // After 4 blocks (even count), result is back in f_e3
        // (block 0: e3->ping, block 1: ping->e3, block 2: e3->ping, block 3: ping->e3)
    }
    writeTS("transformer");

    // ============================================================
    // DECODER
    // ============================================================

    // --- dec3: TransConv 128->128 @ r3->r2, concat with e2, reduce 256->128, DSC(128) ---
    {
        // TransposeConv: f_e3 (128ch @ r3) -> f_dec_tmp1 (128ch @ r2)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeTransConv);
        TransConvPush p = {};
        p.in_channels = 128; p.out_channels = 128;
        p.in_width = r3.w; p.in_height = r3.h;
        p.out_width = r2.w; p.out_height = r2.h;
        p.weight_offset = w_dec3_tconv.offset;
        p.bias_offset = w_dec3_tconv_b.offset;
        p.input_offset = f_e3;
        p.output_offset = f_dec_tmp1;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        uint32_t gx = (r2.w + 15) / 16;
        uint32_t gy = (r2.h + 15) / 16;
        vkCmdDispatch(cmd, gx, gy, 128);
        addBarrier(cmd);
        dispatch_count++;

        // Concat: [128ch from transconv, 128ch from e2] -> 256ch @ r2
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeConcatSkip);
        ConcatPush cp = {};
        cp.ch_a = 128; cp.ch_b = 128;
        cp.width = r2.w; cp.height = r2.h;
        cp.offset_a = f_dec_tmp1;
        cp.offset_b = f_e2;
        cp.output_offset = f_dec_cat;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cp), &cp);
        uint32_t cat_groups = (256 * r2.pixels() + 255) / 256;
        vkCmdDispatch(cmd, cat_groups, 1, 1);
        addBarrier(cmd);
        dispatch_count++;

        // PW reduce 256->128 (generic, z-dispatch per output channel)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipePWConvGeneric ? pipePWConvGeneric : pipePWConv);
        PWConvPush pw = {};
        pw.in_channels = 256; pw.out_channels = 128;
        pw.width = r2.w; pw.height = r2.h;
        pw.weight_offset = w_dec3_reduce.offset;
        pw.bias_offset = w_dec3_reduce_b.offset;
        pw.input_offset = f_dec_cat;
        pw.output_offset = f_dec3_out;
        pw.relu = 1; pw.residual_offset = -1;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pw), &pw);
        gx = (r2.w + 15) / 16;
        gy = (r2.h + 15) / 16;
        vkCmdDispatch(cmd, gx, gy, 128);
        addBarrier(cmd);
        dispatch_count++;

        // DSCBlock(128) at r2
        if (hasFused128) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeFusedDWPW128);
            FusedDWPWPush fp = {};
            fp.channels = 128; fp.width = r2.w; fp.height = r2.h;
            fp.dw_weight_offset = w_dec3_dw.offset;
            fp.pw_weight_offset = w_dec3_pw.offset;
            fp.pw_bias_offset = w_dec3_pw_b.offset;
            fp.input_offset = f_dec3_out;
            fp.output_offset = f_dec3_out;
            fp.relu = 1; fp.residual_offset = f_dec3_out;
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(fp), &fp);
            uint32_t groups = (r2.pixels() + 255) / 256;
            vkCmdDispatch(cmd, groups, 1, 1);
            addBarrier(cmd);
            dispatch_count++;
        }
    }
    writeTS("dec3");

    // --- dec2: TransConv 128->64 @ r2->r1, concat with e1, reduce 128->64, DSC(64) ---
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeTransConv);
        TransConvPush p = {};
        p.in_channels = 128; p.out_channels = 64;
        p.in_width = r2.w; p.in_height = r2.h;
        p.out_width = r1.w; p.out_height = r1.h;
        p.weight_offset = w_dec2_tconv.offset;
        p.bias_offset = w_dec2_tconv_b.offset;
        p.input_offset = f_dec3_out;
        p.output_offset = f_dec2_tmp;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        uint32_t gx = (r1.w + 15) / 16;
        uint32_t gy = (r1.h + 15) / 16;
        vkCmdDispatch(cmd, gx, gy, 64);
        addBarrier(cmd);
        dispatch_count++;

        // Concat
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeConcatSkip);
        ConcatPush cp = {};
        cp.ch_a = 64; cp.ch_b = 64;
        cp.width = r1.w; cp.height = r1.h;
        cp.offset_a = f_dec2_tmp;
        cp.offset_b = f_e1;
        cp.output_offset = f_dec2_cat;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cp), &cp);
        uint32_t cat_groups = (128 * r1.pixels() + 255) / 256;
        vkCmdDispatch(cmd, cat_groups, 1, 1);
        addBarrier(cmd);
        dispatch_count++;

        // PW reduce 128->64 (generic)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipePWConvGeneric ? pipePWConvGeneric : pipePWConv);
        PWConvPush pw = {};
        pw.in_channels = 128; pw.out_channels = 64;
        pw.width = r1.w; pw.height = r1.h;
        pw.weight_offset = w_dec2_reduce.offset;
        pw.bias_offset = w_dec2_reduce_b.offset;
        pw.input_offset = f_dec2_cat;
        pw.output_offset = f_dec2_out;
        pw.relu = 1; pw.residual_offset = -1;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pw), &pw);
        gx = (r1.w + 15) / 16;
        gy = (r1.h + 15) / 16;
        vkCmdDispatch(cmd, gx, gy, 64);
        addBarrier(cmd);
        dispatch_count++;

        // DSCBlock(64) at r1
        if (hasFused64) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeFusedDWPW64);
            FusedDWPWPush fp = {};
            fp.channels = 64; fp.width = r1.w; fp.height = r1.h;
            fp.dw_weight_offset = w_dec2_dw.offset;
            fp.pw_weight_offset = w_dec2_pw.offset;
            fp.pw_bias_offset = w_dec2_pw_b.offset;
            fp.input_offset = f_dec2_out;
            fp.output_offset = f_dec2_out;
            fp.relu = 1; fp.residual_offset = f_dec2_out;
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(fp), &fp);
            uint32_t groups = (r1.pixels() + 255) / 256;
            vkCmdDispatch(cmd, groups, 1, 1);
            addBarrier(cmd);
            dispatch_count++;
        }
    }
    writeTS("dec2");

    // --- dec1: TransConv 64->32 @ r1->r0, concat with e0, reduce 64->32, DSC(32) ---
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeTransConv);
        TransConvPush p = {};
        p.in_channels = 64; p.out_channels = 32;
        p.in_width = r1.w; p.in_height = r1.h;
        p.out_width = r0.w; p.out_height = r0.h;
        p.weight_offset = w_dec1_tconv.offset;
        p.bias_offset = w_dec1_tconv_b.offset;
        p.input_offset = f_dec2_out;
        p.output_offset = f_dec1_tmp;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        uint32_t gx = (r0.w + 15) / 16;
        uint32_t gy = (r0.h + 15) / 16;
        vkCmdDispatch(cmd, gx, gy, 32);
        addBarrier(cmd);
        dispatch_count++;

        // Concat
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeConcatSkip);
        ConcatPush cp = {};
        cp.ch_a = 32; cp.ch_b = 32;
        cp.width = r0.w; cp.height = r0.h;
        cp.offset_a = f_dec1_tmp;
        cp.offset_b = f_e0;
        cp.output_offset = f_dec1_cat;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cp), &cp);
        uint32_t cat_groups = (64 * r0.pixels() + 255) / 256;
        vkCmdDispatch(cmd, cat_groups, 1, 1);
        addBarrier(cmd);
        dispatch_count++;

        // PW reduce 64->32 (generic)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipePWConvGeneric ? pipePWConvGeneric : pipePWConv);
        PWConvPush pw = {};
        pw.in_channels = 64; pw.out_channels = 32;
        pw.width = r0.w; pw.height = r0.h;
        pw.weight_offset = w_dec1_reduce.offset;
        pw.bias_offset = w_dec1_reduce_b.offset;
        pw.input_offset = f_dec1_cat;
        pw.output_offset = f_dec1_out;
        pw.relu = 1; pw.residual_offset = -1;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pw), &pw);
        gx = (r0.w + 15) / 16;
        gy = (r0.h + 15) / 16;
        vkCmdDispatch(cmd, gx, gy, 32);
        addBarrier(cmd);
        dispatch_count++;

        // DSCBlock(32) at r0
        if (hasFused32) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeFusedDWPW32);
            FusedDWPWPush fp = {};
            fp.channels = 32; fp.width = r0.w; fp.height = r0.h;
            fp.dw_weight_offset = w_dec1_dw.offset;
            fp.pw_weight_offset = w_dec1_pw.offset;
            fp.pw_bias_offset = w_dec1_pw_b.offset;
            fp.input_offset = f_dec1_out;
            fp.output_offset = f_dec1_out;
            fp.relu = 1; fp.residual_offset = f_dec1_out;
            vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(fp), &fp);
            uint32_t groups = (r0.pixels() + 255) / 256;
            vkCmdDispatch(cmd, groups, 1, 1);
            addBarrier(cmd);
            dispatch_count++;
        }
    }
    writeTS("dec1");

    // ============================================================
    // OUTPUT
    // ============================================================

    // Output conv: Conv3x3 32->12 @ r0
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeConv3x3);
        Conv3x3Push p = {};
        p.in_channels = 32; p.out_channels = 12;
        p.width = r0.w; p.height = r0.h;
        p.weight_offset = w_output_conv.offset;
        p.bias_offset = w_output_bias.offset;
        p.input_offset = f_dec1_out;
        p.output_offset = f_output_12;
        p.relu = 0;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        uint32_t gx = (r0.w + 15) / 16;
        uint32_t gy = (r0.h + 15) / 16;
        vkCmdDispatch(cmd, gx, gy, 12);
        addBarrier(cmd);
        dispatch_count++;
    }
    writeTS("output_conv");

    // PixelShuffle(2) + Sigmoid -> 1080x1920
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipePixelShuffle);
        PixelShufflePush p = {};
        p.render_width = r0.w; p.render_height = r0.h;
        p.display_width = rD.w; p.display_height = rD.h;
        p.scale = 2;
        p.input_offset = f_output_12;
        p.output_offset = f_display;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        uint32_t gx = (rD.w + 15) / 16;
        uint32_t gy = (rD.h + 15) / 16;
        vkCmdDispatch(cmd, gx, gy, 1);
        addBarrier(cmd);
        dispatch_count++;
    }
    writeTS("pixelshuffle");

    vkEndCommandBuffer(cmd);
    printf("Command buffer recorded: %d dispatches, %d timestamp markers\n", dispatch_count, ts_idx);

    // ================================================================
    // BENCHMARK
    // ================================================================

    printf("\n--- Warmup (%d iterations) ---\n", warmup);
    for (int i = 0; i < warmup; i++) {
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &si, fence);
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &fence);
    }

    printf("--- Benchmarking (%d iterations) ---\n", loops);
    double total_ms = 0, min_ms = 1e9, max_ms = 0;

    // Accumulate per-stage times
    std::vector<double> stage_totals(ts_idx, 0.0);

    auto wall_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < loops; i++) {
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &si, fence);
        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &fence);

        uint64_t ts[MAX_TIMESTAMPS];
        vkGetQueryPoolResults(device, queryPool, 0, ts_idx, sizeof(ts), ts,
                              sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);

        double frame_ms = (ts[ts_idx-1] - ts[0]) * props.limits.timestampPeriod / 1e6;
        total_ms += frame_ms;
        if (frame_ms < min_ms) min_ms = frame_ms;
        if (frame_ms > max_ms) max_ms = frame_ms;

        for (int s = 1; s < ts_idx; s++) {
            double dt = (ts[s] - ts[s-1]) * props.limits.timestampPeriod / 1e6;
            stage_totals[s] += dt;
        }
    }
    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    // ================================================================
    // RESULTS
    // ================================================================

    printf("\n================================================================\n");
    printf("  RESULTS — Prism V2 Full Pipeline\n");
    printf("  GPU: %s\n", props.deviceName);
    printf("  540x960 -> 1080x1920 (2x upscale)\n");
    printf("  %d total dispatches per frame\n", dispatch_count);
    printf("================================================================\n\n");

    printf("--- Per-stage GPU timing (avg over %d frames) ---\n", loops);
    for (int s = 1; s < ts_idx; s++) {
        double avg = stage_totals[s] / loops;
        printf("  %-20s %6.2f ms\n", stage_names[s] ? stage_names[s] : "???", avg);
    }

    double avg_ms = total_ms / loops;
    printf("\n--- Overall ---\n");
    printf("  GPU time avg:  %6.2f ms  (%.0f FPS)\n", avg_ms, 1000.0 / avg_ms);
    printf("  GPU time min:  %6.2f ms  (%.0f FPS)\n", min_ms, 1000.0 / min_ms);
    printf("  GPU time max:  %6.2f ms\n", max_ms);
    printf("  Wall time avg: %6.2f ms  (includes CPU submit overhead)\n", wall_ms / loops);

    printf("\n--- Budget analysis (16.67ms = 60 FPS) ---\n");
    double budget = 16.67;
    double remaining = budget - avg_ms;
    printf("  Frame budget:  %.2f ms\n", budget);
    printf("  Inference:     %.2f ms\n", avg_ms);
    printf("  Remaining:     %.2f ms (%s)\n", remaining,
           remaining > 0 ? "OK - room for game rendering" : "OVER BUDGET");

    printf("\n--- Memory usage ---\n");
    printf("  Weights: %.2f MB (%d params)\n", total_weights * 2.0 / 1024 / 1024, total_weights);
    printf("  Features: %.2f MB\n", total_feat * 2.0 / 1024 / 1024);
    printf("  Total VRAM: %.2f MB\n", (total_weights + total_feat) * 2.0 / 1024 / 1024);

    // Cleanup
    vkDeviceWaitIdle(device);
    vkDestroyQueryPool(device, queryPool, nullptr);
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, cmdPool, nullptr);
    vkDestroyBuffer(device, weightBuf, nullptr); vkFreeMemory(device, weightMem, nullptr);
    vkDestroyBuffer(device, featBuf, nullptr); vkFreeMemory(device, featMem, nullptr);

    auto destroyPipe = [&](VkPipeline p) { if (p) vkDestroyPipeline(device, p, nullptr); };
    destroyPipe(pipeInputConvV2); destroyPipe(pipeInputConv);
    destroyPipe(pipeStridedConv); destroyPipe(pipeFusedDWPW64);
    destroyPipe(pipeFusedDWPW128); destroyPipe(pipeFusedDWPW32);
    destroyPipe(pipeTransformer); destroyPipe(pipeTransConv);
    destroyPipe(pipeConcatSkip); destroyPipe(pipeConv3x3);
    destroyPipe(pipePWConv); destroyPipe(pipePWConvGeneric); destroyPipe(pipePWConvCoopvec);
    destroyPipe(pipePW64to12); destroyPipe(pipePixelShuffle);
    vkDestroyPipelineLayout(device, pipeLayout, nullptr);
    vkDestroyDescriptorPool(device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    return 0;
}
