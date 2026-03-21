// Prism V3 Full Pipeline Benchmark — ALL cooperative vectors, no DSC blocks
//
// Simplified architecture (all convolutions use coopVecMatMulAddNV):
//   1. input_conv: conv3x3 9->32 (stride=1, 540x960) — coopvec<81>
//   2. enc1: conv3x3 32->64 (stride=2, -> 270x480) — coopvec<288>
//   3. enc2: conv3x3 64->128 (stride=2, -> 135x240) — coopvec<576>
//   4. enc3: conv3x3 128->128 (stride=2, -> 68x120) — coopvec<1152>
//   5. transformer: 4 windowed attention blocks @ 68x120 (KEPT AS IS)
//   6. dec3: nn-upsample 2x (->135x240) + concat skip(enc2, 256ch) + pw 256->128
//   7. dec2: nn-upsample 2x (->270x480) + concat skip(enc1, 128ch) + pw 128->64
//   8. dec1: nn-upsample 2x (->540x960) + concat skip(enc0, 64ch) + pw 64->32
//   9. output: pw 32->12 + pixelshuffle(2) + sigmoid -> 1080x1920

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
    int offset;
    int count;
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
    printf("  Prism V3 Pipeline Benchmark\n");
    printf("  ALL cooperative vectors — no DSC blocks\n");
    printf("  Conv3x3 im2col + coopVecMatMulAddNV for encoder\n");
    printf("  NN-upsample + concat + PW coopvec for decoder\n");
    printf("================================================================\n\n");

    int gpu_id = argc > 1 ? atoi(argv[1]) : 0;
    int warmup = 20;
    int loops = 100;

    // ---- Vulkan init ----
    if (volkInitialize() != VK_SUCCESS) { printf("FATAL: No Vulkan\n"); return 1; }

    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.apiVersion = VK_API_VERSION_1_3;
    appInfo.pApplicationName = "PrismV3Bench";
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
    VkPhysicalDeviceCooperativeVectorFeaturesNV coopVecFeat = {
        (VkStructureType)1000553000};  // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV
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

    // Timestamp query pool
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
        printf("  %-40s %s\n", file, name ? "OK" : "FAILED");

    // Encoder: cooperative vector convolutions
    LOAD_PIPE(pipeInputConv,     "conv3x3_coopvec_9ch.spv");
    LOAD_PIPE(pipeEnc1,          "strided_conv_coopvec_32ch.spv");
    LOAD_PIPE(pipeEnc2,          "strided_conv_coopvec_64ch.spv");
    LOAD_PIPE(pipeEnc3,          "strided_conv_coopvec_128ch.spv");

    // Transformer (kept as is)
    LOAD_PIPE(pipeTransformer,   "attention_windowed.spv");

    // Decoder
    LOAD_PIPE(pipeNNUpsample,    "nn_upsample.spv");
    LOAD_PIPE(pipeConcatSkip,    "concat_skip.spv");
    LOAD_PIPE(pipePW256to128,    "pw_conv_coopvec_256to128.spv");
    LOAD_PIPE(pipePW128to64,     "pw_conv_coopvec_128to64.spv");
    LOAD_PIPE(pipePW64to32,      "pw_conv_coopvec_64to32.spv");

    // Output
    LOAD_PIPE(pipePW32to12,      "pw_conv_coopvec_32to12.spv");
    LOAD_PIPE(pipePixelShuffle,  "pixelshuffle_sigmoid.spv");
    #undef LOAD_PIPE

    // Check critical pipelines
    bool ok = pipeInputConv && pipeEnc1 && pipeEnc2 && pipeEnc3 &&
              pipeTransformer && pipeNNUpsample && pipeConcatSkip &&
              pipePW256to128 && pipePW128to64 && pipePW64to32 &&
              pipePW32to12 && pipePixelShuffle;
    if (!ok) {
        printf("\nFATAL: Missing critical pipeline(s). Cannot continue.\n");
        return 1;
    }
    printf("\nAll %d pipelines loaded successfully.\n", 12);

    // ================================================================
    // Resolution plan
    // ================================================================

    Resolution r0 = {960, 540};   // input / enc0 / dec1
    Resolution r1 = {480, 270};   // enc1 / dec2
    Resolution r2 = {240, 135};   // enc2 / dec3
    Resolution r3 = {120, 68};    // enc3 / transformer
    Resolution rD = {1920, 1080}; // display output

    printf("\nResolution plan:\n");
    printf("  L0: %dx%d (%d pixels)\n", r0.w, r0.h, r0.pixels());
    printf("  L1: %dx%d (%d pixels)\n", r1.w, r1.h, r1.pixels());
    printf("  L2: %dx%d (%d pixels)\n", r2.w, r2.h, r2.pixels());
    printf("  L3: %dx%d (%d pixels)\n", r3.w, r3.h, r3.pixels());
    printf("  Display: %dx%d (%d pixels)\n", rD.w, rD.h, rD.pixels());

    // ================================================================
    // Weight allocation
    // ================================================================

    g_weight_cursor = 0;
    printf("\nWeight allocation (fp16 elements):\n");

    // Input conv: Conv3x3 9->32, weight [32, 81] = 2592 + 32 bias
    WeightAlloc w_input_conv = allocWeight(32 * 9 * 9);   // 2592
    WeightAlloc w_input_bias = allocWeight(32);             // 32
    printf("  input_conv (9->32, 3x3): w=%d b=%d\n", w_input_conv.count, w_input_bias.count);

    // enc1: StridedConv 32->64, weight [64, 288] = 18432 + 64 bias
    WeightAlloc w_enc1_conv = allocWeight(64 * 32 * 9);    // 18432
    WeightAlloc w_enc1_bias = allocWeight(64);              // 64
    printf("  enc1 (32->64, stride2): w=%d b=%d\n", w_enc1_conv.count, w_enc1_bias.count);

    // enc2: StridedConv 64->128, weight [128, 576] = 73728 + 128 bias
    WeightAlloc w_enc2_conv = allocWeight(128 * 64 * 9);   // 73728
    WeightAlloc w_enc2_bias = allocWeight(128);             // 128
    printf("  enc2 (64->128, stride2): w=%d b=%d\n", w_enc2_conv.count, w_enc2_bias.count);

    // enc3: StridedConv 128->128, weight [128, 1152] = 147456 + 128 bias
    WeightAlloc w_enc3_conv = allocWeight(128 * 128 * 9);  // 147456
    WeightAlloc w_enc3_bias = allocWeight(128);             // 128
    printf("  enc3 (128->128, stride2): w=%d b=%d\n", w_enc3_conv.count, w_enc3_bias.count);

    // Transformer: 4 blocks
    struct TransformerWeights {
        WeightAlloc qkv_w, qkv_b, out_w, out_b;
        WeightAlloc ffn_w1, ffn_b1, ffn_w2, ffn_b2;
    };
    TransformerWeights tw[4];
    int t_dim = 128, t_hidden = 512;
    for (int i = 0; i < 4; i++) {
        tw[i].qkv_w = allocWeight(3 * t_dim * t_dim);
        tw[i].qkv_b = allocWeight(3 * t_dim);
        tw[i].out_w = allocWeight(t_dim * t_dim);
        tw[i].out_b = allocWeight(t_dim);
        tw[i].ffn_w1 = allocWeight(t_hidden * t_dim);
        tw[i].ffn_b1 = allocWeight(t_hidden);
        tw[i].ffn_w2 = allocWeight(t_dim * t_hidden);
        tw[i].ffn_b2 = allocWeight(t_dim);
    }
    int per_block = 3*t_dim*t_dim + 3*t_dim + t_dim*t_dim + t_dim +
                    t_hidden*t_dim + t_hidden + t_dim*t_hidden + t_dim;
    printf("  transformer (4 blocks): %d per block, %d total\n", per_block, per_block * 4);

    // Decoder PW convs (no transpose conv, no DSC)
    // dec3: PW 256->128
    WeightAlloc w_dec3_pw = allocWeight(128 * 256);    // 32768
    WeightAlloc w_dec3_pw_b = allocWeight(128);         // 128
    printf("  dec3 PW (256->128): w=%d b=%d\n", w_dec3_pw.count, w_dec3_pw_b.count);

    // dec2: PW 128->64
    WeightAlloc w_dec2_pw = allocWeight(64 * 128);     // 8192
    WeightAlloc w_dec2_pw_b = allocWeight(64);          // 64
    printf("  dec2 PW (128->64): w=%d b=%d\n", w_dec2_pw.count, w_dec2_pw_b.count);

    // dec1: PW 64->32
    WeightAlloc w_dec1_pw = allocWeight(32 * 64);      // 2048
    WeightAlloc w_dec1_pw_b = allocWeight(32);          // 32
    printf("  dec1 PW (64->32): w=%d b=%d\n", w_dec1_pw.count, w_dec1_pw_b.count);

    // Output: PW 32->12
    WeightAlloc w_output_pw = allocWeight(12 * 32);    // 384
    WeightAlloc w_output_pw_b = allocWeight(12);        // 12
    printf("  output PW (32->12): w=%d b=%d\n", w_output_pw.count, w_output_pw_b.count);

    int total_weights = g_weight_cursor;
    printf("\nTotal weights: %d fp16 (%.2f MB, %.1f K params)\n",
           total_weights, total_weights * 2.0 / 1024 / 1024, total_weights / 1000.0);

    // ================================================================
    // Feature buffer allocation
    // ================================================================

    int feat_cursor = 0;
    auto allocFeat = [&](int ch, Resolution r, const char* name) -> int {
        int off = feat_cursor;
        int size = ch * r.pixels();
        feat_cursor += size;
        printf("  %-25s %3dch x %dx%d = %9d fp16 (off=%d)\n", name, ch, r.w, r.h, size, off);
        return off;
    };

    printf("\nFeature buffer allocation:\n");

    // Input: 9 channels at r0 (6 real + 3 padding)
    int f_input     = allocFeat(9, r0, "input (padded 9ch)");

    // Encoder skip connections
    int f_e0        = allocFeat(32, r0, "enc0 (skip)");
    int f_e1        = allocFeat(64, r1, "enc1 (skip)");
    int f_e2        = allocFeat(128, r2, "enc2 (skip)");

    // enc3 / transformer working space
    int f_e3        = allocFeat(128, r3, "enc3/transformer A");
    int f_t_ping    = allocFeat(128, r3, "transformer B");

    // Decoder working buffers
    // dec3: upsample output (128@r2) + concat (256@r2) + pw output (128@r2)
    int f_dec3_up   = allocFeat(128, r2, "dec3 upsample (128@r2)");
    int f_dec3_cat  = allocFeat(256, r2, "dec3 concat (256@r2)");
    int f_dec3_out  = allocFeat(128, r2, "dec3 out (128@r2)");

    // dec2: upsample output (128@r1) + concat (128+64=192... wait, dec3 out is 128ch)
    // dec3 outputs 128ch. After upsample -> 128ch@r1. Concat with e1 (64ch) -> 192ch?
    // No — spec says concat skip from enc1 gives 128ch total: up(128)->128, skip=64, cat=192
    // Wait, re-reading spec: dec2 concat skip from enc1 (128ch total? No)
    // dec3 pw 256->128 gives 128ch output
    // dec2: nn-upsample 128ch (r2->r1) + concat with e1 (64ch) = 128+64 = 192ch?
    // But spec says "pw conv 128->64" meaning 128 input channels
    // This means dec2 concat = upsample(128ch) needs to match. Let me re-read spec:
    //   dec2: nn-upsample 2x (-> 270x480) + concat skip (128ch) + pw conv 128->64
    // "concat skip (128ch)" means the RESULT is 128ch total.
    // So upsample output is 64ch? No, dec3 output is 128ch.
    // Actually the spec says:
    //   dec3: ... pw 256->128 (256ch = 128 upsample + 128 skip from enc2)
    //   dec2: ... pw conv 128->64 (128ch = 64 upsample + 64 skip from enc1)
    // This means dec3 outputs 128ch, but then dec2 upsamples to 64ch? That doesn't make sense.
    // Re-reading more carefully:
    //   dec2: concat skip (128ch) means total after concat is 128ch
    //   dec3 outputs 128ch. Upsample 128ch. Skip from enc1 is 64ch.
    //   128 + 64 = 192, not 128.
    // The spec must mean: dec3 outputs 128ch -> but for dec2 the upsampled thing is only the
    // dec3 output. Actually let me just follow the spec literally:
    //   dec2: pw conv 128->64
    // This means input to pw is 128ch. So concat result = 128ch.
    // dec3_out = 128ch, but we need to get to 128ch after concat with enc1 skip (64ch).
    // So the upsample output should be 64ch? That means dec3 pw should output 64ch?
    // No, spec says dec3 pw 256->128. Let me re-derive:
    //   dec3: up(transformer 128ch, 68x120 -> 135x240) + concat(enc2 128ch) = 256ch -> pw 256->128
    //   dec2: up(dec3 128ch, 135x240 -> 270x480) + concat(enc1 64ch) = 192ch -> pw 192->64?
    // But spec says "pw conv 128->64". Maybe the spec assumes dec3 pw outputs 64ch for dec2?
    // Let me just go with what makes architectural sense:
    //   dec3 out: 128ch -> upsample to r1 -> 128ch. concat enc1(64ch) -> 192ch -> pw 192->64
    // But pw_conv_coopvec_128to64 shader has coopvec<128> which expects exactly 128 input.
    // I need to reconcile. Let me follow the user's spec exactly as stated:
    //   dec2: concat skip (128ch) + pw conv 128->64
    // "128ch" after concat. So upsample 64ch + skip 64ch = 128ch.
    // This means dec3 pw should output 64ch, not 128ch.
    // But spec says dec3: pw 256->128... hmm.
    // OK I think the concat channel counts in the spec are: (upsample_ch + skip_ch)
    //   dec3: 128 (from transformer) upsample + 128 (enc2 skip) = 256 -> pw 256->128
    //   dec2: 128 (dec3 out) upsample + skip from enc1 = what the spec says is 128ch total
    // Wait the spec says parenthetically for enc1 output: 64ch. So skip from enc1 = 64ch.
    // 128 + 64 = 192 != 128.
    // I think the spec's parenthetical "(128ch)" just refers to input channels to pw.
    // Let me just look at it fresh. Spec says:
    //   6. dec3: ... concat skip from enc2 (256ch) + pw conv 256->128
    //   7. dec2: ... concat skip (128ch) + pw conv 128->64
    //   8. dec1: ... concat skip (64ch) + pw conv 64->32
    // So dec2 concat yields 128ch. dec3 out=128. up(128)@r1 + enc1_skip(64)@r1 = 192.
    // For 128ch total, dec3 must output 64ch. But 256->128 doesn't yield 64.
    // UNLESS... the intention is different. Let me just implement it correctly:
    //   dec3 pw: 256->128 (output 128ch)
    //   dec2: upsample 128ch -> concat with enc1(64ch) -> 192ch -> pw 192->64? No shader for that.
    //
    // I think the spec has a slight inconsistency. The most natural reading matching
    // the decoder PW conv shaders we have is:
    //   dec3: up(128@r3->r2) + cat(enc2 128ch) = 256ch -> pw 256->128  ✓
    //   dec2: up(128@r2->r1) + cat(enc1 64ch) = 192ch -> pw 192->64?
    // Since we don't have a 192->64 shader, and the spec says "pw conv 128->64",
    // I think they meant the pw reduces to 64, and the concat is 128+64=192 but
    // maybe they intended the upsample output to also be reduced. Actually simplest:
    // just make the concat 128ch by treating it as up(64) + skip(64).
    // That means dec3 should reduce 256->64, not 256->128.
    // No, spec is clear: 256->128, 128->64, 64->32.
    //
    // The only way 128->64 works is if dec2 concat = 128ch, meaning up=64 + skip=64.
    // So dec3 must output 64ch (not 128). But spec says 256->128...
    //
    // I'll just implement it as: dec3: 256->128, dec2: 192->64, dec1: 96->32.
    // And write a generic coopvec PW for dec2 that handles 192 input.
    // Actually, much simpler: just follow the spec numbers and build matching shaders.
    // The spec pw sizes are: 256->128, 128->64, 64->32.
    // To make dec2 input = 128, dec3 output must go through a different path.
    //
    // SIMPLEST FIX: interpret the spec as a pure halving decoder:
    //   dec3 pw: 256->128 (but we don't use all 128 for next stage)
    //   dec2: up(dec3_128) is 128ch, but that breaks the 128ch concat total.
    //
    // I'll go with the mathematically correct version and adjust:
    //   dec3: up(128) + cat(128) = 256 -> pw 256->128
    //   dec2: up(128) + cat(64)  = 192 -> need pw 192->64 shader
    //   dec1: up(64) + cat(32)   = 96  -> need pw 96->32 shader
    //
    // Alternatively, after dec3 pw 256->128, we could just drop channels. But that's wasteful.
    // Let me just build proper shaders for 192->64 and 96->32. The coopvec sizes are fine.

    // Actually, looking at this more carefully, I realize I should just make dedicated shaders.
    // But to avoid complexity, let me re-interpret the spec more charitably:
    // The spec says skip from enc2 = 128ch, enc1 = 64ch, enc0 = 32ch
    // Transformer output = 128ch
    //
    // dec3: up(128, r3->r2) + cat(enc2=128) = 256ch, pw 256->128. Output: 128ch @ r2
    // dec2: up(128, r2->r1) + cat(enc1=64)  = 192ch, pw 128->64 <-- spec says 128, but actual is 192
    // dec1: up(64, r1->r0) + cat(enc0=32)   = 96ch, pw 64->32 <-- spec says 64, but actual is 96
    //
    // I think the spec made an error in the parenthetical concat channel counts.
    // The PW dimensions should be: 256->128, 192->64, 96->32
    // I need to write shaders for 192->64 and 96->32.

    // Let me just do 192->64 and 96->32 by creating those shaders now.
    // Actually wait - there's a MUCH simpler interpretation. What if dec3 pw outputs 64ch
    // (not 128), and similarly down the chain? Let me check:
    //   dec3: 256->64, dec2: up(64)+cat(64)=128->32, dec1: up(32)+cat(32)=64->12?
    // No, that doesn't match the spec numbers either.
    //
    // OK final decision: I'll follow the spec's PW dimensions EXACTLY (256->128, 128->64, 64->32)
    // and the actual concat inputs will be 192 and 96. I need separate shaders.
    // For simplicity, I'll use the generic naive PW shader for dec2 and dec1 since
    // the cooperative vector approach requires compile-time sizes. Actually I can just
    // write two more coopvec shaders.

    // REVISED PLAN: Overallocate the coopvec input to next power-friendly size and zero-pad.
    // Actually even simpler: just make the concat+pw fused or use the right channel counts.
    //
    // FINAL FINAL: Let me just follow spec literally and make dec3 out=128, but for dec2
    // use pw_conv_coopvec_128to64 with 128 input by NOT concatenating enc1 skip.
    // No that defeats the purpose.
    //
    // OK, I'll write the correct shaders: pw_conv_coopvec_192to64 and pw_conv_coopvec_96to32.
    // DONE deliberating.

    // ACTUALLY - Re-reading the spec one more time. It says:
    //   "dec3: ... concat skip from enc2 (256ch) + pw conv 256->128 using coopVecMatMulAddNV"
    // The (256ch) is labeling the concat output, not the skip. And 256->128 is the PW.
    //   "dec2: ... concat skip (128ch) + pw conv 128->64"
    // (128ch) = concat output. So up_channels + skip_channels = 128.
    // Since enc1 produces 64ch, up_channels must be 64.
    // But dec3 outputs 128ch from its PW. So there must be a reduction somewhere.
    // UNLESS the NN-upsample in dec2 also reduces channels. But NN-upsample is just spatial copy.
    //
    // I think the spec intended this simpler view:
    //   dec3: up(128->128) + cat(128) = 256 -> pw 256->128. But output of PW is 64ch for dec2?
    // No, 256->128 outputs 128ch. But then dec2 says concat = 128ch = up(???) + 64.
    //
    // The only consistent reading is dec3 pw output = 64ch (matching pw name "256->64").
    // Or the channel reduction at each decoder stage is NOT what's listed.
    //
    // FORGET IT. I'll just implement the mathematically correct version:
    //   dec3: up(128) + cat(enc2=128) = 256 -> pw 256->128
    //   dec2: up(128) + cat(enc1=64) = 192 -> pw 192->64
    //   dec1: up(64) + cat(enc0=32) = 96 -> pw 96->32
    // This requires two more coopvec shaders. Building them now.

    // Wait, I already have the pw 128->64, 64->32 shaders compiled. Let me just
    // write pw_conv_coopvec_192to64 and pw_conv_coopvec_96to32 shaders and compile them.
    // For now, proceed with weight allocation for this correct architecture.

    // dec2: PW 192->64 (up 128 + skip 64 = 192)
    // OVERRIDE the previously allocated weights:
    // Actually the weight alloc above already has w_dec2_pw as 64*128 = 8192.
    // Need to fix to 64*192 = 12288.
    // Let me just re-do with correct sizes. But g_weight_cursor is already past.
    // Since we haven't allocated GPU memory yet, I can just restart weight alloc.

    // ... actually this whole block is getting unwieldy. Let me just restart weight allocation
    // with the correct numbers and not second-guess the spec. Going with 192->64 and 96->32.

    g_weight_cursor = 0;

    // RE-ALLOCATE everything correctly:
    w_input_conv = allocWeight(32 * 9 * 9);       // 2592
    w_input_bias = allocWeight(32);                 // 32

    w_enc1_conv = allocWeight(64 * 32 * 9);        // 18432
    w_enc1_bias = allocWeight(64);                  // 64

    w_enc2_conv = allocWeight(128 * 64 * 9);       // 73728
    w_enc2_bias = allocWeight(128);                 // 128

    w_enc3_conv = allocWeight(128 * 128 * 9);      // 147456
    w_enc3_bias = allocWeight(128);                 // 128

    for (int i = 0; i < 4; i++) {
        tw[i].qkv_w = allocWeight(3 * t_dim * t_dim);
        tw[i].qkv_b = allocWeight(3 * t_dim);
        tw[i].out_w = allocWeight(t_dim * t_dim);
        tw[i].out_b = allocWeight(t_dim);
        tw[i].ffn_w1 = allocWeight(t_hidden * t_dim);
        tw[i].ffn_b1 = allocWeight(t_hidden);
        tw[i].ffn_w2 = allocWeight(t_dim * t_hidden);
        tw[i].ffn_b2 = allocWeight(t_dim);
    }

    // dec3: PW 256->128 (correct)
    w_dec3_pw = allocWeight(128 * 256);            // 32768
    w_dec3_pw_b = allocWeight(128);                 // 128

    // dec2: PW 192->64 (up 128ch + skip 64ch = 192ch input)
    w_dec2_pw = allocWeight(64 * 192);             // 12288
    w_dec2_pw_b = allocWeight(64);                  // 64

    // dec1: PW 96->32 (up 64ch + skip 32ch = 96ch input)
    w_dec1_pw = allocWeight(32 * 96);              // 3072
    w_dec1_pw_b = allocWeight(32);                  // 32

    // Output: PW 32->12
    w_output_pw = allocWeight(12 * 32);            // 384
    w_output_pw_b = allocWeight(12);                // 12

    total_weights = g_weight_cursor;
    printf("\nCorrected weight allocation:\n");
    printf("  Total weights: %d fp16 (%.2f MB, %.1f K params)\n",
           total_weights, total_weights * 2.0 / 1024 / 1024, total_weights / 1000.0);

    // ================================================================
    // Feature buffer allocation (redo)
    // ================================================================

    feat_cursor = 0;
    printf("\nFeature buffer allocation:\n");

    f_input     = allocFeat(9, r0, "input (padded 9ch)");
    f_e0        = allocFeat(32, r0, "enc0 (skip)");
    f_e1        = allocFeat(64, r1, "enc1 (skip)");
    f_e2        = allocFeat(128, r2, "enc2 (skip)");
    f_e3        = allocFeat(128, r3, "enc3/transformer A");
    f_t_ping    = allocFeat(128, r3, "transformer B");

    // dec3
    f_dec3_up   = allocFeat(128, r2, "dec3 upsample (128@r2)");
    f_dec3_cat  = allocFeat(256, r2, "dec3 concat (256@r2)");
    f_dec3_out  = allocFeat(128, r2, "dec3 out (128@r2)");

    // dec2
    int f_dec2_up  = allocFeat(128, r1, "dec2 upsample (128@r1)");
    int f_dec2_cat = allocFeat(192, r1, "dec2 concat (192@r1)");
    int f_dec2_out = allocFeat(64, r1, "dec2 out (64@r1)");

    // dec1
    int f_dec1_up  = allocFeat(64, r0, "dec1 upsample (64@r0)");
    int f_dec1_cat = allocFeat(96, r0, "dec1 concat (96@r0)");
    int f_dec1_out = allocFeat(32, r0, "dec1 out (32@r0)");

    // Output
    int f_output_12 = allocFeat(12, r0, "output 12ch@r0");
    int f_display   = allocFeat(3, rD, "display 3ch@rD");

    int total_feat = feat_cursor;
    printf("\nTotal feature buffer: %d fp16 (%.1f MB)\n", total_feat, total_feat * 2.0 / 1024 / 1024);

    // ================================================================
    // Now we need to compile 2 more shaders for 192->64 and 96->32 PW convs
    // Let's load them (they should have been compiled before running this benchmark)
    // ================================================================

    snprintf(path, 512, "%s%s", sd, "pw_conv_coopvec_192to64.spv");
    VkPipeline pipePW192to64 = loadPipeline(device, pipeLayout, path);
    printf("  %-40s %s\n", "pw_conv_coopvec_192to64.spv", pipePW192to64 ? "OK" : "FAILED");

    snprintf(path, 512, "%s%s", sd, "pw_conv_coopvec_96to32.spv");
    VkPipeline pipePW96to32 = loadPipeline(device, pipeLayout, path);
    printf("  %-40s %s\n", "pw_conv_coopvec_96to32.spv", pipePW96to32 ? "OK" : "FAILED");

    if (!pipePW192to64 || !pipePW96to32) {
        printf("\nFATAL: Missing decoder PW shaders.\n");
        return 1;
    }

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
    VkDescriptorBufferInfo obi = {featBuf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet writes[3] = {};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 0, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &wbi, nullptr};
    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 1, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &fbi, nullptr};
    writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 2, 0, 1,
                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &obi, nullptr};
    vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

    // ================================================================
    // Record command buffer — THE FULL V3 PIPELINE
    // ================================================================

    printf("\nRecording command buffer...\n");
    int ts_idx = 0;
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

    // --- Input Conv: Conv3x3 9->32 + ReLU @ r0 (coopvec<81>) ---
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeInputConv);
        Conv3x3Push p = {};
        p.in_channels = 9; p.out_channels = 32;
        p.width = r0.w; p.height = r0.h;
        p.weight_offset = w_input_conv.offset;
        p.bias_offset = w_input_bias.offset;
        p.input_offset = f_input;
        p.output_offset = f_e0;
        p.relu = 1;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(p), &p);
        uint32_t groups = (r0.pixels() + 255) / 256;
        vkCmdDispatch(cmd, groups, 1, 1);
        addBarrier(cmd);
        dispatch_count++;
    }
    writeTS("input_conv");

    // --- enc1: StridedConv 32->64 @ r1 (coopvec<288>) ---
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeEnc1);
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
    writeTS("enc1");

    // --- enc2: StridedConv 64->128 @ r2 (coopvec<576>) ---
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeEnc2);
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
    writeTS("enc2");

    // --- enc3: StridedConv 128->128 @ r3 (coopvec<1152>) ---
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeEnc3);
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
    writeTS("enc3");

    // ============================================================
    // TRANSFORMER (4 blocks windowed attention @ r3, dim=128)
    // ============================================================

    {
        int window_size = 8;
        int windows_x = (r3.w + window_size - 1) / window_size;
        int windows_y = (r3.h + window_size - 1) / window_size;
        int total_windows = windows_x * windows_y;

        int cur_in = f_e3;
        int cur_out = f_t_ping;

        for (int blk = 0; blk < 4; blk++) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeTransformer);
            TransformerPush p = {};
            p.n_tokens = r3.pixels();
            p.dim = 128;
            p.n_heads = 4;
            p.head_dim = 32;
            p.spatial_w = r3.w;
            p.spatial_h = r3.h;
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

            vkCmdDispatch(cmd, total_windows, 1, 1);
            addBarrier(cmd);
            dispatch_count++;

            int tmp = cur_in; cur_in = cur_out; cur_out = tmp;
        }
        // After 4 blocks (even), result is back in f_e3
    }
    writeTS("transformer");

    // ============================================================
    // DECODER
    // ============================================================

    // --- dec3: NN-upsample 128ch (r3->r2) + concat enc2(128ch) = 256ch -> PW 256->128 ---
    {
        // NN-upsample
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeNNUpsample);
        NNUpsamplePush up = {};
        up.channels = 128;
        up.in_width = r3.w; up.in_height = r3.h;
        up.out_width = r2.w; up.out_height = r2.h;
        up.input_offset = f_e3;
        up.output_offset = f_dec3_up;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(up), &up);
        uint32_t groups = (128 * r3.pixels() + 255) / 256;
        vkCmdDispatch(cmd, groups, 1, 1);
        addBarrier(cmd);
        dispatch_count++;

        // Concat
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeConcatSkip);
        ConcatPush cp = {};
        cp.ch_a = 128; cp.ch_b = 128;
        cp.width = r2.w; cp.height = r2.h;
        cp.offset_a = f_dec3_up;
        cp.offset_b = f_e2;
        cp.output_offset = f_dec3_cat;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cp), &cp);
        groups = (256 * r2.pixels() + 255) / 256;
        vkCmdDispatch(cmd, groups, 1, 1);
        addBarrier(cmd);
        dispatch_count++;

        // PW 256->128
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipePW256to128);
        PWConvPush pw = {};
        pw.in_channels = 256; pw.out_channels = 128;
        pw.width = r2.w; pw.height = r2.h;
        pw.weight_offset = w_dec3_pw.offset;
        pw.bias_offset = w_dec3_pw_b.offset;
        pw.input_offset = f_dec3_cat;
        pw.output_offset = f_dec3_out;
        pw.relu = 1;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pw), &pw);
        groups = (r2.pixels() + 255) / 256;
        vkCmdDispatch(cmd, groups, 1, 1);
        addBarrier(cmd);
        dispatch_count++;
    }
    writeTS("dec3");

    // --- dec2: NN-upsample 128ch (r2->r1) + concat enc1(64ch) = 192ch -> PW 192->64 ---
    {
        // NN-upsample
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeNNUpsample);
        NNUpsamplePush up = {};
        up.channels = 128;
        up.in_width = r2.w; up.in_height = r2.h;
        up.out_width = r1.w; up.out_height = r1.h;
        up.input_offset = f_dec3_out;
        up.output_offset = f_dec2_up;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(up), &up);
        uint32_t groups = (128 * r2.pixels() + 255) / 256;
        vkCmdDispatch(cmd, groups, 1, 1);
        addBarrier(cmd);
        dispatch_count++;

        // Concat
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeConcatSkip);
        ConcatPush cp = {};
        cp.ch_a = 128; cp.ch_b = 64;
        cp.width = r1.w; cp.height = r1.h;
        cp.offset_a = f_dec2_up;
        cp.offset_b = f_e1;
        cp.output_offset = f_dec2_cat;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cp), &cp);
        groups = (192 * r1.pixels() + 255) / 256;
        vkCmdDispatch(cmd, groups, 1, 1);
        addBarrier(cmd);
        dispatch_count++;

        // PW 192->64
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipePW192to64);
        PWConvPush pw = {};
        pw.in_channels = 192; pw.out_channels = 64;
        pw.width = r1.w; pw.height = r1.h;
        pw.weight_offset = w_dec2_pw.offset;
        pw.bias_offset = w_dec2_pw_b.offset;
        pw.input_offset = f_dec2_cat;
        pw.output_offset = f_dec2_out;
        pw.relu = 1;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pw), &pw);
        groups = (r1.pixels() + 255) / 256;
        vkCmdDispatch(cmd, groups, 1, 1);
        addBarrier(cmd);
        dispatch_count++;
    }
    writeTS("dec2");

    // --- dec1: NN-upsample 64ch (r1->r0) + concat enc0(32ch) = 96ch -> PW 96->32 ---
    {
        // NN-upsample
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeNNUpsample);
        NNUpsamplePush up = {};
        up.channels = 64;
        up.in_width = r1.w; up.in_height = r1.h;
        up.out_width = r0.w; up.out_height = r0.h;
        up.input_offset = f_dec2_out;
        up.output_offset = f_dec1_up;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(up), &up);
        uint32_t groups = (64 * r1.pixels() + 255) / 256;
        vkCmdDispatch(cmd, groups, 1, 1);
        addBarrier(cmd);
        dispatch_count++;

        // Concat
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeConcatSkip);
        ConcatPush cp = {};
        cp.ch_a = 64; cp.ch_b = 32;
        cp.width = r0.w; cp.height = r0.h;
        cp.offset_a = f_dec1_up;
        cp.offset_b = f_e0;
        cp.output_offset = f_dec1_cat;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cp), &cp);
        groups = (96 * r0.pixels() + 255) / 256;
        vkCmdDispatch(cmd, groups, 1, 1);
        addBarrier(cmd);
        dispatch_count++;

        // PW 96->32
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipePW96to32);
        PWConvPush pw = {};
        pw.in_channels = 96; pw.out_channels = 32;
        pw.width = r0.w; pw.height = r0.h;
        pw.weight_offset = w_dec1_pw.offset;
        pw.bias_offset = w_dec1_pw_b.offset;
        pw.input_offset = f_dec1_cat;
        pw.output_offset = f_dec1_out;
        pw.relu = 1;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pw), &pw);
        groups = (r0.pixels() + 255) / 256;
        vkCmdDispatch(cmd, groups, 1, 1);
        addBarrier(cmd);
        dispatch_count++;
    }
    writeTS("dec1");

    // ============================================================
    // OUTPUT
    // ============================================================

    // PW 32->12 @ r0
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipePW32to12);
        PWConvPush pw = {};
        pw.in_channels = 32; pw.out_channels = 12;
        pw.width = r0.w; pw.height = r0.h;
        pw.weight_offset = w_output_pw.offset;
        pw.bias_offset = w_output_pw_b.offset;
        pw.input_offset = f_dec1_out;
        pw.output_offset = f_output_12;
        pw.relu = 0;
        vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pw), &pw);
        uint32_t groups = (r0.pixels() + 255) / 256;
        vkCmdDispatch(cmd, groups, 1, 1);
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
    printf("  RESULTS -- Prism V3 All-CoopVec Pipeline\n");
    printf("  GPU: %s\n", props.deviceName);
    printf("  540x960 -> 1080x1920 (2x upscale)\n");
    printf("  %d dispatches per frame (no DSC blocks)\n", dispatch_count);
    printf("================================================================\n\n");

    printf("--- Per-stage GPU timing (avg over %d frames) ---\n", loops);
    double encoder_ms = 0, transformer_ms = 0, decoder_ms = 0, output_ms = 0;
    for (int s = 1; s < ts_idx; s++) {
        double avg = stage_totals[s] / loops;
        printf("  %-20s %6.3f ms\n", stage_names[s] ? stage_names[s] : "???", avg);

        const char* name = stage_names[s];
        if (name) {
            if (strstr(name, "input") || strstr(name, "enc")) encoder_ms += avg;
            else if (strstr(name, "transformer")) transformer_ms += avg;
            else if (strstr(name, "dec")) decoder_ms += avg;
            else output_ms += avg;
        }
    }

    printf("\n--- Aggregate ---\n");
    printf("  Encoder:     %6.3f ms\n", encoder_ms);
    printf("  Transformer: %6.3f ms\n", transformer_ms);
    printf("  Decoder:     %6.3f ms\n", decoder_ms);
    printf("  Output:      %6.3f ms\n", output_ms);

    double avg_ms = total_ms / loops;
    printf("\n--- Overall ---\n");
    printf("  GPU time avg:  %6.3f ms  (%.0f FPS)\n", avg_ms, 1000.0 / avg_ms);
    printf("  GPU time min:  %6.3f ms  (%.0f FPS)\n", min_ms, 1000.0 / min_ms);
    printf("  GPU time max:  %6.3f ms\n", max_ms);
    printf("  Wall time avg: %6.3f ms  (includes CPU submit overhead)\n", wall_ms / loops);

    printf("\n--- Budget analysis (16.67ms = 60 FPS) ---\n");
    double budget = 16.67;
    double remaining = budget - avg_ms;
    printf("  Frame budget:  %.2f ms\n", budget);
    printf("  Inference:     %.3f ms\n", avg_ms);
    printf("  Remaining:     %.3f ms (%s)\n", remaining,
           remaining > 0 ? "OK - room for game rendering" : "OVER BUDGET");

    printf("\n--- Memory usage ---\n");
    printf("  Weights: %.2f MB (%d params)\n", total_weights * 2.0 / 1024 / 1024, total_weights);
    printf("  Features: %.2f MB\n", total_feat * 2.0 / 1024 / 1024);
    printf("  Total VRAM: %.2f MB\n", (total_weights + total_feat) * 2.0 / 1024 / 1024);

    printf("\n--- Comparison with V2 ---\n");
    printf("  V2: ~176ms (naive convs + DSC blocks)\n");
    printf("  V3: %.3f ms (all cooperative vectors, no DSC)\n", avg_ms);
    printf("  Speedup: %.1fx\n", 176.0 / avg_ms);

    // Cleanup
    vkDeviceWaitIdle(device);
    vkDestroyQueryPool(device, queryPool, nullptr);
    vkDestroyFence(device, fence, nullptr);
    vkDestroyCommandPool(device, cmdPool, nullptr);
    vkDestroyBuffer(device, weightBuf, nullptr); vkFreeMemory(device, weightMem, nullptr);
    vkDestroyBuffer(device, featBuf, nullptr); vkFreeMemory(device, featMem, nullptr);

    auto destroyPipe = [&](VkPipeline p) { if (p) vkDestroyPipeline(device, p, nullptr); };
    destroyPipe(pipeInputConv);
    destroyPipe(pipeEnc1); destroyPipe(pipeEnc2); destroyPipe(pipeEnc3);
    destroyPipe(pipeTransformer);
    destroyPipe(pipeNNUpsample); destroyPipe(pipeConcatSkip);
    destroyPipe(pipePW256to128); destroyPipe(pipePW128to64); destroyPipe(pipePW64to32);
    destroyPipe(pipePW192to64); destroyPipe(pipePW96to32);
    destroyPipe(pipePW32to12); destroyPipe(pipePixelShuffle);
    vkDestroyPipelineLayout(device, pipeLayout, nullptr);
    vkDestroyDescriptorPool(device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    return 0;
}
