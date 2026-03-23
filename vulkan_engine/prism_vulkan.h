#pragma once
/*
 * PrismVulkan — Native Vulkan compute inference engine
 *
 * Zero CPU overhead during inference:
 * - ALL dispatches recorded in ONE command buffer
 * - Pipeline barriers between dispatches (GPU-only, no CPU sync)
 * - Pre-allocated buffers, pre-recorded command buffer
 * - Only vkQueueSubmit + vkWaitForFences per frame
 *
 * This is how DLSS/FSR Neural work internally.
 *
 * Supports two modes:
 * - Standalone: creates own Vulkan instance/device (for benchmarks/testing)
 * - External: accepts external VkDevice + VkQueue (for OptiScaler integration)
 */

#define VK_USE_PLATFORM_WIN32_KHR
#include "deps/volk.h"

#include <vector>
#include <string>
#include <cstdint>

namespace prism {

struct PrismVulkanConfig {
    int channels = 64;       // feature channels
    int n_blocks = 4;        // DSC blocks
    int scale = 2;           // upscale factor
    int render_w = 960;
    int render_h = 540;
    int gpu_id = 0;
    std::string shader_dir = "shaders/";  // path to SPIR-V shader files
};

// Push constants for each shader type
struct Conv3x3Push {
    int32_t in_channels, out_channels, width, height;
    int32_t weight_offset, bias_offset, input_offset, output_offset;
    int32_t relu;
};

struct DWConv3x3Push {
    int32_t channels, width, height;
    int32_t weight_offset, bias_offset, input_offset, output_offset;
    int32_t relu;
};

struct PWConvPush {
    int32_t in_channels, out_channels, width, height;
    int32_t weight_offset, bias_offset, input_offset, output_offset;
    int32_t relu, residual_offset;
};

struct FusedDWPWPush {
    int32_t channels, width, height;
    int32_t dw_weight_offset, pw_weight_offset, pw_bias_offset;
    int32_t input_offset, output_offset;
    int32_t relu, residual_offset;
};

struct ConvertPush {
    int32_t channels, width, height;
    int32_t input_offset, output_offset;
};

struct PixelShufflePush {
    int32_t render_width, render_height, display_width, display_height;
    int32_t scale, input_offset, output_offset;
};

class PrismVulkan {
public:
    PrismVulkan() = default;
    ~PrismVulkan();

    // Standalone mode: creates own Vulkan instance/device
    bool Init(const PrismVulkanConfig& config);

    // External device mode: uses provided Vulkan device (for OptiScaler interop)
    // The caller must ensure the device supports required extensions
    bool InitWithDevice(const PrismVulkanConfig& config,
                        VkInstance instance, VkPhysicalDevice physical,
                        VkDevice device, VkQueue queue, uint32_t queueFamily);

    bool LoadWeights(const char* weights_path);
    void RecordCommandBuffer();

    // Run one frame with CPU staging — returns GPU time in milliseconds
    float Infer(const void* input_fp16, void* output_fp16);

    // Run inference reading/writing directly from/to GPU buffers (zero CPU copy)
    // input_buf must contain 6*render_w*render_h fp16 values
    // output_buf will receive 3*display_w*display_h fp16 values
    // Both buffers must be on the SAME device as this engine
    float InferGPU(VkBuffer input_buf, VkDeviceSize input_offset,
                   VkBuffer output_buf, VkDeviceSize output_offset);

    // Run benchmark (N frames, returns avg ms)
    float Benchmark(int loops = 100, int warmup = 20);

    void Shutdown();

    int GetRenderW() const { return cfg_.render_w; }
    int GetRenderH() const { return cfg_.render_h; }
    int GetDisplayW() const { return cfg_.render_w * cfg_.scale; }
    int GetDisplayH() const { return cfg_.render_h * cfg_.scale; }
    int GetScale() const { return cfg_.scale; }

    VkDevice GetDevice() const { return device_; }
    VkQueue GetQueue() const { return queue_; }
    VkPhysicalDevice GetPhysicalDevice() const { return physical_; }
    VkInstance GetInstance() const { return instance_; }
    VkFence GetFence() const { return fence_; }
    bool IsInitialized() const { return initialized_; }

private:
    PrismVulkanConfig cfg_;
    bool owns_device_ = false;  // true if we created the Vulkan instance/device
    bool initialized_ = false;

    // Vulkan objects
    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queue_family_ = 0;
    VkCommandPool cmd_pool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_buf_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_buf_gpu_ = VK_NULL_HANDLE;  // separate cmd buf for GPU-to-GPU path
    VkFence fence_ = VK_NULL_HANDLE;
    VkQueryPool query_pool_ = VK_NULL_HANDLE;
    float timestamp_period_ = 0;

    // Pipelines
    VkPipelineLayout pipeline_layout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool desc_pool_ = VK_NULL_HANDLE;
    VkDescriptorSet desc_set_ = VK_NULL_HANDLE;

    VkPipeline pipe_conv3x3_ = VK_NULL_HANDLE;
    VkPipeline pipe_dw_conv_ = VK_NULL_HANDLE;
    VkPipeline pipe_pw_conv_ = VK_NULL_HANDLE;
    VkPipeline pipe_pw_conv_coopvec_ = VK_NULL_HANDLE;  // cooperative vector (tensor core)
    VkPipeline pipe_fused_dw_pw_ = VK_NULL_HANDLE;     // fused DW3x3+PW1x1 (tensor core)
    VkPipeline pipe_dw_conv_hwc_ = VK_NULL_HANDLE;
    VkPipeline pipe_chw_to_hwc_ = VK_NULL_HANDLE;
    VkPipeline pipe_hwc_to_chw_ = VK_NULL_HANDLE;
    VkPipeline pipe_pixelshuffle_ = VK_NULL_HANDLE;

    // Buffers
    VkBuffer weight_buf_ = VK_NULL_HANDLE;
    VkDeviceMemory weight_mem_ = VK_NULL_HANDLE;
    VkBuffer feature_buf_ = VK_NULL_HANDLE;
    VkDeviceMemory feature_mem_ = VK_NULL_HANDLE;
    VkBuffer input_staging_ = VK_NULL_HANDLE;
    VkDeviceMemory input_staging_mem_ = VK_NULL_HANDLE;
    VkBuffer output_staging_ = VK_NULL_HANDLE;
    VkDeviceMemory output_staging_mem_ = VK_NULL_HANDLE;

    // Feature buffer layout (3 regions for ping-pong + residual)
    int feat_region_size_ = 0;  // fp16 elements per region
    // Region 0: residual / main
    // Region 1: ping
    // Region 2: pong

    bool InitVulkan();
    bool InitCommon();  // shared init after device is ready (cmd pool, fence, queries)
    bool CreatePipelines();
    bool AllocateBuffers();
    VkPipeline CreateComputePipeline(const char* spv_path, uint32_t push_size);

    uint32_t FindMemoryType(uint32_t type_filter, VkMemoryPropertyFlags props);
    void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags props,
                      VkBuffer& buffer, VkDeviceMemory& memory);
    void CopyToDevice(VkBuffer dst, const void* data, VkDeviceSize size);
    void CopyFromDevice(VkBuffer src, void* data, VkDeviceSize size);

    void AddBarrier(VkCommandBuffer cmd);
};

} // namespace prism
