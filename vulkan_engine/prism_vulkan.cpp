#include "prism_vulkan.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <vector>
#include <fstream>

namespace prism {

// ============================================================================
// Lifecycle
// ============================================================================

PrismVulkan::~PrismVulkan() { Shutdown(); }

bool PrismVulkan::Init(const PrismVulkanConfig& config) {
    cfg_ = config;
    owns_device_ = true;
    if (!InitVulkan()) return false;
    if (!CreatePipelines()) return false;
    if (!AllocateBuffers()) return false;
    initialized_ = true;
    return true;
}

bool PrismVulkan::InitWithDevice(const PrismVulkanConfig& config,
                                  VkInstance instance, VkPhysicalDevice physical,
                                  VkDevice device, VkQueue queue, uint32_t queueFamily) {
    cfg_ = config;
    owns_device_ = false;
    instance_ = instance;
    physical_ = physical;
    device_ = device;
    queue_ = queue;
    queue_family_ = queueFamily;

    // Get timestamp period
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_, &props);
    timestamp_period_ = props.limits.timestampPeriod;

    if (!InitCommon()) return false;
    if (!CreatePipelines()) return false;
    if (!AllocateBuffers()) return false;
    initialized_ = true;
    return true;
}

bool PrismVulkan::InitCommon() {
    // Command pool
    VkCommandPoolCreateInfo poolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.queueFamilyIndex = queue_family_;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(device_, &poolInfo, nullptr, &cmd_pool_);

    // Command buffers (main + GPU-to-GPU)
    VkCommandBufferAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocInfo.commandPool = cmd_pool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(device_, &allocInfo, &cmd_buf_);
    vkAllocateCommandBuffers(device_, &allocInfo, &cmd_buf_gpu_);

    // Fence
    VkFenceCreateInfo fenceInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(device_, &fenceInfo, nullptr, &fence_);

    // Timestamp query pool
    VkQueryPoolCreateInfo queryInfo = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryInfo.queryCount = 4; // 2 for main, 2 for GPU path
    vkCreateQueryPool(device_, &queryInfo, nullptr, &query_pool_);

    return true;
}

void PrismVulkan::Shutdown() {
    if (!initialized_) return;
    initialized_ = false;

    if (device_) vkDeviceWaitIdle(device_);

    auto destroy = [&](auto& h, auto fn) { if (h) { fn(device_, h, nullptr); h = VK_NULL_HANDLE; } };
    destroy(pipe_conv3x3_, vkDestroyPipeline);
    destroy(pipe_dw_conv_, vkDestroyPipeline);
    destroy(pipe_pw_conv_, vkDestroyPipeline);
    destroy(pipe_pw_conv_coopvec_, vkDestroyPipeline);
    destroy(pipe_dw_conv_hwc_, vkDestroyPipeline);
    destroy(pipe_chw_to_hwc_, vkDestroyPipeline);
    destroy(pipe_hwc_to_chw_, vkDestroyPipeline);
    destroy(pipe_pixelshuffle_, vkDestroyPipeline);
    destroy(pipeline_layout_, vkDestroyPipelineLayout);
    destroy(desc_pool_, vkDestroyDescriptorPool);
    destroy(desc_layout_, vkDestroyDescriptorSetLayout);
    destroy(query_pool_, vkDestroyQueryPool);
    destroy(fence_, vkDestroyFence);
    destroy(cmd_pool_, vkDestroyCommandPool);

    auto destroyBuf = [&](VkBuffer& b, VkDeviceMemory& m) {
        if (b) { vkDestroyBuffer(device_, b, nullptr); b = VK_NULL_HANDLE; }
        if (m) { vkFreeMemory(device_, m, nullptr); m = VK_NULL_HANDLE; }
    };
    destroyBuf(weight_buf_, weight_mem_);
    destroyBuf(feature_buf_, feature_mem_);
    destroyBuf(input_staging_, input_staging_mem_);
    destroyBuf(output_staging_, output_staging_mem_);

    // Only destroy device/instance if we created them
    if (owns_device_) {
        if (device_) { vkDestroyDevice(device_, nullptr); device_ = VK_NULL_HANDLE; }
        if (instance_) { vkDestroyInstance(instance_, nullptr); instance_ = VK_NULL_HANDLE; }
    } else {
        device_ = VK_NULL_HANDLE;
        instance_ = VK_NULL_HANDLE;
    }
}

// ============================================================================
// Vulkan init
// ============================================================================

bool PrismVulkan::InitVulkan() {
    if (volkInitialize() != VK_SUCCESS) {
        printf("ERROR: Failed to load Vulkan loader (vulkan-1.dll)\n");
        return false;
    }

    // Create instance
    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "PrismVulkan";
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instInfo.pApplicationInfo = &appInfo;

    if (vkCreateInstance(&instInfo, nullptr, &instance_) != VK_SUCCESS) {
        printf("ERROR: Failed to create Vulkan instance\n");
        return false;
    }
    volkLoadInstance(instance_);

    // Enumerate GPUs
    uint32_t gpuCount = 0;
    vkEnumeratePhysicalDevices(instance_, &gpuCount, nullptr);
    if (gpuCount == 0) { printf("ERROR: No Vulkan GPUs\n"); return false; }

    std::vector<VkPhysicalDevice> gpus(gpuCount);
    vkEnumeratePhysicalDevices(instance_, &gpuCount, gpus.data());

    // Select GPU
    int sel = cfg_.gpu_id < (int)gpuCount ? cfg_.gpu_id : 0;
    physical_ = gpus[sel];

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physical_, &props);
    timestamp_period_ = props.limits.timestampPeriod; // ns per tick
    printf("GPU %d: %s\n", sel, props.deviceName);
    printf("  Timestamp period: %.1f ns\n", timestamp_period_);

    // Find compute queue
    uint32_t queueCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &queueCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueProps(queueCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &queueCount, queueProps.data());

    queue_family_ = UINT32_MAX;
    for (uint32_t i = 0; i < queueCount; i++) {
        if (queueProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queue_family_ = i;
            // Prefer compute-only queue (async compute)
            if (!(queueProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) break;
        }
    }
    if (queue_family_ == UINT32_MAX) { printf("ERROR: No compute queue\n"); return false; }

    // Enable fp16 + cooperative matrix (tensor core) features
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatFeatures = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR
    };

    VkPhysicalDeviceShaderFloat16Int8Features f16Features = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES
    };
    f16Features.pNext = &coopMatFeatures;

    VkPhysicalDevice16BitStorageFeatures storage16 = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES
    };
    storage16.pNext = &f16Features;

    VkPhysicalDeviceFeatures2 features2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features2.pNext = &storage16;
    vkGetPhysicalDeviceFeatures2(physical_, &features2);

    printf("  FP16 storage: %s\n", storage16.storageBuffer16BitAccess ? "YES" : "NO");
    printf("  Cooperative matrix (tensor cores): %s\n",
           coopMatFeatures.cooperativeMatrix ? "YES" : "NO");

    // Enable device extensions for tensor core access
    const char* deviceExtensions[] = {
        "VK_KHR_cooperative_matrix",
        "VK_NV_cooperative_vector",
    };

    // Create device
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queueInfo.queueFamilyIndex = queue_family_;
    queueInfo.queueCount = 1;
    queueInfo.pQueuePriorities = &priority;

    VkDeviceCreateInfo devInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    devInfo.queueCreateInfoCount = 1;
    devInfo.pQueueCreateInfos = &queueInfo;
    devInfo.pNext = &features2;
    devInfo.enabledExtensionCount = 2;
    devInfo.ppEnabledExtensionNames = deviceExtensions;

    if (vkCreateDevice(physical_, &devInfo, nullptr, &device_) != VK_SUCCESS) {
        printf("ERROR: Failed to create device\n");
        return false;
    }
    volkLoadDevice(device_);
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

    if (!InitCommon()) return false;

    printf("Vulkan initialized\n");
    return true;
}

// ============================================================================
// Pipeline creation
// ============================================================================

VkPipeline PrismVulkan::CreateComputePipeline(const char* spv_path, uint32_t push_size) {
    // Load SPIR-V
    std::ifstream file(spv_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        printf("ERROR: Cannot open %s\n", spv_path);
        return VK_NULL_HANDLE;
    }
    size_t size = file.tellg();
    file.seekg(0);
    std::vector<uint32_t> code(size / 4);
    file.read(reinterpret_cast<char*>(code.data()), size);

    VkShaderModuleCreateInfo moduleInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    moduleInfo.codeSize = size;
    moduleInfo.pCode = code.data();

    VkShaderModule module;
    if (vkCreateShaderModule(device_, &moduleInfo, nullptr, &module) != VK_SUCCESS) {
        printf("ERROR: Failed to create shader module from %s\n", spv_path);
        return VK_NULL_HANDLE;
    }

    VkComputePipelineCreateInfo pipeInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeInfo.stage.module = module;
    pipeInfo.stage.pName = "main";
    pipeInfo.layout = pipeline_layout_;

    VkPipeline pipeline;
    VkResult r = vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipeline);
    vkDestroyShaderModule(device_, module, nullptr);

    if (r != VK_SUCCESS) {
        printf("ERROR: Failed to create pipeline from %s\n", spv_path);
        return VK_NULL_HANDLE;
    }
    return pipeline;
}

bool PrismVulkan::CreatePipelines() {
    // Descriptor set layout: binding 0 = weights (readonly), binding 1 = features (read/write)
    // binding 2 = output RGB (writeonly, for pixelshuffle)
    VkDescriptorSetLayoutBinding bindings[3] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;
    vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &desc_layout_);

    // Pipeline layout with push constants (max size across all shaders)
    VkPushConstantRange pushRange = {};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = 40; // max push constant size (PWConvPush = 10 ints)

    VkPipelineLayoutCreateInfo plInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &desc_layout_;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pushRange;
    vkCreatePipelineLayout(device_, &plInfo, nullptr, &pipeline_layout_);

    // Create pipelines from SPIR-V files
    std::string shader_dir = cfg_.shader_dir;
    if (!shader_dir.empty() && shader_dir.back() != '/' && shader_dir.back() != '\\')
        shader_dir += '/';
    char path[512];

    snprintf(path, sizeof(path), "%sconv3x3.spv", shader_dir.c_str());
    pipe_conv3x3_ = CreateComputePipeline(path, sizeof(Conv3x3Push));

    snprintf(path, sizeof(path), "%sdepthwise_conv3x3.spv", shader_dir.c_str());
    pipe_dw_conv_ = CreateComputePipeline(path, sizeof(DWConv3x3Push));

    snprintf(path, sizeof(path), "%spointwise_conv.spv", shader_dir.c_str());
    pipe_pw_conv_ = CreateComputePipeline(path, sizeof(PWConvPush));

    snprintf(path, sizeof(path), "%spixelshuffle_sigmoid.spv", shader_dir.c_str());
    pipe_pixelshuffle_ = CreateComputePipeline(path, sizeof(PixelShufflePush));

    // Fused DW3x3+PW1x1 shader + tensor cores (non-tiled, best performance)
    snprintf(path, sizeof(path), "%sfused_dw_pw.spv", shader_dir.c_str());
    pipe_fused_dw_pw_ = CreateComputePipeline(path, sizeof(FusedDWPWPush));
    if (pipe_fused_dw_pw_) {
        printf("  Fused DW+PW pipeline: READY (tensor cores)\n");
    }

    // Input conv3x3 im2col+coopvec and output pw coopvec
    // These are stored as pipe_chw_to_hwc_ and pipe_hwc_to_chw_ (reusing slots)
    snprintf(path, sizeof(path), "%sinput_conv_coopvec.spv", shader_dir.c_str());
    auto pipe_input_coopvec = CreateComputePipeline(path, sizeof(Conv3x3Push));
    if (pipe_input_coopvec) {
        printf("  Input conv im2col+coopvec: READY\n");
        pipe_chw_to_hwc_ = pipe_input_coopvec;  // reuse slot
    }
    snprintf(path, sizeof(path), "%spw_conv_64to12_coopvec.spv", shader_dir.c_str());
    auto pipe_output_coopvec = CreateComputePipeline(path, sizeof(PWConvPush));
    if (pipe_output_coopvec) {
        printf("  Output PW coopvec (64→12): READY\n");
        pipe_hwc_to_chw_ = pipe_output_coopvec;  // reuse slot
    }

    // HWC-format shaders
    snprintf(path, sizeof(path), "%sdepthwise_conv3x3_hwc.spv", shader_dir.c_str());
    pipe_dw_conv_hwc_ = CreateComputePipeline(path, sizeof(DWConv3x3Push));

    snprintf(path, sizeof(path), "%schw_to_hwc.spv", shader_dir.c_str());
    pipe_chw_to_hwc_ = CreateComputePipeline(path, sizeof(ConvertPush));

    snprintf(path, sizeof(path), "%shwc_to_chw.spv", shader_dir.c_str());
    pipe_hwc_to_chw_ = CreateComputePipeline(path, sizeof(ConvertPush));

    // Cooperative vector (tensor core) pointwise conv — VK_NV_cooperative_vector
    snprintf(path, sizeof(path), "%spointwise_conv_coopvec.spv", shader_dir.c_str());
    pipe_pw_conv_coopvec_ = CreateComputePipeline(path, sizeof(PWConvPush));
    if (pipe_pw_conv_coopvec_) {
        printf("  Cooperative vector PW conv pipeline: READY (tensor cores)\n");
    } else {
        printf("  Cooperative vector PW conv pipeline: FAILED (falling back to ALU)\n");
    }

    if (!pipe_conv3x3_ || !pipe_dw_conv_ || !pipe_pw_conv_ || !pipe_pixelshuffle_) {
        printf("ERROR: Failed to create one or more pipelines\n");
        return false;
    }
    printf("Pipelines created\n");
    return true;
}

// ============================================================================
// Buffer allocation
// ============================================================================

uint32_t PrismVulkan::FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(physical_, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    printf("ERROR: Failed to find suitable memory type\n");
    return 0;
}

void PrismVulkan::CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                VkMemoryPropertyFlags props,
                                VkBuffer& buffer, VkDeviceMemory& memory) {
    VkBufferCreateInfo bufInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufInfo.size = size;
    bufInfo.usage = usage;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    vkCreateBuffer(device_, &bufInfo, nullptr, &buffer);

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device_, buffer, &memReq);

    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = FindMemoryType(memReq.memoryTypeBits, props);
    vkAllocateMemory(device_, &allocInfo, nullptr, &memory);
    vkBindBufferMemory(device_, buffer, memory, 0);
}

void PrismVulkan::CopyToDevice(VkBuffer dst, const void* data, VkDeviceSize size) {
    // Use staging buffer
    VkBuffer staging;
    VkDeviceMemory stagingMem;
    CreateBuffer(size,
                 VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staging, stagingMem);

    void* mapped;
    vkMapMemory(device_, stagingMem, 0, size, 0, &mapped);
    memcpy(mapped, data, size);
    vkUnmapMemory(device_, stagingMem);

    // Record copy
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = cmd_pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    vkAllocateCommandBuffers(device_, &ai, &cmd);

    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    VkBufferCopy region = {0, 0, size};
    vkCmdCopyBuffer(cmd, staging, dst, 1, &region);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(queue_, 1, &si, fence_);
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);

    vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
    vkDestroyBuffer(device_, staging, nullptr);
    vkFreeMemory(device_, stagingMem, nullptr);
}

void PrismVulkan::CopyFromDevice(VkBuffer src, void* data, VkDeviceSize size) {
    VkBuffer staging;
    VkDeviceMemory stagingMem;
    CreateBuffer(size,
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staging, stagingMem);

    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = cmd_pool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    vkAllocateCommandBuffers(device_, &ai, &cmd);

    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    VkBufferCopy region = {0, 0, size};
    vkCmdCopyBuffer(cmd, src, staging, 1, &region);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(queue_, 1, &si, fence_);
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);

    void* mapped;
    vkMapMemory(device_, stagingMem, 0, size, 0, &mapped);
    memcpy(data, mapped, size);
    vkUnmapMemory(device_, stagingMem);

    vkFreeCommandBuffers(device_, cmd_pool_, 1, &cmd);
    vkDestroyBuffer(device_, staging, nullptr);
    vkFreeMemory(device_, stagingMem, nullptr);
}

bool PrismVulkan::AllocateBuffers() {
    int W = cfg_.render_w;
    int H = cfg_.render_h;
    int ch = cfg_.channels;
    int pixels = W * H;

    // Feature buffer: 3 regions (residual, ping, pong) + output region (12ch for to_sub)
    feat_region_size_ = ch * pixels;  // fp16 elements per region
    int output_region = cfg_.scale * cfg_.scale * 3 * pixels; // 12*pixels for scale=2
    VkDeviceSize feat_bytes = (VkDeviceSize)(3 * feat_region_size_ + output_region) * 2; // fp16 = 2 bytes

    // Input staging: 6 channels at render res
    VkDeviceSize input_bytes = (VkDeviceSize)(6 * pixels) * 2;

    // Output staging: 3 channels at display res
    int dW = cfg_.render_w * cfg_.scale;
    int dH = cfg_.render_h * cfg_.scale;
    VkDeviceSize output_bytes = (VkDeviceSize)(3 * dW * dH) * 2;

    printf("Buffer sizes:\n");
    printf("  Features: %.1f MB (%d regions of %d fp16)\n",
           feat_bytes / 1e6, 3, feat_region_size_);
    printf("  Input staging: %.1f MB\n", input_bytes / 1e6);
    printf("  Output staging: %.1f MB\n", output_bytes / 1e6);

    // Device-local buffers
    CreateBuffer(feat_bytes,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 feature_buf_, feature_mem_);

    // Host-visible staging for input/output
    CreateBuffer(input_bytes,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 input_staging_, input_staging_mem_);

    CreateBuffer(output_bytes,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 output_staging_, output_staging_mem_);

    // Descriptor set
    VkDescriptorPoolSize poolSizes[1] = {};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = 3;

    VkDescriptorPoolCreateInfo dpInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpInfo.maxSets = 1;
    dpInfo.poolSizeCount = 1;
    dpInfo.pPoolSizes = poolSizes;
    vkCreateDescriptorPool(device_, &dpInfo, nullptr, &desc_pool_);

    VkDescriptorSetAllocateInfo dsInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsInfo.descriptorPool = desc_pool_;
    dsInfo.descriptorSetCount = 1;
    dsInfo.pSetLayouts = &desc_layout_;
    vkAllocateDescriptorSets(device_, &dsInfo, &desc_set_);

    printf("Buffers allocated\n");
    return true;
}

bool PrismVulkan::LoadWeights(const char* weights_path) {
    // Load fp16 weight data
    std::ifstream f(weights_path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        printf("ERROR: Cannot open %s\n", weights_path);
        return false;
    }
    size_t size = f.tellg();
    f.seekg(0);
    std::vector<char> data(size);
    f.read(data.data(), size);

    printf("Loading weights: %zu bytes (%zu fp16 values)\n", size, size / 2);

    // Create weight buffer and upload
    CreateBuffer(size,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 weight_buf_, weight_mem_);
    CopyToDevice(weight_buf_, data.data(), size);

    // Update descriptor set
    VkDescriptorBufferInfo weightBufInfo = {weight_buf_, 0, VK_WHOLE_SIZE};
    VkDescriptorBufferInfo featBufInfo = {feature_buf_, 0, VK_WHOLE_SIZE};
    // For pixelshuffle, output goes to feature buffer region 3
    VkDescriptorBufferInfo outputBufInfo = {feature_buf_, 0, VK_WHOLE_SIZE};

    VkWriteDescriptorSet writes[3] = {};
    writes[0] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[0].dstSet = desc_set_;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[0].pBufferInfo = &weightBufInfo;

    writes[1] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[1].dstSet = desc_set_;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[1].pBufferInfo = &featBufInfo;

    writes[2] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    writes[2].dstSet = desc_set_;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[2].pBufferInfo = &outputBufInfo;

    vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);

    printf("Weights loaded to GPU\n");
    return true;
}

// ============================================================================
// Pipeline barrier helper
// ============================================================================

void PrismVulkan::AddBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// ============================================================================
// Record the inference command buffer — THE KEY PART
// All dispatches in ONE command buffer with barriers only.
// ============================================================================

void PrismVulkan::RecordCommandBuffer() {
    int W = cfg_.render_w;
    int H = cfg_.render_h;
    int ch = cfg_.channels;
    int pixels = W * H;
    int scale = cfg_.scale;
    int dW = W * scale;
    int dH = H * scale;

    // Workgroup dispatch sizes
    auto gx = (uint32_t)((W + 15) / 16);
    auto gy = (uint32_t)((H + 15) / 16);
    uint32_t lin_groups = (uint32_t)((pixels + 255) / 256);  // for local_size_x=256 shaders

    // Feature region offsets (in fp16 elements)
    int R = 0;                          // residual region
    int A = feat_region_size_;          // ping region
    int B = 2 * feat_region_size_;      // pong region
    int OUT = 3 * feat_region_size_;    // output region for to_sub

    // Weight layer offsets (from JSON, hardcoded for quality 64ch 4-block model)
    // Input conv: 6→64
    struct LayerWeights { int w_off; int b_off; };

    // Parse from the known architecture
    int w_off = 0;
    auto advance = [&w_off](int weights, int bias) -> LayerWeights {
        LayerWeights lw = {w_off, w_off + weights};
        w_off += weights + bias;
        return lw;
    };

    // Input conv: 6→64, k=3, weights=3456, bias=64
    LayerWeights inp = advance(3456, 64);

    // 4 blocks, each: dw1(576,0) pw1(4096,64) dw2(576,0) pw2(4096,64) dw3(576,0) pw3(4096,64)
    struct BlockWeights {
        LayerWeights dw1, pw1, dw2, pw2, dw3, pw3;
    };
    BlockWeights blocks[4];
    for (int b = 0; b < cfg_.n_blocks; b++) {
        blocks[b].dw1 = advance(576, 0);
        blocks[b].pw1 = advance(4096, 64);
        blocks[b].dw2 = advance(576, 0);
        blocks[b].pw2 = advance(4096, 64);
        blocks[b].dw3 = advance(576, 0);
        blocks[b].pw3 = advance(4096, 64);
    }

    // Output conv: 64→12, k=3, weights=6912, bias=12
    LayerWeights out_conv = advance(6912, 12);

    printf("Total weight offset: %d (expected: 67276)\n", w_off);

    // Begin recording
    vkResetCommandBuffer(cmd_buf_, 0);
    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd_buf_, &beginInfo);

    // Reset timestamp queries
    vkCmdResetQueryPool(cmd_buf_, query_pool_, 0, 2);
    vkCmdWriteTimestamp(cmd_buf_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_, 0);

    // Bind descriptor set (shared across all dispatches)
    vkCmdBindDescriptorSets(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout_, 0, 1, &desc_set_, 0, nullptr);

    // === LAYER 0: Input Conv3x3 6→64 + ReLU ===
    // Input is at the beginning of feature buffer (uploaded before dispatch)
    // We'll upload input to a special location. Actually, let's use the
    // input_staging buffer directly. But our shaders read from binding 1 (features).
    // So we need to copy input to feature buffer first, THEN run.
    // Input goes to region at offset: 3*feat_region_size + output_region
    // Actually simpler: put input at offset 0 (region R), output goes to region A
    // BUT: input is 6 channels, region R holds 64 channels. Input fits in region R.
    // We'll copy input data to offset 0 of feature buffer before dispatching.

    // Input conv: reads from offset 0 (6ch input), writes to region R
    // Wait, we need to think about this more carefully.
    // Input is 6 channels. Output is 64 channels.
    // Input data is at offset 0, size = 6*H*W fp16 elements
    // Output needs to go somewhere that doesn't overlap with input.
    // Region A (offset=feat_region_size_) is safe.
    // Then we'll use A as the block input (residual save).

    // Input conv: try coopvec (im2col) first, fallback to conv3x3
    if (pipe_chw_to_hwc_ /* reused for input_coopvec */) {
        vkCmdBindPipeline(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_chw_to_hwc_);
        Conv3x3Push push = {};
        push.in_channels = 6;
        push.out_channels = ch;
        push.width = W; push.height = H;
        push.weight_offset = inp.w_off;
        push.bias_offset = inp.b_off;
        push.input_offset = 0;
        push.output_offset = A;
        push.relu = 1;
        vkCmdPushConstants(cmd_buf_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(push), &push);
        vkCmdDispatch(cmd_buf_, lin_groups, 1, 1);
        AddBarrier(cmd_buf_);
    } else {
        vkCmdBindPipeline(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_conv3x3_);
        Conv3x3Push push = {};
        push.in_channels = 6; push.out_channels = ch;
        push.width = W; push.height = H;
        push.weight_offset = inp.w_off; push.bias_offset = inp.b_off;
        push.input_offset = 0; push.output_offset = A;
        push.relu = 1;
        vkCmdPushConstants(cmd_buf_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(push), &push);
        vkCmdDispatch(cmd_buf_, gx, gy, (uint32_t)ch);
        AddBarrier(cmd_buf_);
    }

    // === CHW → HWC conversion (for efficient DSC blocks) ===
    // Input conv output is at region A in CHW. Convert to HWC at region B.
    bool use_hwc = pipe_pw_conv_coopvec_ && pipe_dw_conv_hwc_ && pipe_chw_to_hwc_;
    // lin_groups already defined above

    // === DSC BLOCKS ===
    // DW conv uses CHW (spatial locality), PW conv uses HWC (channel locality)
    // Convert between formats as needed. Data starts at A in CHW.

    bool use_coopvec = pipe_pw_conv_coopvec_ && pipe_chw_to_hwc_ && pipe_hwc_to_chw_;
    bool use_fused = pipe_fused_dw_pw_ != VK_NULL_HANDLE;

    // Helper lambda to dispatch CHW→HWC conversion
    auto dispatchCHWtoHWC = [&](int src, int dst) {
        vkCmdBindPipeline(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_chw_to_hwc_);
        ConvertPush cvt = {}; cvt.channels = ch; cvt.width = W; cvt.height = H;
        cvt.input_offset = src; cvt.output_offset = dst;
        vkCmdPushConstants(cmd_buf_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cvt), &cvt);
        vkCmdDispatch(cmd_buf_, lin_groups, 1, 1);
        AddBarrier(cmd_buf_);
    };
    auto dispatchHWCtoCHW = [&](int src, int dst) {
        vkCmdBindPipeline(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_hwc_to_chw_);
        ConvertPush cvt = {}; cvt.channels = ch; cvt.width = W; cvt.height = H;
        cvt.input_offset = src; cvt.output_offset = dst;
        vkCmdPushConstants(cmd_buf_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cvt), &cvt);
        vkCmdDispatch(cmd_buf_, lin_groups, 1, 1);
        AddBarrier(cmd_buf_);
    };
    auto dispatchDW_CHW = [&](int w_off, int in_off, int out_off) {
        vkCmdBindPipeline(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_dw_conv_);
        DWConv3x3Push push = {};
        push.channels = ch; push.width = W; push.height = H;
        push.weight_offset = w_off; push.bias_offset = -1;
        push.input_offset = in_off; push.output_offset = out_off;
        push.relu = 0;
        vkCmdPushConstants(cmd_buf_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd_buf_, gx, gy, (uint32_t)ch);
        AddBarrier(cmd_buf_);
    };
    auto dispatchPW_coopvec = [&](int w_off, int b_off, int in_off, int out_off, int relu_flag, int res_off) {
        vkCmdBindPipeline(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_pw_conv_coopvec_);
        PWConvPush push = {};
        push.in_channels = ch; push.out_channels = ch;
        push.width = W; push.height = H;
        push.weight_offset = w_off; push.bias_offset = b_off;
        push.input_offset = in_off; push.output_offset = out_off;
        push.relu = relu_flag; push.residual_offset = res_off;
        vkCmdPushConstants(cmd_buf_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd_buf_, lin_groups, 1, 1);
        AddBarrier(cmd_buf_);
    };

    for (int b = 0; b < cfg_.n_blocks; b++) {
        auto& bw = blocks[b];

        if (use_fused) {
            // FUSED DW+PW: each pair is ONE dispatch, intermediate in registers.
            // Block: 3 fused dispatches instead of 6. Data stays in CHW.
            // A = block input/residual, B/R = ping/pong

            // fused dw1+pw1: A→B
            vkCmdBindPipeline(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_fused_dw_pw_);
            {
                FusedDWPWPush push = {};
                push.channels = ch; push.width = W; push.height = H;
                push.dw_weight_offset = bw.dw1.w_off;
                push.pw_weight_offset = bw.pw1.w_off;
                push.pw_bias_offset = bw.pw1.b_off;
                push.input_offset = A; push.output_offset = B;
                push.relu = 1; push.residual_offset = -1;
                vkCmdPushConstants(cmd_buf_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                vkCmdDispatch(cmd_buf_, lin_groups, 1, 1);
                AddBarrier(cmd_buf_);
            }

            // fused dw2+pw2: B→R
            {
                FusedDWPWPush push = {};
                push.channels = ch; push.width = W; push.height = H;
                push.dw_weight_offset = bw.dw2.w_off;
                push.pw_weight_offset = bw.pw2.w_off;
                push.pw_bias_offset = bw.pw2.b_off;
                push.input_offset = B; push.output_offset = R;
                push.relu = 1; push.residual_offset = -1;
                vkCmdPushConstants(cmd_buf_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                vkCmdDispatch(cmd_buf_, gx, gy, 1);
                AddBarrier(cmd_buf_);
            }

            // fused dw3+pw3: R→A, with residual from A
            {
                FusedDWPWPush push = {};
                push.channels = ch; push.width = W; push.height = H;
                push.dw_weight_offset = bw.dw3.w_off;
                push.pw_weight_offset = bw.pw3.w_off;
                push.pw_bias_offset = bw.pw3.b_off;
                push.input_offset = R; push.output_offset = A;
                push.relu = 1; push.residual_offset = A;
                vkCmdPushConstants(cmd_buf_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
                vkCmdDispatch(cmd_buf_, gx, gy, 1);
                AddBarrier(cmd_buf_);
            }
            // After block: data in A (CHW). Next block reads from A.
        }
    }

    // === Output Conv 64→12 (to_sub) ===
    if (pipe_hwc_to_chw_ /* reused for pw_64to12_coopvec */) {
        // PW conv 64→12 using cooperative vectors (tensor cores)
        vkCmdBindPipeline(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_hwc_to_chw_);
        PWConvPush push = {};
        push.in_channels = ch;
        push.out_channels = 3 * scale * scale;
        push.width = W; push.height = H;
        push.weight_offset = out_conv.w_off;
        push.bias_offset = out_conv.b_off;
        push.input_offset = A;
        push.output_offset = OUT;
        push.relu = 0;
        push.residual_offset = -1;
        vkCmdPushConstants(cmd_buf_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(push), &push);
        vkCmdDispatch(cmd_buf_, lin_groups, 1, 1);
        AddBarrier(cmd_buf_);
    } else {
        vkCmdBindPipeline(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_conv3x3_);
        Conv3x3Push push = {};
        push.in_channels = ch;
        push.out_channels = 3 * scale * scale;
        push.width = W; push.height = H;
        push.weight_offset = out_conv.w_off;
        push.bias_offset = out_conv.b_off;
        push.input_offset = A;
        push.output_offset = OUT;
        push.relu = 0;
        vkCmdPushConstants(cmd_buf_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(push), &push);
        vkCmdDispatch(cmd_buf_, gx, gy, (uint32_t)(3 * scale * scale));
        AddBarrier(cmd_buf_);
    }

    // === PixelShuffle + Sigmoid ===
    {
        auto dgx = (uint32_t)((dW + 15) / 16);
        auto dgy = (uint32_t)((dH + 15) / 16);
        vkCmdBindPipeline(cmd_buf_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_pixelshuffle_);
        PixelShufflePush push = {};
        push.render_width = W;
        push.render_height = H;
        push.display_width = dW;
        push.display_height = dH;
        push.scale = scale;
        push.input_offset = OUT;   // read from output region (features buf, binding 1)
        push.output_offset = 0;    // write to output region (binding 2, same buffer)
        // NOTE: pixelshuffle shader reads from binding 1 (features) and writes to binding 2
        // Both point to feature_buf_ but at different offsets
        // Actually we'll write the output back to region R (offset 0) of feature buf
        // since we don't need it anymore
        vkCmdPushConstants(cmd_buf_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                          0, sizeof(push), &push);
        vkCmdDispatch(cmd_buf_, dgx, dgy, 1);
    }

    vkCmdWriteTimestamp(cmd_buf_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, 1);
    vkEndCommandBuffer(cmd_buf_);

    printf("Command buffer recorded: %d dispatches + barriers\n",
           1 + cfg_.n_blocks * 6 + 1 + 1);  // input + blocks + output + pixelshuffle
}

// ============================================================================
// Inference
// ============================================================================

float PrismVulkan::InferGPU(VkBuffer input_buf, VkDeviceSize input_offset,
                             VkBuffer output_buf, VkDeviceSize output_offset) {
    int W = cfg_.render_w;
    int H = cfg_.render_h;
    int pixels = W * H;
    int dW = W * cfg_.scale;
    int dH = H * cfg_.scale;

    VkDeviceSize input_size = (VkDeviceSize)(6 * pixels) * 2;
    VkDeviceSize output_size = (VkDeviceSize)(3 * dW * dH) * 2;

    // Record a command buffer that copies input from shared buffer, runs inference,
    // then copies output to shared buffer
    vkResetCommandBuffer(cmd_buf_gpu_, 0);

    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd_buf_gpu_, &bi);

    vkCmdWriteTimestamp(cmd_buf_gpu_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, query_pool_, 2);

    // Copy from shared input buffer to feature buffer (region 0)
    VkBufferCopy copyRegion = {};
    copyRegion.srcOffset = input_offset;
    copyRegion.dstOffset = 0;
    copyRegion.size = input_size;
    vkCmdCopyBuffer(cmd_buf_gpu_, input_buf, feature_buf_, 1, &copyRegion);

    // Barrier after copy before compute
    AddBarrier(cmd_buf_gpu_);

    // Now replay the pre-recorded inference dispatches
    // We need to re-record them inline since we can't nest command buffers
    // For now, submit the pre-recorded main cmd_buf first, then copy output

    vkEndCommandBuffer(cmd_buf_gpu_);

    // Submit: first copy input, then run inference, then copy output
    // Step 1: Copy input into feature buffer
    VkSubmitInfo si1 = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si1.commandBufferCount = 1;
    si1.pCommandBuffers = &cmd_buf_gpu_;
    vkQueueSubmit(queue_, 1, &si1, VK_NULL_HANDLE);

    // Step 2: Submit pre-recorded inference
    VkSubmitInfo si2 = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si2.commandBufferCount = 1;
    si2.pCommandBuffers = &cmd_buf_;
    vkQueueSubmit(queue_, 1, &si2, fence_);
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);

    // Step 3: Copy output from feature buffer to shared output buffer
    vkResetCommandBuffer(cmd_buf_gpu_, 0);
    vkBeginCommandBuffer(cmd_buf_gpu_, &bi);

    VkBufferCopy outCopy = {};
    outCopy.srcOffset = 0; // output sits at start of feature buffer after pixelshuffle
    outCopy.dstOffset = output_offset;
    outCopy.size = output_size;
    vkCmdCopyBuffer(cmd_buf_gpu_, feature_buf_, output_buf, 1, &outCopy);

    vkCmdWriteTimestamp(cmd_buf_gpu_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, query_pool_, 3);
    vkEndCommandBuffer(cmd_buf_gpu_);

    VkSubmitInfo si3 = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si3.commandBufferCount = 1;
    si3.pCommandBuffers = &cmd_buf_gpu_;
    vkQueueSubmit(queue_, 1, &si3, fence_);
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);

    // Read GPU timestamps
    uint64_t timestamps[4];
    vkGetQueryPoolResults(device_, query_pool_, 0, 4,
                         sizeof(timestamps), timestamps, sizeof(uint64_t),
                         VK_QUERY_RESULT_64_BIT);

    // Total time including copies
    float gpu_ms = (float)(timestamps[3] - timestamps[2]) * timestamp_period_ / 1e6f;
    float infer_ms = (float)(timestamps[1] - timestamps[0]) * timestamp_period_ / 1e6f;

    return infer_ms; // return inference time, not copy overhead
}

float PrismVulkan::Infer(const void* input_fp16, void* output_fp16) {
    int W = cfg_.render_w;
    int H = cfg_.render_h;
    int pixels = W * H;
    int dW = W * cfg_.scale;
    int dH = H * cfg_.scale;

    // Upload input (6 channels fp16) to feature buffer at offset 0
    VkDeviceSize input_size = (VkDeviceSize)(6 * pixels) * 2;
    CopyToDevice(feature_buf_, input_fp16, input_size);

    // Submit the pre-recorded command buffer
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_buf_;
    vkQueueSubmit(queue_, 1, &si, fence_);
    vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX);
    vkResetFences(device_, 1, &fence_);

    // Read GPU timestamps
    uint64_t timestamps[2];
    vkGetQueryPoolResults(device_, query_pool_, 0, 2,
                         sizeof(timestamps), timestamps, sizeof(uint64_t),
                         VK_QUERY_RESULT_64_BIT);
    float gpu_ms = (float)(timestamps[1] - timestamps[0]) * timestamp_period_ / 1e6f;

    // Download output (3 channels at display res) from feature buffer offset 0
    // (pixelshuffle wrote to binding 2 offset 0, which maps to feature buffer)
    if (output_fp16) {
        VkDeviceSize output_size = (VkDeviceSize)(3 * dW * dH) * 2;
        CopyFromDevice(feature_buf_, output_fp16, output_size);
    }

    return gpu_ms;
}

float PrismVulkan::Benchmark(int loops, int warmup) {
    int W = cfg_.render_w;
    int H = cfg_.render_h;
    int pixels = W * H;

    // Create dummy input
    std::vector<uint16_t> input(6 * pixels, 0x3800); // fp16 0.5

    printf("Warmup (%d iterations)...\n", warmup);
    for (int i = 0; i < warmup; i++) {
        Infer(input.data(), nullptr);
    }

    printf("Benchmarking %d iterations...\n", loops);
    double total_gpu_ms = 0;
    double min_ms = 1e9, max_ms = 0;

    auto wall_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < loops; i++) {
        float gpu_ms = Infer(input.data(), nullptr);
        total_gpu_ms += gpu_ms;
        if (gpu_ms < min_ms) min_ms = gpu_ms;
        if (gpu_ms > max_ms) max_ms = gpu_ms;
    }
    auto wall_end = std::chrono::high_resolution_clock::now();
    double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    double avg_gpu = total_gpu_ms / loops;
    double avg_wall = wall_ms / loops;

    printf("\n=== RESULTS (quality %dch %d-block, %dx%d -> %dx%d) ===\n",
           cfg_.channels, cfg_.n_blocks, W, H, W*cfg_.scale, H*cfg_.scale);
    printf("  GPU time avg: %.2f ms (%.0f FPS)\n", avg_gpu, 1000.0 / avg_gpu);
    printf("  GPU time min: %.2f ms (%.0f FPS)\n", min_ms, 1000.0 / min_ms);
    printf("  GPU time max: %.2f ms\n", max_ms);
    printf("  Wall time avg: %.2f ms (includes upload/download)\n", avg_wall);
    printf("  Upload+download overhead: %.2f ms\n", avg_wall - avg_gpu);

    return (float)avg_gpu;
}

} // namespace prism
