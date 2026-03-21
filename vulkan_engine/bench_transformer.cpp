// Benchmark: Windowed Attention Transformer on tensor cores
// One dispatch per transformer block (QKV + attention + FFN all fused!)

#define VK_USE_PLATFORM_WIN32_KHR
#include "deps/volk.h"
#include <stdio.h>
#include <vector>
#include <fstream>
#include <chrono>

int main(int argc, char** argv) {
    int gpu_id = argc > 1 ? atoi(argv[1]) : 0;
    int n_blocks = argc > 2 ? atoi(argv[2]) : 4;

    printf("=== Prism V2 Windowed Transformer Benchmark ===\n");
    volkInitialize();

    VkApplicationInfo ai = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    VkInstance inst; vkCreateInstance(&ici, nullptr, &inst);
    volkLoadInstance(inst);

    uint32_t gc = 0; vkEnumeratePhysicalDevices(inst, &gc, nullptr);
    std::vector<VkPhysicalDevice> gpus(gc);
    vkEnumeratePhysicalDevices(inst, &gc, gpus.data());
    VkPhysicalDevice phys = gpus[gpu_id];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    printf("GPU: %s\n", props.deviceName);

    // Features chain
    VkPhysicalDeviceShaderFloat16Int8Features f16 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDevice16BitStorageFeatures s16 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    s16.pNext = &f16;
    VkPhysicalDeviceFeatures2 feat2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    feat2.pNext = &s16;
    vkGetPhysicalDeviceFeatures2(phys, &feat2);

    uint32_t qc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qc, nullptr);
    std::vector<VkQueueFamilyProperties> qp(qc);
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qc, qp.data());
    uint32_t qf = 0;
    for (uint32_t i = 0; i < qc; i++) if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = qf; qi.queueCount = 1; qi.pQueuePriorities = &prio;
    const char* exts[] = {"VK_NV_cooperative_vector"};
    VkDeviceCreateInfo di = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    di.pNext = &feat2; di.enabledExtensionCount = 1; di.ppEnabledExtensionNames = exts;
    VkDevice dev; vkCreateDevice(phys, &di, nullptr, &dev);
    volkLoadDevice(dev);
    VkQueue queue; vkGetDeviceQueue(dev, qf, 0, &queue);

    VkCommandPool pool;
    VkCommandPoolCreateInfo cpi = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = qf; cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    vkCreateCommandPool(dev, &cpi, nullptr, &pool);
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    vkAllocateCommandBuffers(dev, &cai, &cmd);
    VkFence fence; VkFenceCreateInfo fi = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    vkCreateFence(dev, &fi, nullptr, &fence);
    VkQueryPool qpool;
    VkQueryPoolCreateInfo qpci = {VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP; qpci.queryCount = 2;
    vkCreateQueryPool(dev, &qpci, nullptr, &qpool);

    // Parameters
    int dim = 128, spatial_w = 120, spatial_h = 68, win = 8;
    int N = spatial_w * spatial_h;  // 8160 tokens
    int hidden = dim * 4;  // 512

    // Weights per block: QKV(3*dim*dim + 3*dim) + OutProj(dim*dim + dim) + FFN(dim*hidden + hidden + hidden*dim + dim)
    int wpb = 3*dim*dim + 3*dim + dim*dim + dim + dim*hidden + hidden + hidden*dim + dim;
    int total_w = wpb * n_blocks;

    // Feature buffer: just 2 * N * dim (ping-pong) + shared mem handles the rest
    int feat_sz = 2 * N * dim;

    printf("Tokens: %d (%dx%d), dim: %d, window: %dx%d\n", N, spatial_w, spatial_h, dim, win, win);
    printf("Blocks: %d, params per block: %d (%.0f KB)\n", n_blocks, wpb, wpb*2.0/1024);
    printf("Total weights: %.1f MB, features: %.1f MB\n", total_w*2.0/1e6, feat_sz*2.0/1e6);

    // Buffers
    auto findMem = [&](uint32_t tf, VkMemoryPropertyFlags p) -> uint32_t {
        VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((tf & (1<<i)) && (mp.memoryTypes[i].propertyFlags & p) == p) return i;
        return 0;
    };
    auto makeBuf = [&](VkDeviceSize sz, VkBufferUsageFlags u, VkBuffer& b, VkDeviceMemory& m) {
        VkBufferCreateInfo bi = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size = sz; bi.usage = u;
        vkCreateBuffer(dev, &bi, nullptr, &b);
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b, &mr);
        VkMemoryAllocateInfo ai2 = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai2.allocationSize = mr.size; ai2.memoryTypeIndex = findMem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        vkAllocateMemory(dev, &ai2, nullptr, &m); vkBindBufferMemory(dev, b, m, 0);
    };

    VkBuffer wBuf, fBuf; VkDeviceMemory wMem, fMem;
    makeBuf(total_w * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, wBuf, wMem);
    makeBuf(feat_sz * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, fBuf, fMem);

    // Descriptors
    VkDescriptorSetLayoutBinding binds[2] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo dli = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = 2; dli.pBindings = binds;
    VkDescriptorSetLayout dl; vkCreateDescriptorSetLayout(dev, &dli, nullptr, &dl);

    VkPushConstantRange pcr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    VkPipelineLayoutCreateInfo pli = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &dl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; vkCreatePipelineLayout(dev, &pli, nullptr, &pl);

    VkDescriptorPoolSize dps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &dps;
    VkDescriptorPool dp; vkCreateDescriptorPool(dev, &dpci, nullptr, &dp);
    VkDescriptorSetAllocateInfo dsai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dp; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &dl;
    VkDescriptorSet ds; vkAllocateDescriptorSets(dev, &dsai, &ds);

    VkDescriptorBufferInfo wbi = {wBuf, 0, VK_WHOLE_SIZE}, fbi = {fBuf, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet wr[2] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 0,0,1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &wbi, nullptr},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, ds, 1,0,1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &fbi, nullptr}};
    vkUpdateDescriptorSets(dev, 2, wr, 0, nullptr);

    // Load pipeline
    std::ifstream sf("shaders/attention_windowed.spv", std::ios::binary|std::ios::ate);
    size_t sz = sf.tellg(); sf.seekg(0);
    std::vector<uint32_t> code(sz/4); sf.read((char*)code.data(), sz);
    VkShaderModuleCreateInfo smi = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = sz; smi.pCode = code.data();
    VkShaderModule sm; vkCreateShaderModule(dev, &smi, nullptr, &sm);
    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpci.stage.module = sm; cpci.stage.pName = "main";
    cpci.layout = pl;
    VkPipeline pipe; vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe);
    vkDestroyShaderModule(dev, sm, nullptr);
    printf("Pipeline created\n\n");

    // Record: n_blocks dispatches, each is one full transformer block
    int windows_x = (spatial_w + win - 1) / win;
    int windows_y = (spatial_h + win - 1) / win;
    int n_windows = windows_x * windows_y;
    printf("Windows: %d (%dx%d)\n", n_windows, windows_x, windows_y);

    auto barrier = [&]() {
        VkMemoryBarrier b = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &b, 0, nullptr, 0, nullptr);
    };

    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    vkBeginCommandBuffer(cmd, &bi);
    vkCmdResetQueryPool(cmd, qpool, 0, 2);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qpool, 0);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);

    int w_off = 0;
    int in_off = 0, out_off = N * dim;

    for (int blk = 0; blk < n_blocks; blk++) {
        struct {
            int n_tokens, dim, n_heads, head_dim, spatial_w, spatial_h, window_size;
            int qkv_w, qkv_b, out_w, out_b, ffn_w1, ffn_b1, ffn_w2, ffn_b2;
            int input_offset, output_offset;
        } push;
        push.n_tokens = N; push.dim = dim; push.n_heads = 4; push.head_dim = 32;
        push.spatial_w = spatial_w; push.spatial_h = spatial_h; push.window_size = win;
        push.qkv_w = w_off;
        push.qkv_b = w_off + 3*dim*dim;
        push.out_w = w_off + 3*dim*dim + 3*dim;
        push.out_b = w_off + 3*dim*dim + 3*dim + dim*dim;
        int ffn_base = w_off + 3*dim*dim + 3*dim + dim*dim + dim;
        push.ffn_w1 = ffn_base;
        push.ffn_b1 = ffn_base + dim*hidden;
        push.ffn_w2 = ffn_base + dim*hidden + hidden;
        push.ffn_b2 = ffn_base + dim*hidden + hidden + hidden*dim;
        push.input_offset = in_off;
        push.output_offset = out_off;
        vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(cmd, (uint32_t)n_windows, 1, 1);
        barrier();

        w_off += wpb;
        int tmp = in_off; in_off = out_off; out_off = tmp;
    }

    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qpool, 1);
    vkEndCommandBuffer(cmd);

    // Benchmark
    printf("\nWarmup...\n");
    for (int i = 0; i < 20; i++) {
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &si, fence);
        vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX); vkResetFences(dev, 1, &fence);
    }

    int loops = 200;
    printf("Benchmarking %d iterations...\n", loops);
    double total = 0, mn = 1e9, mx = 0;
    for (int i = 0; i < loops; i++) {
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &si, fence);
        vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX); vkResetFences(dev, 1, &fence);
        uint64_t ts[2];
        vkGetQueryPoolResults(dev, qpool, 0, 2, sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        double ms = (ts[1]-ts[0]) * props.limits.timestampPeriod / 1e6;
        total += ms; if (ms < mn) mn = ms; if (ms > mx) mx = ms;
    }

    printf("\n=== RESULTS: %d transformer blocks, %d tokens, dim=%d, window=%dx%d ===\n",
           n_blocks, N, dim, win, win);
    printf("  GPU avg: %.2f ms (%.0f FPS)\n", total/loops, 1000.0*loops/total);
    printf("  GPU min: %.2f ms\n", mn);
    printf("  GPU max: %.2f ms\n", mx);
    printf("\nFull pipeline (add encoder ~0.3ms + decoder ~0.7ms + pixelshuffle ~0.1ms):\n");
    printf("  Estimated total: ~%.1f ms\n", total/loops + 1.1);

    vkDeviceWaitIdle(dev);
    return 0;
}
