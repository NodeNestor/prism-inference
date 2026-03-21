// Prism inference benchmark — C++ ncnn Vulkan, zero Python
// Build: cl /EHsc /O2 benchmark.cpp /I../ncnn/include /link ../ncnn/lib/ncnn.lib
// Run:   benchmark.exe model.ncnn.param model.ncnn.bin 540 960

#include <stdio.h>
#include <chrono>
#include <string>

#include "net.h"
#include "gpu.h"

int main(int argc, char** argv)
{
    if (argc < 5) {
        printf("Usage: benchmark <param> <bin> <height> <width> [gpu_id] [loops]\n");
        return 1;
    }

    const char* param_path = argv[1];
    const char* bin_path = argv[2];
    int height = atoi(argv[3]);
    int width = atoi(argv[4]);
    int gpu_id = argc > 5 ? atoi(argv[5]) : 0;
    int loops = argc > 6 ? atoi(argv[6]) : 100;

    printf("=== Prism Inference Benchmark (ncnn C++ Vulkan) ===\n");
    printf("Model: %s\n", param_path);
    printf("Input: %dx%d\n", width, height);
    printf("GPU: %d\n", gpu_id);
    printf("Loops: %d\n\n", loops);

    // Init Vulkan
    ncnn::create_gpu_instance();
    printf("GPUs found: %d\n", ncnn::get_gpu_count());
    for (int i = 0; i < ncnn::get_gpu_count(); i++) {
        const ncnn::GpuInfo& info = ncnn::get_gpu_info(i);
        printf("  GPU %d: %s\n", i, info.device_name());
    }

    // Load model
    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_arithmetic = true;
    net.opt.num_threads = 1;
    net.set_vulkan_device(gpu_id);

    int ret = net.load_param(param_path);
    if (ret != 0) { printf("Failed to load param\n"); return 1; }
    ret = net.load_model(bin_path);
    if (ret != 0) { printf("Failed to load model\n"); return 1; }
    printf("Model loaded\n\n");

    // Pre-acquire Vulkan allocators (reuse across frames — critical for speed)
    ncnn::VkAllocator* blob_alloc = net.vulkan_device()->acquire_blob_allocator();
    ncnn::VkAllocator* staging_alloc = net.vulkan_device()->acquire_staging_allocator();

    // Create input mats
    ncnn::Mat color(width, height, 6);  // 6ch: color(3)+depth(1)+mv(2)
    color.fill(0.5f);

    // Warmup
    printf("Warmup (20 iterations)...\n");
    for (int i = 0; i < 20; i++) {
        ncnn::Extractor ex = net.create_extractor();
        ex.set_blob_vkallocator(blob_alloc);
        ex.set_workspace_vkallocator(blob_alloc);
        ex.set_staging_vkallocator(staging_alloc);
        ex.input("input", color);
        ncnn::Mat out;
        ex.extract("output", out);
    }

    // Benchmark
    printf("Benchmarking %d iterations...\n", loops);
    double total_ms = 0;
    double min_ms = 1e9;
    double max_ms = 0;

    for (int i = 0; i < loops; i++) {
        ncnn::Extractor ex = net.create_extractor();
        ex.set_blob_vkallocator(blob_alloc);
        ex.set_workspace_vkallocator(blob_alloc);
        ex.set_staging_vkallocator(staging_alloc);
        ex.input("input", color);

        auto t0 = std::chrono::high_resolution_clock::now();
        ncnn::Mat out;
        ex.extract("output", out);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;

        if (i == 0) {
            printf("Output: %dx%dx%d\n\n", out.w, out.h, out.c);
        }
    }

    double avg_ms = total_ms / loops;
    printf("=== RESULTS ===\n");
    printf("  avg: %.2f ms (%.0f FPS)\n", avg_ms, 1000.0 / avg_ms);
    printf("  min: %.2f ms (%.0f FPS)\n", min_ms, 1000.0 / min_ms);
    printf("  max: %.2f ms\n", max_ms);

    // Cleanup
    net.vulkan_device()->reclaim_blob_allocator(blob_alloc);
    net.vulkan_device()->reclaim_staging_allocator(staging_alloc);
    ncnn::destroy_gpu_instance();

    return 0;
}
