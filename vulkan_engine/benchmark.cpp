// Prism Vulkan compute engine benchmark
// Build: cmake -B build && cmake --build build --config Release
// Run:   benchmark.exe <weights_file> <height> <width> [gpu_id] [loops]

#include "prism_vulkan.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Usage: benchmark <weights_file> <height> <width> [gpu_id] [loops] [channels] [blocks]\n");
        printf("Example: benchmark prism_quality.weights 540 960\n");
        return 1;
    }

    const char* weights_path = argv[1];
    int height = atoi(argv[2]);
    int width = atoi(argv[3]);
    int gpu_id = argc > 4 ? atoi(argv[4]) : 0;
    int loops = argc > 5 ? atoi(argv[5]) : 100;
    int channels = argc > 6 ? atoi(argv[6]) : 64;
    int blocks = argc > 7 ? atoi(argv[7]) : 4;

    printf("=== Prism Vulkan Compute Engine Benchmark ===\n");
    printf("Weights: %s\n", weights_path);
    printf("Render: %dx%d -> Display: %dx%d (2x)\n", width, height, width*2, height*2);
    printf("Model: %dch, %d blocks\n", channels, blocks);
    printf("Loops: %d\n\n", loops);

    prism::PrismVulkanConfig config;
    config.channels = channels;
    config.n_blocks = blocks;
    config.scale = 2;
    config.render_w = width;
    config.render_h = height;
    config.gpu_id = gpu_id;

    prism::PrismVulkan engine;

    if (!engine.Init(config)) {
        printf("FATAL: Failed to init Vulkan\n");
        return 1;
    }

    if (!engine.LoadWeights(weights_path)) {
        printf("FATAL: Failed to load weights\n");
        return 1;
    }

    engine.RecordCommandBuffer();
    engine.Benchmark(loops, 20);

    return 0;
}
