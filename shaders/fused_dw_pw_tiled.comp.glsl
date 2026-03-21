#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// FUSED DW3x3 + PW1x1 with SHARED MEMORY TILING
//
// Each workgroup processes a 16×16 pixel tile.
// Step 1: Cooperatively load 18×18 border tile × 64 channels into shared memory
// Step 2: Each thread reads DW 3×3 from shared memory (L1 speed, no global reads!)
// Step 3: Feed DW result to coopVecMatMulAdd on tensor cores
// Step 4: Write output to global memory
//
// This eliminates ~80% of global memory reads vs the non-tiled version.

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PC {
    int channels;
    int width;
    int height;
    int dw_weight_offset;
    int pw_weight_offset;
    int pw_bias_offset;
    int input_offset;
    int output_offset;
    int relu;
    int residual_offset;
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

// Shared memory tile: 18×18 pixels × 64 channels = 20,736 fp16 = 40.5KB
// Split into groups of 4 channels to reduce shared memory if needed
shared float16_t tile[64][18 * 18];  // [channel][spatial]

void main() {
    int lx = int(gl_LocalInvocationID.x);  // 0..15
    int ly = int(gl_LocalInvocationID.y);  // 0..15
    int local_idx = ly * 16 + lx;          // 0..255

    // Tile origin in global coordinates
    int tile_x = int(gl_WorkGroupID.x) * 16;
    int tile_y = int(gl_WorkGroupID.y) * 16;

    // Global pixel for this thread
    int gx = tile_x + lx;
    int gy = tile_y + ly;
    int pixels = width * height;

    // === STEP 1: Cooperative load of 18×18 tile for all 64 channels ===
    // Total elements: 64 × 324 = 20,736
    // 256 threads, each loads ~81 elements
    int tile_elements = 18 * 18;  // 324

    for (int ch = 0; ch < 64; ch++) {
        // Each thread loads some pixels of this channel's tile
        for (int i = local_idx; i < tile_elements; i += 256) {
            int ty = i / 18;  // local tile y (0..17)
            int tx = i % 18;  // local tile x (0..17)

            // Global coordinates (with -1 offset for border)
            int global_y = tile_y + ty - 1;
            int global_x = tile_x + tx - 1;

            float16_t val = float16_t(0.0);
            if (global_y >= 0 && global_y < height && global_x >= 0 && global_x < width) {
                val = features[input_offset + ch * pixels + global_y * width + global_x];
            }
            tile[ch][i] = val;
        }
    }
    barrier();

    if (gx >= width || gy >= height) return;

    // === STEP 2: DW Conv 3×3 from shared memory → registers ===
    // This pixel is at tile position (lx+1, ly+1) in the 18×18 tile
    coopvecNV<float16_t, 64> dw_out;

    for (int ch = 0; ch < 64; ch++) {
        float sum = 0.0;
        int w_base = dw_weight_offset + ch * 9;

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                // Read from shared memory — L1 speed!
                int tile_pos = (ly + 1 + ky) * 18 + (lx + 1 + kx);
                float val = float(tile[ch][tile_pos]);
                sum += val * float(weights[w_base + (ky+1) * 3 + (kx+1)]);
            }
        }
        dw_out[ch] = float16_t(sum);
    }

    // === STEP 3: PW Conv 1×1 using tensor cores ===
    coopvecNV<float16_t, 64> pw_out;
    coopVecMatMulAddNV(
        pw_out, dw_out,
        gl_ComponentTypeFloat16NV,
        weights, pw_weight_offset * 2,
        gl_ComponentTypeFloat16NV,
        weights, pw_bias_offset * 2,
        gl_ComponentTypeFloat16NV,
        64, 64,
        gl_CooperativeVectorMatrixLayoutRowMajorNV,
        false, 64 * 2
    );

    // === STEP 4: Residual + ReLU + write output (CHW) ===
    int pixel_idx = gy * width + gx;
    for (int ch = 0; ch < 64; ch++) {
        float v = float(pw_out[ch]);
        if (residual_offset >= 0) {
            v += float(features[residual_offset + ch * pixels + pixel_idx]);
        }
        if (relu == 1 && v < 0.0) v = 0.0;
        features[output_offset + ch * pixels + pixel_idx] = float16_t(v);
    }
}
