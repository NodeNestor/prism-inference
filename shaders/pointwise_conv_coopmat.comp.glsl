#version 450
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_memory_scope_semantics : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require

// 1x1 Pointwise Convolution using TENSOR CORES (cooperative matrix)
//
// This is a GEMM: Output[M,N] = Weight[M,K] × Input[K,N]
//   M = out_channels (64)
//   K = in_channels (64)
//   N = pixels_per_tile (16)
//
// Using 16×16×16 cooperative matrix multiply (runs on tensor cores).
// Each subgroup (warp of 32 threads) processes one 16×16×16 tile.
// We tile M, K, N to cover the full computation.
//
// Memory layout:
//   Weights: [out_ch × in_ch] fp16 (row-major, out_ch rows of in_ch)
//   Features: [ch/4][H*W][4] pack4 format — NOT USED HERE
//   Features: [ch][H*W] CHW format — each channel is contiguous pixels
//
// For cooperative matrix load from CHW buffer:
//   Input matrix [K=in_ch, N=16 pixels]: stride = H*W (channels are H*W apart)
//   This is COLUMN-MAJOR from the input's perspective since each "column"
//   (channel) is stored as a contiguous run in HW dimension.

// One subgroup per tile of 16 output pixels
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    int in_channels;   // K dimension (64)
    int out_channels;  // M dimension (64)
    int width;
    int height;
    int weight_offset; // fp16 offset into weight buffer
    int bias_offset;   // fp16 offset for bias
    int input_offset;  // fp16 offset into feature buffer
    int output_offset; // fp16 offset into feature buffer
    int relu;
    int residual_offset; // -1 = no residual
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

// Shared memory — flattened 1D arrays for coopMatLoad compatibility
shared float16_t input_tile[64 * 16];   // [K][N] = [in_ch][16 pixels], stride=16
shared float16_t result_tile[16 * 16];  // [M_tile][N] = [16 out_ch][16 pixels], stride=16

void main() {
    int pixels = width * height;
    int subgroup_id = int(gl_WorkGroupID.x);  // which group of 16 pixels
    int pixel_base = subgroup_id * 16;  // first pixel in this tile
    int lane = int(gl_SubgroupInvocationID);

    // Load input tile into shared memory: [in_ch][16 pixels] row-major, stride=16
    for (int idx = int(gl_LocalInvocationID.x); idx < in_channels * 16; idx += 32) {
        int ic = idx / 16;
        int p = idx % 16;
        int global_pixel = pixel_base + p;
        float16_t val = float16_t(0.0);
        if (global_pixel < pixels) {
            val = features[input_offset + ic * pixels + global_pixel];
        }
        input_tile[ic * 16 + p] = val;
    }
    barrier();

    // For each M-tile (16 output channels at a time):
    for (int m_tile = 0; m_tile < out_channels; m_tile += 16) {
        // Initialize accumulator to zero
        coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator> acc =
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseAccumulator>(float16_t(0.0));

        // Accumulate over K-tiles (16 input channels at a time):
        for (int k_tile = 0; k_tile < in_channels; k_tile += 16) {
            // Load weight sub-matrix [16×16] from weight buffer
            // Weight layout: [out_ch × in_ch], row-major
            // We want weight[m_tile..m_tile+15][k_tile..k_tile+15]
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseA> matW;
            coopMatLoad(matW, weights, weight_offset + m_tile * in_channels + k_tile,
                       in_channels, gl_CooperativeMatrixLayoutRowMajor);

            // Load input sub-matrix [16×16] from shared memory
            // input_tile[k_tile..k_tile+15][0..15]
            coopmat<float16_t, gl_ScopeSubgroup, 16, 16, gl_MatrixUseB> matI;
            coopMatLoad(matI, input_tile, k_tile * 16, 16,
                       gl_CooperativeMatrixLayoutRowMajor);

            // Multiply-accumulate on tensor cores!
            acc = coopMatMulAdd(matW, matI, acc);
        }

        // Store result: acc is [16 out_ch × 16 pixels]
        // First store to shared memory, then write out with bias/relu/residual
        coopMatStore(acc, result_tile, 0, 16, gl_CooperativeMatrixLayoutRowMajor);
        barrier();

        // Each thread writes some of the 16×16 = 256 output values
        for (int idx = int(gl_LocalInvocationID.x); idx < 256; idx += 32) {
            int oc_local = idx / 16;
            int p = idx % 16;
            int oc = m_tile + oc_local;
            int global_pixel = pixel_base + p;

            if (oc < out_channels && global_pixel < pixels) {
                float val = float(result_tile[oc_local * 16 + p]);

                // Add bias
                val += float(weights[bias_offset + oc]);

                // Add residual
                if (residual_offset >= 0) {
                    val += float(features[residual_offset + oc * pixels + global_pixel]);
                }

                // ReLU
                if (relu == 1 && val < 0.0) val = 0.0;

                features[output_offset + oc * pixels + global_pixel] = float16_t(val);
            }
        }
        barrier();
    }
}
