#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// FUSED MULTI-BLOCK dim=256 Windowed Attention
//
// Processes ALL transformer blocks in a SINGLE dispatch.
// Token state stays in registers between blocks — zero global memory
// round-trips, zero dispatch barriers between blocks.
//
// With windowed attention, each 8x8 window is self-contained, so
// a workgroup can process all blocks without cross-workgroup sync.
//
// This saves per-block:
//   - 1 global memory write (8160 × 256 × 2 = 4MB)
//   - 1 global memory read (same 4MB)
//   - 1 dispatch barrier (~0.05ms)
// For 12 blocks that's 11 round-trips saved = ~8-10MB bandwidth + ~0.5ms barriers
//
// Weight layout: blocks are sequential with fixed stride.
// block b's qkv_w = base_qkv_w + b * weight_stride

layout(local_size_x = 64) in;

layout(push_constant) uniform PC {
    int n_tokens;       // 8160
    int dim;            // 256
    int n_heads;        // 8
    int head_dim;       // 32
    int spatial_w, spatial_h, window_size;  // 120, 68, 8
    int n_blocks;       // 4, 8, or 12
    int weight_stride;  // fp16 elements between consecutive blocks
    // Base weight offsets (for block 0)
    int qkv_w_offset, qkv_b_offset;
    int out_w_offset, out_b_offset;
    int ffn_w1_offset, ffn_b1_offset;
    int ffn_w2_offset, ffn_b2_offset;
    int input_offset, output_offset;
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

shared float16_t shared_k[64 * 32];
shared float16_t shared_v[64 * 32];

void main() {
    int window_idx = int(gl_WorkGroupID.x);
    int local_idx = int(gl_LocalInvocationID.x);

    int windows_x = (spatial_w + window_size - 1) / window_size;
    int win_y = window_idx / windows_x;
    int win_x = window_idx % windows_x;
    int tok_y = local_idx / window_size;
    int tok_x = local_idx % window_size;
    int global_y = win_y * window_size + tok_y;
    int global_x = win_x * window_size + tok_x;
    bool valid = (global_y < spatial_h && global_x < spatial_w);
    int global_token = valid ? (global_y * spatial_w + global_x) : 0;

    // Load input ONCE from global memory
    coopvecNV<float16_t, 256> state;
    if (valid) {
        coopVecLoadNV(state, features, (input_offset + global_token * dim) * 2);
    } else {
        state = coopvecNV<float16_t, 256>(float16_t(0.0));
    }

    int window_tokens = min(64, n_tokens);

    // === PROCESS ALL BLOCKS — state stays in registers ===
    for (int blk = 0; blk < n_blocks; blk++) {
        int boff = blk * weight_stride;

        // QKV Projection
        coopvecNV<float16_t, 256> Q, K_full, V_full;
        coopVecMatMulAddNV(Q, state, gl_ComponentTypeFloat16NV,
            weights, (qkv_w_offset + boff) * 2, gl_ComponentTypeFloat16NV,
            weights, (qkv_b_offset + boff) * 2, gl_ComponentTypeFloat16NV,
            256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        coopVecMatMulAddNV(K_full, state, gl_ComponentTypeFloat16NV,
            weights, (qkv_w_offset + boff + 256*256) * 2, gl_ComponentTypeFloat16NV,
            weights, (qkv_b_offset + boff + 256) * 2, gl_ComponentTypeFloat16NV,
            256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        coopVecMatMulAddNV(V_full, state, gl_ComponentTypeFloat16NV,
            weights, (qkv_w_offset + boff + 2*256*256) * 2, gl_ComponentTypeFloat16NV,
            weights, (qkv_b_offset + boff + 2*256) * 2, gl_ComponentTypeFloat16NV,
            256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);

        // Per-head windowed attention
        coopvecNV<float16_t, 256> attn_out = coopvecNV<float16_t, 256>(float16_t(0.0));

        for (int h = 0; h < 8; h++) {
            int ch = h * 32;

            for (int d = 0; d < 32; d++) {
                shared_k[local_idx * 32 + d] = K_full[ch + d];
                shared_v[local_idx * 32 + d] = V_full[ch + d];
            }
            barrier();

            float q[32];
            for (int d = 0; d < 32; d++) q[d] = float(Q[ch + d]);

            float scores[64];
            float max_s = -1e9;
            for (int j = 0; j < window_tokens; j++) {
                float s = 0.0;
                for (int d = 0; d < 32; d++)
                    s += q[d] * float(shared_k[j * 32 + d]);
                s *= 0.1767766953;
                scores[j] = s;
                if (s > max_s) max_s = s;
            }

            float sum_exp = 0.0;
            float result[32];
            for (int d = 0; d < 32; d++) result[d] = 0.0;
            for (int j = 0; j < window_tokens; j++) {
                float w = exp(scores[j] - max_s);
                sum_exp += w;
                for (int d = 0; d < 32; d++)
                    result[d] += w * float(shared_v[j * 32 + d]);
            }

            float inv = 1.0 / max(sum_exp, 1e-6);
            for (int d = 0; d < 32; d++)
                attn_out[ch + d] = float16_t(result[d] * inv);

            barrier();
        }

        // Output projection + residual 1
        coopvecNV<float16_t, 256> proj_out;
        coopVecMatMulAddNV(proj_out, attn_out, gl_ComponentTypeFloat16NV,
            weights, (out_w_offset + boff) * 2, gl_ComponentTypeFloat16NV,
            weights, (out_b_offset + boff) * 2, gl_ComponentTypeFloat16NV,
            256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        for (int i = 0; i < 256; i++) proj_out[i] = proj_out[i] + state[i];

        // FFN split into 4 chunks of coopvec<256>
        coopvecNV<float16_t, 256> ffn_accum = coopvecNV<float16_t, 256>(float16_t(0.0));

        for (int chunk = 0; chunk < 4; chunk++) {
            coopvecNV<float16_t, 256> h_chunk;
            coopVecMatMulAddNV(h_chunk, proj_out, gl_ComponentTypeFloat16NV,
                weights, (ffn_w1_offset + boff + chunk * 256 * 256) * 2, gl_ComponentTypeFloat16NV,
                weights, (ffn_b1_offset + boff + chunk * 256) * 2, gl_ComponentTypeFloat16NV,
                256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);

            for (int i = 0; i < 256; i++) {
                float x = float(h_chunk[i]);
                h_chunk[i] = float16_t(0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x*x*x))));
            }

            coopvecNV<float16_t, 256> out_chunk;
            coopVecMatMulAddNV(out_chunk, h_chunk, gl_ComponentTypeFloat16NV,
                weights, (ffn_w2_offset + boff + chunk * 256) * 2, gl_ComponentTypeFloat16NV,
                weights, (ffn_b2_offset + boff) * 2, gl_ComponentTypeFloat16NV,
                256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 1024*2);

            for (int i = 0; i < 256; i++)
                ffn_accum[i] = float16_t(float(ffn_accum[i]) + float(out_chunk[i]));
        }

        // Bias correction + residual 2 → update state for next block
        for (int i = 0; i < 256; i++) {
            state[i] = float16_t(float(ffn_accum[i])
                       - 3.0 * float(weights[ffn_b2_offset + boff + i])
                       + float(proj_out[i]));
        }
    }

    // Write final output ONCE to global memory
    if (valid) {
        coopVecStoreNV(state, features, (output_offset + global_token * dim) * 2);
    }
}
