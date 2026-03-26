#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// OPTIMIZED dim=256 Windowed Self-Attention
//
// Key design choices to avoid the coopvec<512> bottleneck:
//   - All projections use coopvec<256> (QKV, output proj)
//   - FFN 256→1024→256 split into 4x coopvec<256> chunks
//   - Per-head shared memory (8KB) to fit 256-dim tokens in 48KB
//   - 8 heads × 32 head_dim, processed sequentially
//   - Maximum coopvec size: 256 (avoids register pressure cliff)
//
// Architecture: dim=256, 8 heads, head_dim=32, FFN hidden=1024
// 8x8 windows, 64 threads/WG, 135 workgroups

layout(local_size_x = 64) in;

layout(push_constant) uniform PC {
    int n_tokens;       // 8160
    int dim;            // 256
    int n_heads;        // 8
    int head_dim;       // 32
    int spatial_w, spatial_h, window_size;
    int qkv_w_offset, qkv_b_offset;    // [3*256, 256]
    int out_w_offset, out_b_offset;      // [256, 256]
    int ffn_w1_offset, ffn_b1_offset;    // [1024, 256]
    int ffn_w2_offset, ffn_b2_offset;    // [256, 1024]
    int input_offset, output_offset;
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

// Per-head shared memory: 64 tokens × 32 head_dim × 2 (K+V) = 8KB
shared float16_t shared_k[64 * 32];  // 4 KB
shared float16_t shared_v[64 * 32];  // 4 KB

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

    // === Load input ===
    coopvecNV<float16_t, 256> inp;
    if (valid) {
        coopVecLoadNV(inp, features, (input_offset + global_token * dim) * 2);
    } else {
        inp = coopvecNV<float16_t, 256>(float16_t(0.0));
    }

    // === QKV Projection (3x coopvec<256> matmuls) ===
    coopvecNV<float16_t, 256> Q, K_full, V_full;
    coopVecMatMulAddNV(Q, inp, gl_ComponentTypeFloat16NV,
        weights, qkv_w_offset * 2, gl_ComponentTypeFloat16NV,
        weights, qkv_b_offset * 2, gl_ComponentTypeFloat16NV,
        256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
    coopVecMatMulAddNV(K_full, inp, gl_ComponentTypeFloat16NV,
        weights, (qkv_w_offset + 256*256) * 2, gl_ComponentTypeFloat16NV,
        weights, (qkv_b_offset + 256) * 2, gl_ComponentTypeFloat16NV,
        256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
    coopVecMatMulAddNV(V_full, inp, gl_ComponentTypeFloat16NV,
        weights, (qkv_w_offset + 2*256*256) * 2, gl_ComponentTypeFloat16NV,
        weights, (qkv_b_offset + 2*256) * 2, gl_ComponentTypeFloat16NV,
        256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);

    // === Per-head windowed attention ===
    coopvecNV<float16_t, 256> attn_out = coopvecNV<float16_t, 256>(float16_t(0.0));
    int window_tokens = min(64, n_tokens);

    for (int h = 0; h < 8; h++) {
        int ch_off = h * 32;

        // Store this head's K,V slice to shared memory
        for (int d = 0; d < 32; d++) {
            shared_k[local_idx * 32 + d] = K_full[ch_off + d];
            shared_v[local_idx * 32 + d] = V_full[ch_off + d];
        }
        barrier();

        // Extract Q for this head
        float q[32];
        for (int d = 0; d < 32; d++) q[d] = float(Q[ch_off + d]);

        // Attention scores
        float scores[64];
        float max_s = -1e9;
        for (int j = 0; j < window_tokens; j++) {
            float s = 0.0;
            for (int d = 0; d < 32; d++)
                s += q[d] * float(shared_k[j * 32 + d]);
            s *= 0.1767766953;  // 1/sqrt(32)
            scores[j] = s;
            if (s > max_s) max_s = s;
        }

        // Softmax + weighted sum
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
            attn_out[ch_off + d] = float16_t(result[d] * inv);

        barrier();  // before next head overwrites shared
    }

    // K_full, V_full no longer needed — compiler can free registers

    // === Output Projection ===
    coopvecNV<float16_t, 256> proj_out;
    coopVecMatMulAddNV(proj_out, attn_out, gl_ComponentTypeFloat16NV,
        weights, out_w_offset * 2, gl_ComponentTypeFloat16NV,
        weights, out_b_offset * 2, gl_ComponentTypeFloat16NV,
        256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);

    // Residual 1
    for (int i = 0; i < 256; i++) proj_out[i] = proj_out[i] + inp[i];

    // Q, attn_out no longer needed

    // === FFN: 256→1024→256, split into 4 chunks of coopvec<256> ===
    // Process each chunk: W1_chunk → GELU → W2_chunk → accumulate
    // This avoids ever creating coopvec<512> or larger

    coopvecNV<float16_t, 256> ffn_accum = coopvecNV<float16_t, 256>(float16_t(0.0));

    for (int chunk = 0; chunk < 4; chunk++) {
        // W1 chunk: 256→256 (rows chunk*256..(chunk+1)*256-1 of W1[1024,256])
        coopvecNV<float16_t, 256> h_chunk;
        coopVecMatMulAddNV(h_chunk, proj_out, gl_ComponentTypeFloat16NV,
            weights, (ffn_w1_offset + chunk * 256 * 256) * 2, gl_ComponentTypeFloat16NV,
            weights, (ffn_b1_offset + chunk * 256) * 2, gl_ComponentTypeFloat16NV,
            256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);

        // GELU
        for (int i = 0; i < 256; i++) {
            float x = float(h_chunk[i]);
            h_chunk[i] = float16_t(0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x*x*x))));
        }

        // W2 chunk: 256→256 (columns chunk*256..(chunk+1)*256-1 of W2[256,1024])
        coopvecNV<float16_t, 256> out_chunk;
        coopVecMatMulAddNV(out_chunk, h_chunk, gl_ComponentTypeFloat16NV,
            weights, (ffn_w2_offset + chunk * 256) * 2, gl_ComponentTypeFloat16NV,
            weights, ffn_b2_offset * 2, gl_ComponentTypeFloat16NV,
            256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 1024*2);

        // Accumulate
        for (int i = 0; i < 256; i++)
            ffn_accum[i] = float16_t(float(ffn_accum[i]) + float(out_chunk[i]));
    }

    // Correct for bias added 4x (want 1x) + residual 2
    for (int i = 0; i < 256; i++) {
        float val = float(ffn_accum[i])
                  - 3.0 * float(weights[ffn_b2_offset + i])
                  + float(proj_out[i]);
        ffn_accum[i] = float16_t(val);
    }

    if (valid) {
        coopVecStoreNV(ffn_accum, features, (output_offset + global_token * dim) * 2);
    }
}
