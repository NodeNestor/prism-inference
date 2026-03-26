#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Windowed Self-Attention — 16x16 windows, 256 threads/WG
// Per-head K/V in shared memory to fit 256 tokens.
// 256 tokens per window (vs 64 in 8x8). 40 workgroups (vs 135).
// Better workgroup size for GPU occupancy.

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int n_tokens, dim, n_heads, head_dim;
    int spatial_w, spatial_h, window_size;
    int qkv_w_offset, qkv_b_offset;
    int out_w_offset, out_b_offset;
    int ffn_w1_offset, ffn_b1_offset;
    int ffn_w2_offset, ffn_b2_offset;
    int input_offset, output_offset;
};

layout(set = 0, binding = 0) readonly buffer W { float16_t weights[]; };
layout(set = 0, binding = 1) buffer F { float16_t features[]; };

// Per-head shared memory: K and V for 256 tokens at head_dim=32
shared float16_t shared_k[256 * 32];  // 16 KB
shared float16_t shared_v[256 * 32];  // 16 KB

void main() {
    int window_idx = int(gl_WorkGroupID.x);
    int local_idx = int(gl_LocalInvocationID.x);  // 0..255

    int windows_x = (spatial_w + window_size - 1) / window_size;
    int win_y = window_idx / windows_x;
    int win_x = window_idx % windows_x;
    int tok_y = local_idx / window_size;
    int tok_x = local_idx % window_size;
    int global_y = win_y * window_size + tok_y;
    int global_x = win_x * window_size + tok_x;
    bool valid = (global_y < spatial_h && global_x < spatial_w);
    int global_token = valid ? (global_y * spatial_w + global_x) : 0;

    // Load input and project QKV
    coopvecNV<float16_t, 128> inp;
    if (valid) {
        coopVecLoadNV(inp, features, (input_offset + global_token * dim) * 2);
    } else {
        inp = coopvecNV<float16_t, 128>(float16_t(0.0));
    }

    coopvecNV<float16_t, 128> Q, K_full, V_full;
    coopVecMatMulAddNV(Q, inp, gl_ComponentTypeFloat16NV,
        weights, qkv_w_offset * 2, gl_ComponentTypeFloat16NV,
        weights, qkv_b_offset * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);
    coopVecMatMulAddNV(K_full, inp, gl_ComponentTypeFloat16NV,
        weights, (qkv_w_offset + 128*128) * 2, gl_ComponentTypeFloat16NV,
        weights, (qkv_b_offset + 128) * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);
    coopVecMatMulAddNV(V_full, inp, gl_ComponentTypeFloat16NV,
        weights, (qkv_w_offset + 2*128*128) * 2, gl_ComponentTypeFloat16NV,
        weights, (qkv_b_offset + 2*128) * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

    // Windowed attention: process heads sequentially (shared mem per-head)
    coopvecNV<float16_t, 128> attn_out = coopvecNV<float16_t, 128>(float16_t(0.0));

    int window_tokens = min(256, n_tokens);

    for (int h = 0; h < n_heads; h++) {
        int ch_off = h * head_dim;

        // Store this head's K and V to shared memory
        for (int d = 0; d < head_dim; d++) {
            shared_k[local_idx * head_dim + d] = K_full[ch_off + d];
            shared_v[local_idx * head_dim + d] = V_full[ch_off + d];
        }
        barrier();

        // Compute attention for this head
        float scale = 1.0 / sqrt(float(head_dim));
        float q[32];
        for (int d = 0; d < head_dim; d++) q[d] = float(Q[ch_off + d]);

        float scores[256];
        float max_s = -1e9;
        for (int j = 0; j < window_tokens; j++) {
            float s = 0.0;
            for (int d = 0; d < head_dim; d++)
                s += q[d] * float(shared_k[j * head_dim + d]);
            scores[j] = s * scale;
            if (scores[j] > max_s) max_s = scores[j];
        }

        float sum_exp = 0.0;
        float result[32];
        for (int d = 0; d < head_dim; d++) result[d] = 0.0;

        for (int j = 0; j < window_tokens; j++) {
            float w = exp(scores[j] - max_s);
            sum_exp += w;
            for (int d = 0; d < head_dim; d++)
                result[d] += w * float(shared_v[j * head_dim + d]);
        }

        float inv = 1.0 / max(sum_exp, 1e-6);
        for (int d = 0; d < head_dim; d++)
            attn_out[ch_off + d] = float16_t(result[d] * inv);

        barrier();  // sync before next head overwrites shared memory
    }

    // Output projection
    coopvecNV<float16_t, 128> proj_out;
    coopVecMatMulAddNV(proj_out, attn_out, gl_ComponentTypeFloat16NV,
        weights, out_w_offset * 2, gl_ComponentTypeFloat16NV,
        weights, out_b_offset * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

    for (int i = 0; i < 128; i++) proj_out[i] = proj_out[i] + inp[i];

    // FFN
    coopvecNV<float16_t, 512> ffn_hidden;
    coopVecMatMulAddNV(ffn_hidden, proj_out, gl_ComponentTypeFloat16NV,
        weights, ffn_w1_offset * 2, gl_ComponentTypeFloat16NV,
        weights, ffn_b1_offset * 2, gl_ComponentTypeFloat16NV,
        512, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

    for (int i = 0; i < 512; i++) {
        float x = float(ffn_hidden[i]);
        ffn_hidden[i] = float16_t(0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x*x*x))));
    }

    coopvecNV<float16_t, 128> ffn_out;
    coopVecMatMulAddNV(ffn_out, ffn_hidden, gl_ComponentTypeFloat16NV,
        weights, ffn_w2_offset * 2, gl_ComponentTypeFloat16NV,
        weights, ffn_b2_offset * 2, gl_ComponentTypeFloat16NV,
        128, 512, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 512*2);

    for (int i = 0; i < 128; i++) ffn_out[i] = ffn_out[i] + proj_out[i];

    if (valid) {
        coopVecStoreNV(ffn_out, features, (output_offset + global_token * dim) * 2);
    }
}
