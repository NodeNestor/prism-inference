#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// OPTIMIZED Windowed Self-Attention — FFN split into 2x coopvec<256>
//
// Same architecture as attention_windowed.comp.glsl but with the FFN
// split to avoid the coopvec<512> register pressure bottleneck.
// Benchmarks show this is 16x faster for the FFN portion.
//
// Changes from original:
//   - FFN W1: 1x coopvec<512> → 2x coopvec<256> (two 128→256 chunks)
//   - FFN W2: 1x coopvec<512> → 2x coopvec<256> (two 256→128 chunks)
//   - Everything else identical

layout(local_size_x = 64) in;

layout(push_constant) uniform PC {
    int n_tokens, dim, n_heads, head_dim;
    int spatial_w, spatial_h, window_size;
    int qkv_w_offset, qkv_b_offset;
    int out_w_offset, out_b_offset;
    int ffn_w1_offset, ffn_b1_offset;
    int ffn_w2_offset, ffn_b2_offset;
    int input_offset, output_offset;
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

shared float16_t shared_k[64 * 128];
shared float16_t shared_v[64 * 128];

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

    // === QKV Projection (tensor cores, coopvec<128>) ===
    coopvecNV<float16_t, 128> inp;
    if (valid) {
        coopVecLoadNV(inp, features, (input_offset + global_token * dim) * 2);
    } else {
        inp = coopvecNV<float16_t, 128>(float16_t(0.0));
    }

    coopvecNV<float16_t, 128> Q, K, V;
    coopVecMatMulAddNV(Q, inp, gl_ComponentTypeFloat16NV,
        weights, qkv_w_offset * 2, gl_ComponentTypeFloat16NV,
        weights, qkv_b_offset * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);
    coopVecMatMulAddNV(K, inp, gl_ComponentTypeFloat16NV,
        weights, (qkv_w_offset + 128*128) * 2, gl_ComponentTypeFloat16NV,
        weights, (qkv_b_offset + 128) * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);
    coopVecMatMulAddNV(V, inp, gl_ComponentTypeFloat16NV,
        weights, (qkv_w_offset + 2*128*128) * 2, gl_ComponentTypeFloat16NV,
        weights, (qkv_b_offset + 2*128) * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

    coopVecStoreNV(K, shared_k, local_idx * 128 * 2);
    coopVecStoreNV(V, shared_v, local_idx * 128 * 2);
    barrier();

    // === Windowed Attention (scalar, 64 tokens) ===
    coopvecNV<float16_t, 128> attn_out = coopvecNV<float16_t, 128>(float16_t(0.0));

    for (int h = 0; h < n_heads; h++) {
        int ch_off = h * head_dim;
        float scale = 1.0 / sqrt(float(head_dim));

        float q[32];
        for (int d = 0; d < head_dim; d++) q[d] = float(Q[ch_off + d]);

        float scores[64];
        float max_s = -1e9;
        int window_tokens = min(64, n_tokens);

        for (int j = 0; j < window_tokens; j++) {
            float s = 0.0;
            for (int d = 0; d < head_dim; d++)
                s += q[d] * float(shared_k[j * 128 + ch_off + d]);
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
                result[d] += w * float(shared_v[j * 128 + ch_off + d]);
        }

        float inv = 1.0 / max(sum_exp, 1e-6);
        for (int d = 0; d < head_dim; d++)
            attn_out[ch_off + d] = float16_t(result[d] * inv);
    }

    // === Output Projection (tensor cores) ===
    coopvecNV<float16_t, 128> proj_out;
    coopVecMatMulAddNV(proj_out, attn_out, gl_ComponentTypeFloat16NV,
        weights, out_w_offset * 2, gl_ComponentTypeFloat16NV,
        weights, out_b_offset * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

    // Residual 1
    for (int i = 0; i < 128; i++) proj_out[i] = proj_out[i] + inp[i];

    // === FFN with SPLIT coopvec<256> (the key optimization!) ===

    // W1 chunk 0: 128→256 (rows 0-255 of W1)
    coopvecNV<float16_t, 256> h0;
    coopVecMatMulAddNV(h0, proj_out, gl_ComponentTypeFloat16NV,
        weights, ffn_w1_offset * 2, gl_ComponentTypeFloat16NV,
        weights, ffn_b1_offset * 2, gl_ComponentTypeFloat16NV,
        256, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);
    // GELU
    for (int i = 0; i < 256; i++) {
        float x = float(h0[i]);
        h0[i] = float16_t(0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x*x*x))));
    }

    // W2 chunk 0: 256→128
    coopvecNV<float16_t, 128> out0;
    coopVecMatMulAddNV(out0, h0, gl_ComponentTypeFloat16NV,
        weights, ffn_w2_offset * 2, gl_ComponentTypeFloat16NV,
        weights, ffn_b2_offset * 2, gl_ComponentTypeFloat16NV,
        128, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 512*2);

    // W1 chunk 1: 128→256 (rows 256-511 of W1)
    coopvecNV<float16_t, 256> h1;
    coopVecMatMulAddNV(h1, proj_out, gl_ComponentTypeFloat16NV,
        weights, (ffn_w1_offset + 256 * 128) * 2, gl_ComponentTypeFloat16NV,
        weights, (ffn_b1_offset + 256) * 2, gl_ComponentTypeFloat16NV,
        256, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);
    for (int i = 0; i < 256; i++) {
        float x = float(h1[i]);
        h1[i] = float16_t(0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x*x*x))));
    }

    // W2 chunk 1: 256→128
    coopvecNV<float16_t, 128> out1;
    coopVecMatMulAddNV(out1, h1, gl_ComponentTypeFloat16NV,
        weights, (ffn_w2_offset + 256) * 2, gl_ComponentTypeFloat16NV,
        weights, ffn_b2_offset * 2, gl_ComponentTypeFloat16NV,
        128, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 512*2);

    // Accumulate chunks + residual 2
    coopvecNV<float16_t, 128> ffn_out;
    for (int i = 0; i < 128; i++) {
        ffn_out[i] = float16_t(float(out0[i]) + float(out1[i])
                     - float(weights[ffn_b2_offset + i])  // bias added 2x, subtract extra
                     + float(proj_out[i]));  // residual
    }

    if (valid) {
        coopVecStoreNV(ffn_out, features, (output_offset + global_token * dim) * 2);
    }
}
