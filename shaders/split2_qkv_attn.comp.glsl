#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// 2-Split Pass 1: QKV Projection + Windowed Attention
// No FFN = no coopvec<256/512> = low register pressure = high occupancy
// Writes attention output + saves input for residual in pass 2

layout(local_size_x = 64) in;

layout(push_constant) uniform PC {
    int n_tokens, dim, n_heads, head_dim;
    int spatial_w, spatial_h, window_size;
    int qkv_w_offset, qkv_b_offset;
    int out_w_offset, out_b_offset;
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

    coopvecNV<float16_t, 128> attn_out = coopvecNV<float16_t, 128>(float16_t(0.0));

    for (int h = 0; h < n_heads; h++) {
        int ch_off = h * head_dim;
        float scale = 1.0 / sqrt(float(head_dim));
        float q[32];
        for (int d = 0; d < head_dim; d++) q[d] = float(Q[ch_off + d]);

        float scores[64];
        float max_s = -1e9;
        int wt = min(64, n_tokens);

        for (int j = 0; j < wt; j++) {
            float s = 0.0;
            for (int d = 0; d < head_dim; d++)
                s += q[d] * float(shared_k[j * 128 + ch_off + d]);
            scores[j] = s * scale;
            if (scores[j] > max_s) max_s = scores[j];
        }

        float sum_exp = 0.0;
        float result[32];
        for (int d = 0; d < head_dim; d++) result[d] = 0.0;
        for (int j = 0; j < wt; j++) {
            float w = exp(scores[j] - max_s);
            sum_exp += w;
            for (int d = 0; d < head_dim; d++)
                result[d] += w * float(shared_v[j * 128 + ch_off + d]);
        }

        float inv = 1.0 / max(sum_exp, 1e-6);
        for (int d = 0; d < head_dim; d++)
            attn_out[ch_off + d] = float16_t(result[d] * inv);
    }

    // Output projection
    coopvecNV<float16_t, 128> proj_out;
    coopVecMatMulAddNV(proj_out, attn_out, gl_ComponentTypeFloat16NV,
        weights, out_w_offset * 2, gl_ComponentTypeFloat16NV,
        weights, out_b_offset * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

    // Residual + store (pass 2 reads this as input)
    for (int i = 0; i < 128; i++) proj_out[i] = proj_out[i] + inp[i];

    if (valid) {
        coopVecStoreNV(proj_out, features, (output_offset + global_token * dim) * 2);
    }
}
