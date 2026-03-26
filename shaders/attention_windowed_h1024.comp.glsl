#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// WINDOWED Self-Attention — FFN hidden=1024 variant
// Same as base attention_windowed but with 2x wider FFN for scaling benchmarks.
// dim=128, hidden=1024 (base: hidden=512)

layout(local_size_x = 64) in;

layout(push_constant) uniform PC {
    int n_tokens;
    int dim;            // 128
    int n_heads;        // 4
    int head_dim;       // 32
    int spatial_w;
    int spatial_h;
    int window_size;    // 8
    int qkv_w_offset;
    int qkv_b_offset;
    int out_w_offset;
    int out_b_offset;
    int ffn_w1_offset;  // [1024, 128]
    int ffn_b1_offset;
    int ffn_w2_offset;  // [128, 1024]
    int ffn_b2_offset;
    int input_offset;
    int output_offset;
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

    // === STEP 1: QKV Projection (tensor cores) ===
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

    // === STEP 2: Windowed Attention (64 tokens) ===
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
            for (int d = 0; d < head_dim; d++) {
                s += q[d] * float(shared_k[j * 128 + ch_off + d]);
            }
            scores[j] = s * scale;
            if (scores[j] > max_s) max_s = scores[j];
        }

        float sum_exp = 0.0;
        float result[32];
        for (int d = 0; d < head_dim; d++) result[d] = 0.0;

        for (int j = 0; j < window_tokens; j++) {
            float w = exp(scores[j] - max_s);
            sum_exp += w;
            for (int d = 0; d < head_dim; d++) {
                result[d] += w * float(shared_v[j * 128 + ch_off + d]);
            }
        }

        float inv = 1.0 / max(sum_exp, 1e-6);
        for (int d = 0; d < head_dim; d++) {
            attn_out[ch_off + d] = float16_t(result[d] * inv);
        }
    }

    // === STEP 3: Output Projection (tensor cores) ===
    coopvecNV<float16_t, 128> proj_out;
    coopVecMatMulAddNV(proj_out, attn_out, gl_ComponentTypeFloat16NV,
        weights, out_w_offset * 2, gl_ComponentTypeFloat16NV,
        weights, out_b_offset * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

    // Residual 1
    for (int i = 0; i < 128; i++) proj_out[i] = proj_out[i] + inp[i];

    // === STEP 4: FFN with hidden=1024 (tensor cores) ===
    coopvecNV<float16_t, 1024> ffn_hidden;
    coopVecMatMulAddNV(ffn_hidden, proj_out, gl_ComponentTypeFloat16NV,
        weights, ffn_w1_offset * 2, gl_ComponentTypeFloat16NV,
        weights, ffn_b1_offset * 2, gl_ComponentTypeFloat16NV,
        1024, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

    // GELU
    for (int i = 0; i < 1024; i++) {
        float x = float(ffn_hidden[i]);
        ffn_hidden[i] = float16_t(0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x*x*x))));
    }

    coopvecNV<float16_t, 128> ffn_out;
    coopVecMatMulAddNV(ffn_out, ffn_hidden, gl_ComponentTypeFloat16NV,
        weights, ffn_w2_offset * 2, gl_ComponentTypeFloat16NV,
        weights, ffn_b2_offset * 2, gl_ComponentTypeFloat16NV,
        128, 1024, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 1024*2);

    // Residual 2
    for (int i = 0; i < 128; i++) ffn_out[i] = ffn_out[i] + proj_out[i];

    if (valid) {
        coopVecStoreNV(ffn_out, features, (output_offset + global_token * dim) * 2);
    }
}
