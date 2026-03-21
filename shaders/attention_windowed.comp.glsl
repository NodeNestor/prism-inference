#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// WINDOWED Self-Attention with cooperative vectors
// Instead of attending to ALL 8K tokens (N^2 = 67M ops), each token
// attends to its local 8x8 = 64 token window. 128x less compute!
// This is what Swin Transformer and DLSS 5 use.
//
// Full pipeline per token:
//   1. QKV projection (tensor cores)
//   2. Windowed attention (64 tokens, trivial)
//   3. Output projection (tensor cores)
//   4. FFN (tensor cores)
//   5. Residual connections

layout(local_size_x = 64) in;  // one workgroup = one window of 64 tokens

layout(push_constant) uniform PC {
    int n_tokens;       // total tokens (N)
    int dim;            // 128
    int n_heads;        // 4
    int head_dim;       // 32
    int spatial_w;      // width in tokens (120)
    int spatial_h;      // height in tokens (68)
    int window_size;    // 8
    // Weights
    int qkv_w_offset;  // [3*dim, dim]
    int qkv_b_offset;  // [3*dim]
    int out_w_offset;   // [dim, dim]
    int out_b_offset;   // [dim]
    int ffn_w1_offset;  // [hidden, dim]
    int ffn_b1_offset;
    int ffn_w2_offset;  // [dim, hidden]
    int ffn_b2_offset;
    int input_offset;
    int output_offset;
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

// Shared memory for window's K and V (64 tokens × 128 dim = 16KB)
shared float16_t shared_k[64 * 128];  // 16KB
shared float16_t shared_v[64 * 128];  // 16KB

void main() {
    int window_idx = int(gl_WorkGroupID.x);  // which window
    int local_idx = int(gl_LocalInvocationID.x);  // token within window (0..63)

    // Map window to spatial position
    int windows_x = (spatial_w + window_size - 1) / window_size;
    int win_y = window_idx / windows_x;
    int win_x = window_idx % windows_x;

    // Token position within window
    int tok_y = local_idx / window_size;
    int tok_x = local_idx % window_size;

    // Global spatial position
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

    // Store K and V to shared memory
    coopVecStoreNV(K, shared_k, local_idx * 128 * 2);
    coopVecStoreNV(V, shared_v, local_idx * 128 * 2);
    barrier();

    // === STEP 2: Windowed Attention (only 64 tokens!) ===
    // For each head, compute attention over the 64 tokens in this window
    coopvecNV<float16_t, 128> attn_out = coopvecNV<float16_t, 128>(float16_t(0.0));

    for (int h = 0; h < n_heads; h++) {
        int ch_off = h * head_dim;
        float scale = 1.0 / sqrt(float(head_dim));

        // Load Q for this head
        float q[32];
        for (int d = 0; d < head_dim; d++) q[d] = float(Q[ch_off + d]);

        // Compute attention scores + find max
        float scores[64];
        float max_s = -1e9;
        int window_tokens = min(64, n_tokens);  // handle edge windows

        for (int j = 0; j < window_tokens; j++) {
            float s = 0.0;
            for (int d = 0; d < head_dim; d++) {
                s += q[d] * float(shared_k[j * 128 + ch_off + d]);
            }
            scores[j] = s * scale;
            if (scores[j] > max_s) max_s = scores[j];
        }

        // Softmax + weighted sum of V
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

    // Residual 1: attn_out = proj_out + inp
    for (int i = 0; i < 128; i++) proj_out[i] = proj_out[i] + inp[i];

    // === STEP 4: FFN (tensor cores) ===
    coopvecNV<float16_t, 512> ffn_hidden;
    coopVecMatMulAddNV(ffn_hidden, proj_out, gl_ComponentTypeFloat16NV,
        weights, ffn_w1_offset * 2, gl_ComponentTypeFloat16NV,
        weights, ffn_b1_offset * 2, gl_ComponentTypeFloat16NV,
        512, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

    // GELU
    for (int i = 0; i < 512; i++) {
        float x = float(ffn_hidden[i]);
        ffn_hidden[i] = float16_t(0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x*x*x))));
    }

    coopvecNV<float16_t, 128> ffn_out;
    coopVecMatMulAddNV(ffn_out, ffn_hidden, gl_ComponentTypeFloat16NV,
        weights, ffn_w2_offset * 2, gl_ComponentTypeFloat16NV,
        weights, ffn_b2_offset * 2, gl_ComponentTypeFloat16NV,
        128, 512, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 512*2);

    // Residual 2: output = ffn_out + proj_out
    for (int i = 0; i < 128; i++) ffn_out[i] = ffn_out[i] + proj_out[i];

    // Write output
    if (valid) {
        coopVecStoreNV(ffn_out, features, (output_offset + global_token * dim) * 2);
    }
}
