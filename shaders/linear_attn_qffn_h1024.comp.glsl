#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Linear Attention — Phase 3: Query + Attention Output + FFN
// dim=128, FFN hidden=1024 variant (8x expansion)

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int n_tokens;
    int dim;            // 128
    int n_heads;        // 4
    int head_dim;       // 32
    int w_q_offset;
    int b_q_offset;
    int w_o_offset;
    int b_o_offset;
    int ffn_w1_offset;  // [1024, 128]
    int ffn_b1_offset;
    int ffn_w2_offset;  // [128, 1024]
    int ffn_b2_offset;
    int input_offset;
    int output_offset;
    int s_offset;
    int z_offset;
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int tid = int(gl_GlobalInvocationID.x);
    if (tid >= n_tokens) return;

    coopvecNV<float16_t, 128> inp;
    coopVecLoadNV(inp, features, (input_offset + tid * dim) * 2);

    // Q Projection
    coopvecNV<float16_t, 128> Q;
    coopVecMatMulAddNV(Q, inp, gl_ComponentTypeFloat16NV,
        weights, w_q_offset * 2, gl_ComponentTypeFloat16NV,
        weights, b_q_offset * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

    // φ(Q) = ELU+1
    float phi_q[128];
    for (int i = 0; i < 128; i++) {
        float x = float(Q[i]);
        phi_q[i] = x >= 0.0 ? x + 1.0 : exp(x);
    }

    // Linear attention: φ(Q) @ S / (φ(Q) · z) per head
    coopvecNV<float16_t, 128> attn_out;
    for (int h = 0; h < 4; h++) {
        int hoff = h * 32;
        int s_head = s_offset + h * 32 * 32;
        int z_head = z_offset + h * 32;

        float out_h[32];
        for (int j = 0; j < 32; j++) {
            float sum = 0.0;
            for (int i = 0; i < 32; i++) {
                sum += phi_q[hoff + i] * float(features[s_head + i * 32 + j]);
            }
            out_h[j] = sum;
        }

        float norm = 0.0;
        for (int i = 0; i < 32; i++) {
            norm += phi_q[hoff + i] * float(features[z_head + i]);
        }
        norm = max(norm, 1e-6);

        for (int j = 0; j < 32; j++) {
            attn_out[hoff + j] = float16_t(out_h[j] / norm);
        }
    }

    // Output projection
    coopvecNV<float16_t, 128> proj_out;
    coopVecMatMulAddNV(proj_out, attn_out, gl_ComponentTypeFloat16NV,
        weights, w_o_offset * 2, gl_ComponentTypeFloat16NV,
        weights, b_o_offset * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

    for (int i = 0; i < 128; i++) proj_out[i] = proj_out[i] + inp[i];

    // FFN: 128 → 1024 → 128
    coopvecNV<float16_t, 1024> ffn_hidden;
    coopVecMatMulAddNV(ffn_hidden, proj_out, gl_ComponentTypeFloat16NV,
        weights, ffn_w1_offset * 2, gl_ComponentTypeFloat16NV,
        weights, ffn_b1_offset * 2, gl_ComponentTypeFloat16NV,
        1024, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

    for (int i = 0; i < 1024; i++) {
        float x = float(ffn_hidden[i]);
        ffn_hidden[i] = float16_t(0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x*x*x))));
    }

    coopvecNV<float16_t, 128> ffn_out;
    coopVecMatMulAddNV(ffn_out, ffn_hidden, gl_ComponentTypeFloat16NV,
        weights, ffn_w2_offset * 2, gl_ComponentTypeFloat16NV,
        weights, ffn_b2_offset * 2, gl_ComponentTypeFloat16NV,
        128, 1024, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 1024*2);

    for (int i = 0; i < 128; i++) ffn_out[i] = ffn_out[i] + proj_out[i];

    coopVecStoreNV(ffn_out, features, (output_offset + tid * dim) * 2);
}
