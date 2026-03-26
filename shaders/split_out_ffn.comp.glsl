#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Split Pipeline Pass 3: Output Projection + Residual + FFN
// 256 threads/WG, 1 token/thread. Reads attn_out + original input.

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int n_tokens, dim;
    int out_w_offset, out_b_offset;
    int ffn_w1_offset, ffn_b1_offset;
    int ffn_w2_offset, ffn_b2_offset;
    int attn_offset;    // attention output
    int input_offset;   // original input (for residual)
    int output_offset;  // final output
};

layout(set = 0, binding = 0) readonly buffer W { float16_t weights[]; };
layout(set = 0, binding = 1) buffer F { float16_t features[]; };

void main() {
    int tid = int(gl_GlobalInvocationID.x);
    if (tid >= n_tokens) return;

    // Load attention output
    coopvecNV<float16_t, 128> attn_out;
    coopVecLoadNV(attn_out, features, (attn_offset + tid * dim) * 2);

    // Load original input for residual
    coopvecNV<float16_t, 128> inp;
    coopVecLoadNV(inp, features, (input_offset + tid * dim) * 2);

    // Output projection
    coopvecNV<float16_t, 128> proj_out;
    coopVecMatMulAddNV(proj_out, attn_out, gl_ComponentTypeFloat16NV,
        weights, out_w_offset * 2, gl_ComponentTypeFloat16NV,
        weights, out_b_offset * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

    // Residual 1
    for (int i = 0; i < 128; i++) proj_out[i] = proj_out[i] + inp[i];

    // FFN
    coopvecNV<float16_t, 512> hidden;
    coopVecMatMulAddNV(hidden, proj_out, gl_ComponentTypeFloat16NV,
        weights, ffn_w1_offset * 2, gl_ComponentTypeFloat16NV,
        weights, ffn_b1_offset * 2, gl_ComponentTypeFloat16NV,
        512, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

    for (int i = 0; i < 512; i++) {
        float x = float(hidden[i]);
        hidden[i] = float16_t(0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x*x*x))));
    }

    coopvecNV<float16_t, 128> ffn_out;
    coopVecMatMulAddNV(ffn_out, hidden, gl_ComponentTypeFloat16NV,
        weights, ffn_w2_offset * 2, gl_ComponentTypeFloat16NV,
        weights, ffn_b2_offset * 2, gl_ComponentTypeFloat16NV,
        128, 512, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 512*2);

    for (int i = 0; i < 128; i++) ffn_out[i] = ffn_out[i] + proj_out[i];

    coopVecStoreNV(ffn_out, features, (output_offset + tid * dim) * 2);
}
