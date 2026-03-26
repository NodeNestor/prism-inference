#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// FFN W2 only: 512→128 + residual. Reads 512-dim hidden from global memory.
// Second dispatch of 2-dispatch FFN approach.

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int n_tokens, dim, hidden_dim;
    int w2_offset, b2_offset;
    int hidden_offset;   // 512-dim hidden state
    int residual_offset; // original input for residual
    int output_offset;
};

layout(set = 0, binding = 0) readonly buffer W { float16_t weights[]; };
layout(set = 0, binding = 1) buffer F { float16_t features[]; };

void main() {
    int tid = int(gl_GlobalInvocationID.x);
    if (tid >= n_tokens) return;

    // Load 512-dim hidden state
    coopvecNV<float16_t, 512> hidden;
    coopVecLoadNV(hidden, features, (hidden_offset + tid * 512) * 2);

    // W2: 512→128
    coopvecNV<float16_t, 128> result;
    coopVecMatMulAddNV(result, hidden, gl_ComponentTypeFloat16NV,
        weights, w2_offset * 2, gl_ComponentTypeFloat16NV,
        weights, b2_offset * 2, gl_ComponentTypeFloat16NV,
        128, 512, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 512*2);

    // Residual
    coopvecNV<float16_t, 128> res;
    coopVecLoadNV(res, features, (residual_offset + tid * dim) * 2);
    for (int i = 0; i < 128; i++) result[i] = result[i] + res[i];

    coopVecStoreNV(result, features, (output_offset + tid * dim) * 2);
}
