#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Split Pipeline Pass 1: QKV Projection Only
// 256 threads/WG, 1 token/thread. Low register pressure.
// Writes Q, K, V to global memory for the attention pass.

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int n_tokens, dim;
    int qkv_w_offset, qkv_b_offset;
    int input_offset;
    int q_output_offset, k_output_offset, v_output_offset;
};

layout(set = 0, binding = 0) readonly buffer W { float16_t weights[]; };
layout(set = 0, binding = 1) buffer F { float16_t features[]; };

void main() {
    int tid = int(gl_GlobalInvocationID.x);
    if (tid >= n_tokens) return;

    coopvecNV<float16_t, 128> inp;
    coopVecLoadNV(inp, features, (input_offset + tid * dim) * 2);

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

    coopVecStoreNV(Q, features, (q_output_offset + tid * dim) * 2);
    coopVecStoreNV(K, features, (k_output_offset + tid * dim) * 2);
    coopVecStoreNV(V, features, (v_output_offset + tid * dim) * 2);
}
