#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Benchmark: 12 matmul calls per thread (same sized 256×256 each)
// Simulates a full transformer block's matmul count
layout(local_size_x = 256) in;

layout(push_constant) uniform PC { int n_tokens, w_offset, b_offset, in_offset, out_offset; };
layout(set = 0, binding = 0) readonly buffer W { float16_t weights[]; };
layout(set = 0, binding = 1) buffer F { float16_t features[]; };

void main() {
    int tid = int(gl_GlobalInvocationID.x);
    if (tid >= n_tokens) return;
    coopvecNV<float16_t, 256> v;
    coopVecLoadNV(v, features, (in_offset + tid * 256) * 2);

    coopvecNV<float16_t, 256> r;
    // 12 sequential matmul calls (each reuses same weights — tests per-call overhead)
    coopVecMatMulAddNV(r, v, gl_ComponentTypeFloat16NV, weights, w_offset*2, gl_ComponentTypeFloat16NV, weights, b_offset*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
    coopVecMatMulAddNV(r, r, gl_ComponentTypeFloat16NV, weights, w_offset*2, gl_ComponentTypeFloat16NV, weights, b_offset*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
    coopVecMatMulAddNV(r, r, gl_ComponentTypeFloat16NV, weights, w_offset*2, gl_ComponentTypeFloat16NV, weights, b_offset*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
    coopVecMatMulAddNV(r, r, gl_ComponentTypeFloat16NV, weights, w_offset*2, gl_ComponentTypeFloat16NV, weights, b_offset*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
    coopVecMatMulAddNV(r, r, gl_ComponentTypeFloat16NV, weights, w_offset*2, gl_ComponentTypeFloat16NV, weights, b_offset*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
    coopVecMatMulAddNV(r, r, gl_ComponentTypeFloat16NV, weights, w_offset*2, gl_ComponentTypeFloat16NV, weights, b_offset*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
    coopVecMatMulAddNV(r, r, gl_ComponentTypeFloat16NV, weights, w_offset*2, gl_ComponentTypeFloat16NV, weights, b_offset*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
    coopVecMatMulAddNV(r, r, gl_ComponentTypeFloat16NV, weights, w_offset*2, gl_ComponentTypeFloat16NV, weights, b_offset*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
    coopVecMatMulAddNV(r, r, gl_ComponentTypeFloat16NV, weights, w_offset*2, gl_ComponentTypeFloat16NV, weights, b_offset*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
    coopVecMatMulAddNV(r, r, gl_ComponentTypeFloat16NV, weights, w_offset*2, gl_ComponentTypeFloat16NV, weights, b_offset*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
    coopVecMatMulAddNV(r, r, gl_ComponentTypeFloat16NV, weights, w_offset*2, gl_ComponentTypeFloat16NV, weights, b_offset*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
    coopVecMatMulAddNV(r, r, gl_ComponentTypeFloat16NV, weights, w_offset*2, gl_ComponentTypeFloat16NV, weights, b_offset*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);

    coopVecStoreNV(r, features, (out_offset + tid * 256) * 2);
}
