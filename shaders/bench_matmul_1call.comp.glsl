#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Benchmark: 1 matmul call per thread (256×256)
layout(local_size_x = 256) in;

layout(push_constant) uniform PC { int n_tokens, w_offset, b_offset, in_offset, out_offset; };
layout(set = 0, binding = 0) readonly buffer W { float16_t weights[]; };
layout(set = 0, binding = 1) buffer F { float16_t features[]; };

void main() {
    int tid = int(gl_GlobalInvocationID.x);
    if (tid >= n_tokens) return;
    coopvecNV<float16_t, 256> inp;
    coopVecLoadNV(inp, features, (in_offset + tid * 256) * 2);
    coopvecNV<float16_t, 256> result;
    coopVecMatMulAddNV(result, inp, gl_ComponentTypeFloat16NV,
        weights, w_offset * 2, gl_ComponentTypeFloat16NV,
        weights, b_offset * 2, gl_ComponentTypeFloat16NV,
        256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
    coopVecStoreNV(result, features, (out_offset + tid * 256) * 2);
}
