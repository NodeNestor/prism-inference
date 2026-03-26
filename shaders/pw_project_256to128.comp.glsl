#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Projection: 256 → 128 dim (adapter after dim=256 transformer)

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int n_tokens;
    int w_offset, b_offset;
    int input_offset, output_offset;
};

layout(set = 0, binding = 0) readonly buffer W { float16_t weights[]; };
layout(set = 0, binding = 1) buffer F { float16_t features[]; };

void main() {
    int tid = int(gl_GlobalInvocationID.x);
    if (tid >= n_tokens) return;

    coopvecNV<float16_t, 256> inp;
    coopVecLoadNV(inp, features, (input_offset + tid * 256) * 2);

    coopvecNV<float16_t, 128> proj;
    coopVecMatMulAddNV(proj, inp, gl_ComponentTypeFloat16NV,
        weights, w_offset * 2, gl_ComponentTypeFloat16NV,
        weights, b_offset * 2, gl_ComponentTypeFloat16NV,
        128, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);

    coopVecStoreNV(proj, features, (output_offset + tid * 128) * 2);
}
