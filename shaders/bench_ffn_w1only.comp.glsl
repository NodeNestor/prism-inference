#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// FFN W1 only: 128→512 + ReLU. Writes 512-dim hidden to global memory.
// For 2-dispatch FFN approach. Only uses coopvec<128> input, outputs via loop.

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int n_tokens, dim, hidden_dim;
    int w1_offset, b1_offset;
    int input_offset, output_offset;
};

layout(set = 0, binding = 0) readonly buffer W { float16_t weights[]; };
layout(set = 0, binding = 1) buffer F { float16_t features[]; };

void main() {
    int tid = int(gl_GlobalInvocationID.x);
    if (tid >= n_tokens) return;

    coopvecNV<float16_t, 128> inp;
    coopVecLoadNV(inp, features, (input_offset + tid * dim) * 2);

    // W1: 128→512 using coopvec<512>
    coopvecNV<float16_t, 512> hidden;
    coopVecMatMulAddNV(hidden, inp, gl_ComponentTypeFloat16NV,
        weights, w1_offset * 2, gl_ComponentTypeFloat16NV,
        weights, b1_offset * 2, gl_ComponentTypeFloat16NV,
        512, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

    // ReLU + store
    for (int i = 0; i < 512; i++) {
        float x = float(hidden[i]);
        features[output_offset + tid * 512 + i] = float16_t(x * float(x > 0.0));
    }
}
