#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Isolated FFN benchmark — 1 token per thread
// Tests raw tensor core throughput at dim=128, FFN=512
// Minimal register pressure: just the FFN ops

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int n_tokens, dim;
    int w1_offset, b1_offset, w2_offset, b2_offset;
    int input_offset, output_offset;
};

layout(set = 0, binding = 0) readonly buffer W { float16_t weights[]; };
layout(set = 0, binding = 1) buffer F { float16_t features[]; };

void main() {
    int tid = int(gl_GlobalInvocationID.x);
    if (tid >= n_tokens) return;

    coopvecNV<float16_t, 128> inp;
    coopVecLoadNV(inp, features, (input_offset + tid * dim) * 2);

    coopvecNV<float16_t, 512> hidden;
    coopVecMatMulAddNV(hidden, inp, gl_ComponentTypeFloat16NV,
        weights, w1_offset * 2, gl_ComponentTypeFloat16NV,
        weights, b1_offset * 2, gl_ComponentTypeFloat16NV,
        512, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

    for (int i = 0; i < 512; i++) {
        float x = float(hidden[i]);
        hidden[i] = float16_t(x * float(x > 0.0));  // ReLU (cheaper than GELU for benchmarking)
    }

    coopvecNV<float16_t, 128> result;
    coopVecMatMulAddNV(result, hidden, gl_ComponentTypeFloat16NV,
        weights, w2_offset * 2, gl_ComponentTypeFloat16NV,
        weights, b2_offset * 2, gl_ComponentTypeFloat16NV,
        128, 512, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 512*2);

    for (int i = 0; i < 128; i++) result[i] = result[i] + inp[i];

    coopVecStoreNV(result, features, (output_offset + tid * dim) * 2);
}
