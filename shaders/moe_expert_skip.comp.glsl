#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// MoE Expert FFN with early skip for non-assigned tokens.
// Testing if coopVecMatMulAddNV allows per-thread divergence.
// If it works: only ~25% of threads do compute (with 4 experts).
// If it crashes: need packed approach with indirect dispatch.

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int n_tokens, dim;
    int expert_id;
    int w1_offset, b1_offset, w2_offset, b2_offset;
    int input_offset, output_offset;
    int assignment_offset;
};

layout(set = 0, binding = 0) readonly buffer W { float16_t weights[]; };
layout(set = 0, binding = 1) buffer F { float16_t features[]; };
layout(set = 0, binding = 2) readonly buffer Aux { int aux[]; };

void main() {
    int tid = int(gl_GlobalInvocationID.x);
    if (tid >= n_tokens) return;

    // Skip if not assigned to this expert
    if (aux[assignment_offset + tid] != expert_id) return;

    coopvecNV<float16_t, 256> inp;
    coopVecLoadNV(inp, features, (input_offset + tid * dim) * 2);

    coopvecNV<float16_t, 256> hidden;
    coopVecMatMulAddNV(hidden, inp, gl_ComponentTypeFloat16NV,
        weights, w1_offset * 2, gl_ComponentTypeFloat16NV,
        weights, b1_offset * 2, gl_ComponentTypeFloat16NV,
        256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);

    for (int i = 0; i < 256; i++) {
        float x = float(hidden[i]);
        hidden[i] = float16_t(0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x*x*x))));
    }

    coopvecNV<float16_t, 256> result;
    coopVecMatMulAddNV(result, hidden, gl_ComponentTypeFloat16NV,
        weights, w2_offset * 2, gl_ComponentTypeFloat16NV,
        weights, b2_offset * 2, gl_ComponentTypeFloat16NV,
        256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);

    for (int i = 0; i < 256; i++) result[i] = result[i] + inp[i];
    coopVecStoreNV(result, features, (output_offset + tid * dim) * 2);
}
