#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Linear Attention — Phase 1: KV Projection + Feature Map
//
// For each token: project K and V, apply ELU+1 feature map to K.
// Store projected φ(K) and V to buffer for the reduce phase.
//
// φ(x) = ELU(x) + 1 = (x >= 0) ? x + 1 : exp(x)
// Always positive — required for linear attention stability.
//
// This replaces the monolithic windowed attention shader with a
// multi-dispatch pipeline that has much better GPU occupancy.

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int n_tokens;       // total tokens (8160)
    int dim;            // 128
    int w_k_offset;     // K weight [dim, dim]
    int b_k_offset;     // K bias [dim]
    int w_v_offset;     // V weight [dim, dim]
    int b_v_offset;     // V bias [dim]
    int input_offset;   // input features in feature buffer
    int k_output_offset;// where to store φ(K)
    int v_output_offset;// where to store V
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int tid = int(gl_GlobalInvocationID.x);
    if (tid >= n_tokens) return;

    // Load input token
    coopvecNV<float16_t, 128> inp;
    coopVecLoadNV(inp, features, (input_offset + tid * dim) * 2);

    // Project K
    coopvecNV<float16_t, 128> K;
    coopVecMatMulAddNV(K, inp, gl_ComponentTypeFloat16NV,
        weights, w_k_offset * 2, gl_ComponentTypeFloat16NV,
        weights, b_k_offset * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

    // Apply feature map φ(K) = ELU+1
    for (int i = 0; i < 128; i++) {
        float x = float(K[i]);
        K[i] = float16_t(x >= 0.0 ? x + 1.0 : exp(x));
    }

    // Project V
    coopvecNV<float16_t, 128> V;
    coopVecMatMulAddNV(V, inp, gl_ComponentTypeFloat16NV,
        weights, w_v_offset * 2, gl_ComponentTypeFloat16NV,
        weights, b_v_offset * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

    // Store φ(K) and V
    coopVecStoreNV(K, features, (k_output_offset + tid * dim) * 2);
    coopVecStoreNV(V, features, (v_output_offset + tid * dim) * 2);
}
