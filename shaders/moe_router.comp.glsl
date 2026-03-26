#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// MoE Router: assigns each token to one of N experts
// Writes expert_id per token + per-expert token counts for indirect dispatch

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int n_tokens, dim, n_experts;
    int router_w, router_b;     // [n_experts, dim]
    int input_offset;
    int assignment_offset;      // output: int32 per token (expert id)
    int count_offset;           // output: int32 per expert (token count) — atomically incremented
};

layout(set = 0, binding = 0) readonly buffer W { float16_t weights[]; };
layout(set = 0, binding = 1) buffer F { float16_t features[]; };
// Binding 2: assignment + count buffer (int32)
layout(set = 0, binding = 2) buffer Aux { int aux[]; };

void main() {
    int tid = int(gl_GlobalInvocationID.x);
    if (tid >= n_tokens) return;

    coopvecNV<float16_t, 256> inp;
    coopVecLoadNV(inp, features, (input_offset + tid * dim) * 2);

    // Router matmul: 256→n_experts (use coopvec<4> for 4 experts)
    // We support up to 4 experts with coopvec<4>
    coopvecNV<float16_t, 4> logits;
    coopVecMatMulAddNV(logits, inp, gl_ComponentTypeFloat16NV,
        weights, router_w * 2, gl_ComponentTypeFloat16NV,
        weights, router_b * 2, gl_ComponentTypeFloat16NV,
        4, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);

    // Argmax
    int best = 0;
    float bv = float(logits[0]);
    for (int e = 1; e < n_experts; e++) {
        float v = float(logits[e]);
        if (v > bv) { bv = v; best = e; }
    }

    // Write assignment
    aux[assignment_offset + tid] = best;

    // Atomic increment expert count
    atomicAdd(aux[count_offset + best], 1);
}
