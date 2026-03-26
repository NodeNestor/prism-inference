#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
// MoE FFN: 16 experts, 256→256→256 each. Compile-time unrolled.
// Router picks best expert, all computed (no warp divergence).

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int n_tokens, dim;
    int router_w, router_b;
    int expert_stride;
    int exp0_w1, exp0_b1, exp0_w2, exp0_b2;
    int input_offset, output_offset;
};

layout(set = 0, binding = 0) readonly buffer W { float16_t weights[]; };
layout(set = 0, binding = 1) buffer F { float16_t features[]; };

void main() {
    int tid = int(gl_GlobalInvocationID.x);
    if (tid >= n_tokens) return;
    coopvecNV<float16_t, 256> inp;
    coopVecLoadNV(inp, features, (input_offset + tid * dim) * 2);

    // Router: 256→16
    coopvecNV<float16_t, 16> logits;
    coopVecMatMulAddNV(logits, inp, gl_ComponentTypeFloat16NV,
        weights, router_w * 2, gl_ComponentTypeFloat16NV,
        weights, router_b * 2, gl_ComponentTypeFloat16NV,
        16, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
    int best = 0; float bv = float(logits[0]);
    for (int e = 1; e < 16; e++) { float v = float(logits[e]); if (v > bv) { bv = v; best = e; } }

    coopvecNV<float16_t, 256> result = coopvecNV<float16_t, 256>(float16_t(0.0));
    { // Expert 0
        int eoff = 0 * expert_stride;
        coopvecNV<float16_t, 256> h;
        coopVecMatMulAddNV(h, inp, gl_ComponentTypeFloat16NV, weights, (exp0_w1+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b1+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        for (int i = 0; i < 256; i++) { float x = float(h[i]); h[i] = float16_t(0.5*x*(1.0+tanh(0.7978846*(x+0.044715*x*x*x)))); }
        coopvecNV<float16_t, 256> eo;
        coopVecMatMulAddNV(eo, h, gl_ComponentTypeFloat16NV, weights, (exp0_w2+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b2+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        float m = (best == 0) ? 1.0 : 0.0;
        for (int i = 0; i < 256; i++) result[i] = float16_t(float(result[i]) + float(eo[i]) * m);
    }
    { // Expert 1
        int eoff = 1 * expert_stride;
        coopvecNV<float16_t, 256> h;
        coopVecMatMulAddNV(h, inp, gl_ComponentTypeFloat16NV, weights, (exp0_w1+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b1+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        for (int i = 0; i < 256; i++) { float x = float(h[i]); h[i] = float16_t(0.5*x*(1.0+tanh(0.7978846*(x+0.044715*x*x*x)))); }
        coopvecNV<float16_t, 256> eo;
        coopVecMatMulAddNV(eo, h, gl_ComponentTypeFloat16NV, weights, (exp0_w2+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b2+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        float m = (best == 1) ? 1.0 : 0.0;
        for (int i = 0; i < 256; i++) result[i] = float16_t(float(result[i]) + float(eo[i]) * m);
    }
    { // Expert 2
        int eoff = 2 * expert_stride;
        coopvecNV<float16_t, 256> h;
        coopVecMatMulAddNV(h, inp, gl_ComponentTypeFloat16NV, weights, (exp0_w1+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b1+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        for (int i = 0; i < 256; i++) { float x = float(h[i]); h[i] = float16_t(0.5*x*(1.0+tanh(0.7978846*(x+0.044715*x*x*x)))); }
        coopvecNV<float16_t, 256> eo;
        coopVecMatMulAddNV(eo, h, gl_ComponentTypeFloat16NV, weights, (exp0_w2+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b2+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        float m = (best == 2) ? 1.0 : 0.0;
        for (int i = 0; i < 256; i++) result[i] = float16_t(float(result[i]) + float(eo[i]) * m);
    }
    { // Expert 3
        int eoff = 3 * expert_stride;
        coopvecNV<float16_t, 256> h;
        coopVecMatMulAddNV(h, inp, gl_ComponentTypeFloat16NV, weights, (exp0_w1+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b1+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        for (int i = 0; i < 256; i++) { float x = float(h[i]); h[i] = float16_t(0.5*x*(1.0+tanh(0.7978846*(x+0.044715*x*x*x)))); }
        coopvecNV<float16_t, 256> eo;
        coopVecMatMulAddNV(eo, h, gl_ComponentTypeFloat16NV, weights, (exp0_w2+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b2+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        float m = (best == 3) ? 1.0 : 0.0;
        for (int i = 0; i < 256; i++) result[i] = float16_t(float(result[i]) + float(eo[i]) * m);
    }
    { // Expert 4
        int eoff = 4 * expert_stride;
        coopvecNV<float16_t, 256> h;
        coopVecMatMulAddNV(h, inp, gl_ComponentTypeFloat16NV, weights, (exp0_w1+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b1+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        for (int i = 0; i < 256; i++) { float x = float(h[i]); h[i] = float16_t(0.5*x*(1.0+tanh(0.7978846*(x+0.044715*x*x*x)))); }
        coopvecNV<float16_t, 256> eo;
        coopVecMatMulAddNV(eo, h, gl_ComponentTypeFloat16NV, weights, (exp0_w2+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b2+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        float m = (best == 4) ? 1.0 : 0.0;
        for (int i = 0; i < 256; i++) result[i] = float16_t(float(result[i]) + float(eo[i]) * m);
    }
    { // Expert 5
        int eoff = 5 * expert_stride;
        coopvecNV<float16_t, 256> h;
        coopVecMatMulAddNV(h, inp, gl_ComponentTypeFloat16NV, weights, (exp0_w1+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b1+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        for (int i = 0; i < 256; i++) { float x = float(h[i]); h[i] = float16_t(0.5*x*(1.0+tanh(0.7978846*(x+0.044715*x*x*x)))); }
        coopvecNV<float16_t, 256> eo;
        coopVecMatMulAddNV(eo, h, gl_ComponentTypeFloat16NV, weights, (exp0_w2+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b2+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        float m = (best == 5) ? 1.0 : 0.0;
        for (int i = 0; i < 256; i++) result[i] = float16_t(float(result[i]) + float(eo[i]) * m);
    }
    { // Expert 6
        int eoff = 6 * expert_stride;
        coopvecNV<float16_t, 256> h;
        coopVecMatMulAddNV(h, inp, gl_ComponentTypeFloat16NV, weights, (exp0_w1+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b1+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        for (int i = 0; i < 256; i++) { float x = float(h[i]); h[i] = float16_t(0.5*x*(1.0+tanh(0.7978846*(x+0.044715*x*x*x)))); }
        coopvecNV<float16_t, 256> eo;
        coopVecMatMulAddNV(eo, h, gl_ComponentTypeFloat16NV, weights, (exp0_w2+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b2+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        float m = (best == 6) ? 1.0 : 0.0;
        for (int i = 0; i < 256; i++) result[i] = float16_t(float(result[i]) + float(eo[i]) * m);
    }
    { // Expert 7
        int eoff = 7 * expert_stride;
        coopvecNV<float16_t, 256> h;
        coopVecMatMulAddNV(h, inp, gl_ComponentTypeFloat16NV, weights, (exp0_w1+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b1+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        for (int i = 0; i < 256; i++) { float x = float(h[i]); h[i] = float16_t(0.5*x*(1.0+tanh(0.7978846*(x+0.044715*x*x*x)))); }
        coopvecNV<float16_t, 256> eo;
        coopVecMatMulAddNV(eo, h, gl_ComponentTypeFloat16NV, weights, (exp0_w2+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b2+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        float m = (best == 7) ? 1.0 : 0.0;
        for (int i = 0; i < 256; i++) result[i] = float16_t(float(result[i]) + float(eo[i]) * m);
    }
    { // Expert 8
        int eoff = 8 * expert_stride;
        coopvecNV<float16_t, 256> h;
        coopVecMatMulAddNV(h, inp, gl_ComponentTypeFloat16NV, weights, (exp0_w1+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b1+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        for (int i = 0; i < 256; i++) { float x = float(h[i]); h[i] = float16_t(0.5*x*(1.0+tanh(0.7978846*(x+0.044715*x*x*x)))); }
        coopvecNV<float16_t, 256> eo;
        coopVecMatMulAddNV(eo, h, gl_ComponentTypeFloat16NV, weights, (exp0_w2+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b2+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        float m = (best == 8) ? 1.0 : 0.0;
        for (int i = 0; i < 256; i++) result[i] = float16_t(float(result[i]) + float(eo[i]) * m);
    }
    { // Expert 9
        int eoff = 9 * expert_stride;
        coopvecNV<float16_t, 256> h;
        coopVecMatMulAddNV(h, inp, gl_ComponentTypeFloat16NV, weights, (exp0_w1+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b1+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        for (int i = 0; i < 256; i++) { float x = float(h[i]); h[i] = float16_t(0.5*x*(1.0+tanh(0.7978846*(x+0.044715*x*x*x)))); }
        coopvecNV<float16_t, 256> eo;
        coopVecMatMulAddNV(eo, h, gl_ComponentTypeFloat16NV, weights, (exp0_w2+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b2+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        float m = (best == 9) ? 1.0 : 0.0;
        for (int i = 0; i < 256; i++) result[i] = float16_t(float(result[i]) + float(eo[i]) * m);
    }
    { // Expert 10
        int eoff = 10 * expert_stride;
        coopvecNV<float16_t, 256> h;
        coopVecMatMulAddNV(h, inp, gl_ComponentTypeFloat16NV, weights, (exp0_w1+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b1+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        for (int i = 0; i < 256; i++) { float x = float(h[i]); h[i] = float16_t(0.5*x*(1.0+tanh(0.7978846*(x+0.044715*x*x*x)))); }
        coopvecNV<float16_t, 256> eo;
        coopVecMatMulAddNV(eo, h, gl_ComponentTypeFloat16NV, weights, (exp0_w2+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b2+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        float m = (best == 10) ? 1.0 : 0.0;
        for (int i = 0; i < 256; i++) result[i] = float16_t(float(result[i]) + float(eo[i]) * m);
    }
    { // Expert 11
        int eoff = 11 * expert_stride;
        coopvecNV<float16_t, 256> h;
        coopVecMatMulAddNV(h, inp, gl_ComponentTypeFloat16NV, weights, (exp0_w1+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b1+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        for (int i = 0; i < 256; i++) { float x = float(h[i]); h[i] = float16_t(0.5*x*(1.0+tanh(0.7978846*(x+0.044715*x*x*x)))); }
        coopvecNV<float16_t, 256> eo;
        coopVecMatMulAddNV(eo, h, gl_ComponentTypeFloat16NV, weights, (exp0_w2+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b2+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        float m = (best == 11) ? 1.0 : 0.0;
        for (int i = 0; i < 256; i++) result[i] = float16_t(float(result[i]) + float(eo[i]) * m);
    }
    { // Expert 12
        int eoff = 12 * expert_stride;
        coopvecNV<float16_t, 256> h;
        coopVecMatMulAddNV(h, inp, gl_ComponentTypeFloat16NV, weights, (exp0_w1+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b1+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        for (int i = 0; i < 256; i++) { float x = float(h[i]); h[i] = float16_t(0.5*x*(1.0+tanh(0.7978846*(x+0.044715*x*x*x)))); }
        coopvecNV<float16_t, 256> eo;
        coopVecMatMulAddNV(eo, h, gl_ComponentTypeFloat16NV, weights, (exp0_w2+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b2+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        float m = (best == 12) ? 1.0 : 0.0;
        for (int i = 0; i < 256; i++) result[i] = float16_t(float(result[i]) + float(eo[i]) * m);
    }
    { // Expert 13
        int eoff = 13 * expert_stride;
        coopvecNV<float16_t, 256> h;
        coopVecMatMulAddNV(h, inp, gl_ComponentTypeFloat16NV, weights, (exp0_w1+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b1+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        for (int i = 0; i < 256; i++) { float x = float(h[i]); h[i] = float16_t(0.5*x*(1.0+tanh(0.7978846*(x+0.044715*x*x*x)))); }
        coopvecNV<float16_t, 256> eo;
        coopVecMatMulAddNV(eo, h, gl_ComponentTypeFloat16NV, weights, (exp0_w2+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b2+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        float m = (best == 13) ? 1.0 : 0.0;
        for (int i = 0; i < 256; i++) result[i] = float16_t(float(result[i]) + float(eo[i]) * m);
    }
    { // Expert 14
        int eoff = 14 * expert_stride;
        coopvecNV<float16_t, 256> h;
        coopVecMatMulAddNV(h, inp, gl_ComponentTypeFloat16NV, weights, (exp0_w1+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b1+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        for (int i = 0; i < 256; i++) { float x = float(h[i]); h[i] = float16_t(0.5*x*(1.0+tanh(0.7978846*(x+0.044715*x*x*x)))); }
        coopvecNV<float16_t, 256> eo;
        coopVecMatMulAddNV(eo, h, gl_ComponentTypeFloat16NV, weights, (exp0_w2+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b2+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        float m = (best == 14) ? 1.0 : 0.0;
        for (int i = 0; i < 256; i++) result[i] = float16_t(float(result[i]) + float(eo[i]) * m);
    }
    { // Expert 15
        int eoff = 15 * expert_stride;
        coopvecNV<float16_t, 256> h;
        coopVecMatMulAddNV(h, inp, gl_ComponentTypeFloat16NV, weights, (exp0_w1+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b1+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        for (int i = 0; i < 256; i++) { float x = float(h[i]); h[i] = float16_t(0.5*x*(1.0+tanh(0.7978846*(x+0.044715*x*x*x)))); }
        coopvecNV<float16_t, 256> eo;
        coopVecMatMulAddNV(eo, h, gl_ComponentTypeFloat16NV, weights, (exp0_w2+eoff)*2, gl_ComponentTypeFloat16NV, weights, (exp0_b2+eoff)*2, gl_ComponentTypeFloat16NV, 256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        float m = (best == 15) ? 1.0 : 0.0;
        for (int i = 0; i < 256; i++) result[i] = float16_t(float(result[i]) + float(eo[i]) * m);
    }
    for (int i = 0; i < 256; i++) result[i] = result[i] + inp[i];
    coopVecStoreNV(result, features, (output_offset + tid * dim) * 2);
}
