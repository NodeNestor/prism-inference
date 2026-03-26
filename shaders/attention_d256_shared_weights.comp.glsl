#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// dim=256 with SHARED WEIGHTS (ALBERT-style)
// All blocks reuse the same weight matrices.
// Benefits: weights always in L2 cache after first block,
// total model size stays small, can use many blocks cheaply.
// N blocks controlled via push constant.

layout(local_size_x = 64) in;

layout(push_constant) uniform PC {
    int n_tokens, dim, n_heads, head_dim;
    int spatial_w, spatial_h, window_size;
    int n_blocks;
    int qkv_w, qkv_b, out_w, out_b;
    int ffn_w1, ffn_b1, ffn_w2, ffn_b2;
    int input_offset, output_offset;
};

layout(set = 0, binding = 0) readonly buffer W { float16_t weights[]; };
layout(set = 0, binding = 1) buffer F { float16_t features[]; };

shared float16_t sk[64 * 32];
shared float16_t sv[64 * 32];

void main() {
    int wi = int(gl_WorkGroupID.x);
    int li = int(gl_LocalInvocationID.x);
    int wx = (spatial_w + window_size - 1) / window_size;
    int wy = wi / wx, wxx = wi % wx;
    int ty = li / window_size, tx = li % window_size;
    int gy = wy * window_size + ty, gx = wxx * window_size + tx;
    bool valid = (gy < spatial_h && gx < spatial_w);
    int gt = valid ? (gy * spatial_w + gx) : 0;
    int wt = min(64, n_tokens);

    coopvecNV<float16_t, 256> state;
    if (valid) { coopVecLoadNV(state, features, (input_offset + gt * dim) * 2); }
    else { state = coopvecNV<float16_t, 256>(float16_t(0.0)); }

    // All blocks use SAME weights (stride = 0)
    for (int blk = 0; blk < n_blocks; blk++) {
        // QKV
        coopvecNV<float16_t, 256> Q, K_full, V_full;
        coopVecMatMulAddNV(Q, state, gl_ComponentTypeFloat16NV,
            weights, qkv_w * 2, gl_ComponentTypeFloat16NV,
            weights, qkv_b * 2, gl_ComponentTypeFloat16NV,
            256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        coopVecMatMulAddNV(K_full, state, gl_ComponentTypeFloat16NV,
            weights, (qkv_w + 256*256) * 2, gl_ComponentTypeFloat16NV,
            weights, (qkv_b + 256) * 2, gl_ComponentTypeFloat16NV,
            256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        coopVecMatMulAddNV(V_full, state, gl_ComponentTypeFloat16NV,
            weights, (qkv_w + 2*256*256) * 2, gl_ComponentTypeFloat16NV,
            weights, (qkv_b + 2*256) * 2, gl_ComponentTypeFloat16NV,
            256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);

        // Per-head attention
        coopvecNV<float16_t, 256> ao = coopvecNV<float16_t, 256>(float16_t(0.0));
        for (int h = 0; h < 8; h++) {
            int ch = h * 32;
            for (int d = 0; d < 32; d++) { sk[li*32+d] = K_full[ch+d]; sv[li*32+d] = V_full[ch+d]; }
            barrier();
            float q[32]; for (int d = 0; d < 32; d++) q[d] = float(Q[ch+d]);
            float sc[64]; float mx = -1e9;
            for (int j = 0; j < wt; j++) { float s = 0.0;
                for (int d = 0; d < 32; d++) s += q[d] * float(sk[j*32+d]);
                s *= 0.1767766953; sc[j] = s; if (s > mx) mx = s; }
            float se = 0.0; float rr[32]; for (int d = 0; d < 32; d++) rr[d] = 0.0;
            for (int j = 0; j < wt; j++) { float w = exp(sc[j]-mx); se += w;
                for (int d = 0; d < 32; d++) rr[d] += w * float(sv[j*32+d]); }
            float iv = 1.0 / max(se, 1e-6);
            for (int d = 0; d < 32; d++) ao[ch+d] = float16_t(rr[d] * iv);
            barrier();
        }

        // Output projection + residual
        coopvecNV<float16_t, 256> po;
        coopVecMatMulAddNV(po, ao, gl_ComponentTypeFloat16NV,
            weights, out_w * 2, gl_ComponentTypeFloat16NV,
            weights, out_b * 2, gl_ComponentTypeFloat16NV,
            256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
        for (int i = 0; i < 256; i++) po[i] = po[i] + state[i];

        // FFN split 4x cv256
        coopvecNV<float16_t, 256> fa = coopvecNV<float16_t, 256>(float16_t(0.0));
        for (int c = 0; c < 4; c++) {
            coopvecNV<float16_t, 256> hc;
            coopVecMatMulAddNV(hc, po, gl_ComponentTypeFloat16NV,
                weights, (ffn_w1 + c*256*256) * 2, gl_ComponentTypeFloat16NV,
                weights, (ffn_b1 + c*256) * 2, gl_ComponentTypeFloat16NV,
                256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
            for (int i = 0; i < 256; i++) { float x = float(hc[i]);
                hc[i] = float16_t(0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x*x*x)))); }
            coopvecNV<float16_t, 256> oc;
            coopVecMatMulAddNV(oc, hc, gl_ComponentTypeFloat16NV,
                weights, (ffn_w2 + c*256) * 2, gl_ComponentTypeFloat16NV,
                weights, ffn_b2 * 2, gl_ComponentTypeFloat16NV,
                256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 1024*2);
            for (int i = 0; i < 256; i++) fa[i] = float16_t(float(fa[i]) + float(oc[i]));
        }
        for (int i = 0; i < 256; i++)
            state[i] = float16_t(float(fa[i]) - 3.0 * float(weights[ffn_b2 + i]) + float(po[i]));
    }

    if (valid) { coopVecStoreNV(state, features, (output_offset + gt * dim) * 2); }
}
