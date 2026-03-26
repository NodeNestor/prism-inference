#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// 2-Split Pass 1 (dim=256): QKV + Windowed Attention + Out Proj
// No FFN = low register pressure. Writes proj_out for MoE FFN pass.

layout(local_size_x = 64) in;

layout(push_constant) uniform PC {
    int n_tokens, dim, n_heads, head_dim;
    int spatial_w, spatial_h, window_size;
    int qkv_w, qkv_b, out_w, out_b;
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

    coopvecNV<float16_t, 256> inp;
    if (valid) { coopVecLoadNV(inp, features, (input_offset + gt * dim) * 2); }
    else { inp = coopvecNV<float16_t, 256>(float16_t(0.0)); }

    coopvecNV<float16_t, 256> Q, K_full, V_full;
    coopVecMatMulAddNV(Q, inp, gl_ComponentTypeFloat16NV,
        weights, qkv_w * 2, gl_ComponentTypeFloat16NV,
        weights, qkv_b * 2, gl_ComponentTypeFloat16NV,
        256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
    coopVecMatMulAddNV(K_full, inp, gl_ComponentTypeFloat16NV,
        weights, (qkv_w + 256*256) * 2, gl_ComponentTypeFloat16NV,
        weights, (qkv_b + 256) * 2, gl_ComponentTypeFloat16NV,
        256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
    coopVecMatMulAddNV(V_full, inp, gl_ComponentTypeFloat16NV,
        weights, (qkv_w + 2*256*256) * 2, gl_ComponentTypeFloat16NV,
        weights, (qkv_b + 2*256) * 2, gl_ComponentTypeFloat16NV,
        256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);

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

    coopvecNV<float16_t, 256> po;
    coopVecMatMulAddNV(po, ao, gl_ComponentTypeFloat16NV,
        weights, out_w * 2, gl_ComponentTypeFloat16NV,
        weights, out_b * 2, gl_ComponentTypeFloat16NV,
        256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
    for (int i = 0; i < 256; i++) po[i] = po[i] + inp[i];

    if (valid) { coopVecStoreNV(po, features, (output_offset + gt * dim) * 2); }
}
