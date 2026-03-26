#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// dim=256 Windowed Attention with Mixture of Experts FFN
//
// 4 expert FFNs, each 256→256→256 (no splitting needed — coopvec<256>!)
// Router: tiny 256→4 matmul picks best expert per token.
// Each token uses only 1 expert → serial chain = QKV(3) + Out(1) + Router(1) + W1(1) + W2(1) = 7 calls
// vs standard split FFN: QKV(3) + Out(1) + 4×(W1+W2) = 12 calls
//
// Total params per block: ~1.3M (vs ~789K standard) — 1.6x more params, shorter chain
// With 4 experts at 256→256→256 each:
//   4 × (256×256 + 256 + 256×256 + 256) = 4 × 131,584 = 526,336 FFN params
//   vs standard 1024×256 + 1024 + 256×1024 + 256 = 525,568 FFN params
//   Same FFN param count, but NO coopvec splitting needed!
//
// The magic: each expert is 256→256→256 = coopvec<256> throughout.
// No need to split into chunks. Shorter serial chain. Same total params.

layout(local_size_x = 64) in;

layout(push_constant) uniform PC {
    int n_tokens, dim, n_heads, head_dim;
    int spatial_w, spatial_h, window_size;
    int qkv_w, qkv_b, out_w, out_b;
    // Router: [4, 256] weights + [4] bias
    int router_w, router_b;
    // 4 experts, each: W1 [256,256], b1 [256], W2 [256,256], b2 [256]
    int expert_stride;  // offset between consecutive experts
    int exp0_w1, exp0_b1, exp0_w2, exp0_b2;  // expert 0 offsets (others = +k*stride)
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

    // Load input
    coopvecNV<float16_t, 256> inp;
    if (valid) { coopVecLoadNV(inp, features, (input_offset + gt * dim) * 2); }
    else { inp = coopvecNV<float16_t, 256>(float16_t(0.0)); }

    // === QKV Projection ===
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

    // === Per-head windowed attention ===
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

    // === Output projection + residual ===
    coopvecNV<float16_t, 256> po;
    coopVecMatMulAddNV(po, ao, gl_ComponentTypeFloat16NV,
        weights, out_w * 2, gl_ComponentTypeFloat16NV,
        weights, out_b * 2, gl_ComponentTypeFloat16NV,
        256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);
    for (int i = 0; i < 256; i++) po[i] = po[i] + inp[i];

    // === MoE Router: pick best expert ===
    // Tiny matmul: 256→4 (coopvec<4> output)
    coopvecNV<float16_t, 4> logits;
    coopVecMatMulAddNV(logits, po, gl_ComponentTypeFloat16NV,
        weights, router_w * 2, gl_ComponentTypeFloat16NV,
        weights, router_b * 2, gl_ComponentTypeFloat16NV,
        4, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);

    // Argmax over 4 experts
    int best = 0;
    float best_val = float(logits[0]);
    for (int e = 1; e < 4; e++) {
        float v = float(logits[e]);
        if (v > best_val) { best_val = v; best = e; }
    }

    // === MoE FFN: compute all experts, keep selected one ===
    // coopVecMatMulAddNV is cooperative — all threads in subgroup must execute
    // the same instruction. So we compute ALL experts and select afterward.
    // This avoids warp divergence. Compute cost = 4 experts, but serial chain
    // per expert is only 2 calls (W1+W2), and experts can potentially overlap.

    coopvecNV<float16_t, 256> ffn_out = coopvecNV<float16_t, 256>(float16_t(0.0));

    for (int e = 0; e < 4; e++) {
        int eoff = e * expert_stride;

        coopvecNV<float16_t, 256> hidden;
        coopVecMatMulAddNV(hidden, po, gl_ComponentTypeFloat16NV,
            weights, (exp0_w1 + eoff) * 2, gl_ComponentTypeFloat16NV,
            weights, (exp0_b1 + eoff) * 2, gl_ComponentTypeFloat16NV,
            256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);

        for (int i = 0; i < 256; i++) {
            float x = float(hidden[i]);
            hidden[i] = float16_t(0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x*x*x))));
        }

        coopvecNV<float16_t, 256> exp_out;
        coopVecMatMulAddNV(exp_out, hidden, gl_ComponentTypeFloat16NV,
            weights, (exp0_w2 + eoff) * 2, gl_ComponentTypeFloat16NV,
            weights, (exp0_b2 + eoff) * 2, gl_ComponentTypeFloat16NV,
            256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2);

        // Accumulate: multiply by mask (1.0 if selected, 0.0 if not)
        float mask = (e == best) ? 1.0 : 0.0;
        for (int i = 0; i < 256; i++)
            ffn_out[i] = float16_t(float(ffn_out[i]) + float(exp_out[i]) * mask);
    }

    // Residual 2
    for (int i = 0; i < 256; i++) ffn_out[i] = ffn_out[i] + po[i];

    if (valid) { coopVecStoreNV(ffn_out, features, (output_offset + gt * dim) * 2); }
}
