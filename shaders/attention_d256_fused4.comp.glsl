#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// FUSED 4-BLOCK dim=256 — compile-time unrolled
// Hardcoded 4 blocks (no runtime loop variable).
// Token state stays in registers across all 4 blocks.
// Weight stride passed via push constants.

layout(local_size_x = 64) in;

layout(push_constant) uniform PC {
    int n_tokens, dim, n_heads, head_dim;
    int spatial_w, spatial_h, window_size;
    int weight_stride;
    int qkv_w, qkv_b, out_w, out_b;
    int ffn_w1, ffn_b1, ffn_w2, ffn_b2;
    int input_offset, output_offset;
};

layout(set = 0, binding = 0) readonly buffer W { float16_t weights[]; };
layout(set = 0, binding = 1) buffer F { float16_t features[]; };

shared float16_t sk[64 * 32];
shared float16_t sv[64 * 32];

// One transformer block — inlined as a function-like macro to help compiler
#define TRANSFORMER_BLOCK(boff) { \
    coopvecNV<float16_t, 256> Q_, K_, V_; \
    coopVecMatMulAddNV(Q_, state, gl_ComponentTypeFloat16NV, \
        weights, (qkv_w + boff) * 2, gl_ComponentTypeFloat16NV, \
        weights, (qkv_b + boff) * 2, gl_ComponentTypeFloat16NV, \
        256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2); \
    coopVecMatMulAddNV(K_, state, gl_ComponentTypeFloat16NV, \
        weights, (qkv_w + boff + 256*256) * 2, gl_ComponentTypeFloat16NV, \
        weights, (qkv_b + boff + 256) * 2, gl_ComponentTypeFloat16NV, \
        256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2); \
    coopVecMatMulAddNV(V_, state, gl_ComponentTypeFloat16NV, \
        weights, (qkv_w + boff + 2*256*256) * 2, gl_ComponentTypeFloat16NV, \
        weights, (qkv_b + boff + 2*256) * 2, gl_ComponentTypeFloat16NV, \
        256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2); \
    coopvecNV<float16_t, 256> ao_ = coopvecNV<float16_t, 256>(float16_t(0.0)); \
    for (int h_ = 0; h_ < 8; h_++) { \
        int ch_ = h_ * 32; \
        for (int d_ = 0; d_ < 32; d_++) { sk[li * 32 + d_] = K_[ch_ + d_]; sv[li * 32 + d_] = V_[ch_ + d_]; } \
        barrier(); \
        float qq_[32]; for (int d_ = 0; d_ < 32; d_++) qq_[d_] = float(Q_[ch_ + d_]); \
        float sc_[64]; float mx_ = -1e9; \
        for (int j_ = 0; j_ < wt; j_++) { float s_ = 0.0; \
            for (int d_ = 0; d_ < 32; d_++) s_ += qq_[d_] * float(sk[j_ * 32 + d_]); \
            s_ *= 0.1767766953; sc_[j_] = s_; if (s_ > mx_) mx_ = s_; } \
        float se_ = 0.0; float rr_[32]; for (int d_ = 0; d_ < 32; d_++) rr_[d_] = 0.0; \
        for (int j_ = 0; j_ < wt; j_++) { float w_ = exp(sc_[j_] - mx_); se_ += w_; \
            for (int d_ = 0; d_ < 32; d_++) rr_[d_] += w_ * float(sv[j_ * 32 + d_]); } \
        float iv_ = 1.0 / max(se_, 1e-6); \
        for (int d_ = 0; d_ < 32; d_++) ao_[ch_ + d_] = float16_t(rr_[d_] * iv_); \
        barrier(); \
    } \
    coopvecNV<float16_t, 256> po_; \
    coopVecMatMulAddNV(po_, ao_, gl_ComponentTypeFloat16NV, \
        weights, (out_w + boff) * 2, gl_ComponentTypeFloat16NV, \
        weights, (out_b + boff) * 2, gl_ComponentTypeFloat16NV, \
        256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2); \
    for (int i_ = 0; i_ < 256; i_++) po_[i_] = po_[i_] + state[i_]; \
    coopvecNV<float16_t, 256> fa_ = coopvecNV<float16_t, 256>(float16_t(0.0)); \
    for (int c_ = 0; c_ < 4; c_++) { \
        coopvecNV<float16_t, 256> hc_; \
        coopVecMatMulAddNV(hc_, po_, gl_ComponentTypeFloat16NV, \
            weights, (ffn_w1 + boff + c_ * 256 * 256) * 2, gl_ComponentTypeFloat16NV, \
            weights, (ffn_b1 + boff + c_ * 256) * 2, gl_ComponentTypeFloat16NV, \
            256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256*2); \
        for (int i_ = 0; i_ < 256; i_++) { float x_ = float(hc_[i_]); \
            hc_[i_] = float16_t(0.5 * x_ * (1.0 + tanh(0.7978846 * (x_ + 0.044715 * x_*x_*x_)))); } \
        coopvecNV<float16_t, 256> oc_; \
        coopVecMatMulAddNV(oc_, hc_, gl_ComponentTypeFloat16NV, \
            weights, (ffn_w2 + boff + c_ * 256) * 2, gl_ComponentTypeFloat16NV, \
            weights, (ffn_b2 + boff) * 2, gl_ComponentTypeFloat16NV, \
            256, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 1024*2); \
        for (int i_ = 0; i_ < 256; i_++) fa_[i_] = float16_t(float(fa_[i_]) + float(oc_[i_])); \
    } \
    for (int i_ = 0; i_ < 256; i_++) \
        state[i_] = float16_t(float(fa_[i_]) - 3.0 * float(weights[ffn_b2 + boff + i_]) + float(po_[i_])); \
}

void main() {
    int wi = int(gl_WorkGroupID.x);
    int li = int(gl_LocalInvocationID.x);
    int wx = (spatial_w + window_size - 1) / window_size;
    int wy_ = wi / wx, wxx = wi % wx;
    int ty = li / window_size, tx = li % window_size;
    int gy = wy_ * window_size + ty, gx = wxx * window_size + tx;
    bool valid = (gy < spatial_h && gx < spatial_w);
    int gt = valid ? (gy * spatial_w + gx) : 0;
    int wt = min(64, n_tokens);

    coopvecNV<float16_t, 256> state;
    if (valid) { coopVecLoadNV(state, features, (input_offset + gt * dim) * 2); }
    else { state = coopvecNV<float16_t, 256>(float16_t(0.0)); }

    // 4 blocks, fully unrolled at compile time
    TRANSFORMER_BLOCK(0 * weight_stride)
    TRANSFORMER_BLOCK(1 * weight_stride)
    TRANSFORMER_BLOCK(2 * weight_stride)
    TRANSFORMER_BLOCK(3 * weight_stride)

    if (valid) { coopVecStoreNV(state, features, (output_offset + gt * dim) * 2); }
}
#undef TRANSFORMER_BLOCK
