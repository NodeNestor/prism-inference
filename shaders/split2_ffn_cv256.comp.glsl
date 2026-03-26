#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// 2-Split Pass 2: FFN with split coopvec<256>
// Low register pressure: only coopvec<128> input + 2x coopvec<256> chunks (sequential)
// 256 threads/WG for maximum occupancy

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int n_tokens, dim;
    int ffn_w1_offset, ffn_b1_offset;
    int ffn_w2_offset, ffn_b2_offset;
    int input_offset, output_offset;
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int tid = int(gl_GlobalInvocationID.x);
    if (tid >= n_tokens) return;

    coopvecNV<float16_t, 128> inp;
    coopVecLoadNV(inp, features, (input_offset + tid * dim) * 2);

    // W1 chunk 0: 128→256, GELU, W2 chunk 0: 256→128
    coopvecNV<float16_t, 256> h0;
    coopVecMatMulAddNV(h0, inp, gl_ComponentTypeFloat16NV,
        weights, ffn_w1_offset * 2, gl_ComponentTypeFloat16NV,
        weights, ffn_b1_offset * 2, gl_ComponentTypeFloat16NV,
        256, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);
    for (int i = 0; i < 256; i++) {
        float x = float(h0[i]);
        h0[i] = float16_t(0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x*x*x))));
    }
    coopvecNV<float16_t, 128> out0;
    coopVecMatMulAddNV(out0, h0, gl_ComponentTypeFloat16NV,
        weights, ffn_w2_offset * 2, gl_ComponentTypeFloat16NV,
        weights, ffn_b2_offset * 2, gl_ComponentTypeFloat16NV,
        128, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 512*2);

    // W1 chunk 1: 128→256, GELU, W2 chunk 1: 256→128
    coopvecNV<float16_t, 256> h1;
    coopVecMatMulAddNV(h1, inp, gl_ComponentTypeFloat16NV,
        weights, (ffn_w1_offset + 256 * 128) * 2, gl_ComponentTypeFloat16NV,
        weights, (ffn_b1_offset + 256) * 2, gl_ComponentTypeFloat16NV,
        256, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);
    for (int i = 0; i < 256; i++) {
        float x = float(h1[i]);
        h1[i] = float16_t(0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x*x*x))));
    }
    coopvecNV<float16_t, 128> out1;
    coopVecMatMulAddNV(out1, h1, gl_ComponentTypeFloat16NV,
        weights, (ffn_w2_offset + 256) * 2, gl_ComponentTypeFloat16NV,
        weights, ffn_b2_offset * 2, gl_ComponentTypeFloat16NV,
        128, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 512*2);

    // Accumulate + residual
    for (int i = 0; i < 128; i++) {
        float val = float(out0[i]) + float(out1[i])
                  - float(weights[ffn_b2_offset + i])  // bias correction
                  + float(inp[i]);  // residual
        features[output_offset + tid * dim + i] = float16_t(val);
    }
}
