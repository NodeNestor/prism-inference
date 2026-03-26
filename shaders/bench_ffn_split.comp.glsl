#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// FFN with SPLIT coopvec: do W1 as 4x coopvec<128> instead of 1x coopvec<512>
// Hypothesis: coopvec<512> register pressure kills occupancy.
// If we split into 4 chunks of 128, register pressure drops 4x.

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

    // W1: 128→512, split into 4 chunks of 128→128
    // W1 layout: [512, 128] row-major → rows 0-127 = chunk0, 128-255 = chunk1, etc.
    // Each chunk: [128, 128] sub-matrix

    // Process chunk 0 (hidden[0:128])
    coopvecNV<float16_t, 128> h0;
    coopVecMatMulAddNV(h0, inp, gl_ComponentTypeFloat16NV,
        weights, (w1_offset + 0 * 128 * 128) * 2, gl_ComponentTypeFloat16NV,
        weights, (b1_offset + 0 * 128) * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);
    // ReLU (cheaper than GELU for this test)
    for (int i = 0; i < 128; i++) {
        float x = float(h0[i]);
        h0[i] = float16_t(x * float(x > 0.0));
    }

    // W2 chunk 0: hidden[0:128] → output contribution
    // W2 layout: [128, 512] row-major → columns 0-127 = chunk0 weights
    // For W2 chunked: W2[:,0:128] is a [128, 128] sub-matrix at offset w2_offset
    // But row stride is still 512! So we need stride parameter = 512*2
    coopvecNV<float16_t, 128> out0;
    coopVecMatMulAddNV(out0, h0, gl_ComponentTypeFloat16NV,
        weights, w2_offset * 2, gl_ComponentTypeFloat16NV,
        weights, b2_offset * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 512*2);

    // Process chunk 1 (hidden[128:256])
    coopvecNV<float16_t, 128> h1;
    coopVecMatMulAddNV(h1, inp, gl_ComponentTypeFloat16NV,
        weights, (w1_offset + 1 * 128 * 128) * 2, gl_ComponentTypeFloat16NV,
        weights, (b1_offset + 1 * 128) * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);
    for (int i = 0; i < 128; i++) {
        float x = float(h1[i]);
        h1[i] = float16_t(x * float(x > 0.0));
    }
    coopvecNV<float16_t, 128> out1;
    // No bias for accumulation chunks (bias already added from chunk 0)
    // Actually we need to handle this differently — W2 matmul over chunks accumulates
    // out = h0 @ W2[0:128,:] + h1 @ W2[128:256,:] + h2 @ W2[256:384,:] + h3 @ W2[384:512,:]
    // Each W2 chunk: W2[k*128:(k+1)*128, :] is [128, 128] with row stride 512
    coopVecMatMulAddNV(out1, h1, gl_ComponentTypeFloat16NV,
        weights, (w2_offset + 128) * 2, gl_ComponentTypeFloat16NV,
        weights, b2_offset * 2, gl_ComponentTypeFloat16NV,  // dummy bias (will add properly)
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 512*2);

    // Process chunk 2
    coopvecNV<float16_t, 128> h2;
    coopVecMatMulAddNV(h2, inp, gl_ComponentTypeFloat16NV,
        weights, (w1_offset + 2 * 128 * 128) * 2, gl_ComponentTypeFloat16NV,
        weights, (b1_offset + 2 * 128) * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);
    for (int i = 0; i < 128; i++) {
        float x = float(h2[i]);
        h2[i] = float16_t(x * float(x > 0.0));
    }
    coopvecNV<float16_t, 128> out2;
    coopVecMatMulAddNV(out2, h2, gl_ComponentTypeFloat16NV,
        weights, (w2_offset + 256) * 2, gl_ComponentTypeFloat16NV,
        weights, b2_offset * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 512*2);

    // Process chunk 3
    coopvecNV<float16_t, 128> h3;
    coopVecMatMulAddNV(h3, inp, gl_ComponentTypeFloat16NV,
        weights, (w1_offset + 3 * 128 * 128) * 2, gl_ComponentTypeFloat16NV,
        weights, (b1_offset + 3 * 128) * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);
    for (int i = 0; i < 128; i++) {
        float x = float(h3[i]);
        h3[i] = float16_t(x * float(x > 0.0));
    }
    coopvecNV<float16_t, 128> out3;
    coopVecMatMulAddNV(out3, h3, gl_ComponentTypeFloat16NV,
        weights, (w2_offset + 384) * 2, gl_ComponentTypeFloat16NV,
        weights, b2_offset * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 512*2);

    // Accumulate: result = out0 + out1 + out2 + out3 + bias + residual
    coopvecNV<float16_t, 128> result;
    for (int i = 0; i < 128; i++) {
        result[i] = float16_t(float(out0[i]) + float(out1[i]) + float(out2[i]) + float(out3[i])
                    - 3.0 * float(features[b2_offset + i])  // subtract extra biases (added 4x, want 1x)
                    + float(inp[i]));  // residual
    }

    coopVecStoreNV(result, features, (output_offset + tid * dim) * 2);
}
