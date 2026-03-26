#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// FFN with coopvec<256>: split 512 into 2 chunks of 256
// Tests if 256-wide vectors have better occupancy than 512

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

    // W1 chunk 0: 128→256 (rows 0-255 of W1)
    coopvecNV<float16_t, 256> h0;
    coopVecMatMulAddNV(h0, inp, gl_ComponentTypeFloat16NV,
        weights, w1_offset * 2, gl_ComponentTypeFloat16NV,
        weights, b1_offset * 2, gl_ComponentTypeFloat16NV,
        256, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);
    for (int i = 0; i < 256; i++) {
        float x = float(h0[i]);
        h0[i] = float16_t(x * float(x > 0.0));
    }

    // W2 chunk 0: hidden[0:256] @ W2[0:256, :] → 128-dim partial output
    coopvecNV<float16_t, 128> out0;
    coopVecMatMulAddNV(out0, h0, gl_ComponentTypeFloat16NV,
        weights, w2_offset * 2, gl_ComponentTypeFloat16NV,
        weights, b2_offset * 2, gl_ComponentTypeFloat16NV,
        128, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 512*2);

    // W1 chunk 1: 128→256 (rows 256-511 of W1)
    coopvecNV<float16_t, 256> h1;
    coopVecMatMulAddNV(h1, inp, gl_ComponentTypeFloat16NV,
        weights, (w1_offset + 256 * 128) * 2, gl_ComponentTypeFloat16NV,
        weights, (b1_offset + 256) * 2, gl_ComponentTypeFloat16NV,
        256, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);
    for (int i = 0; i < 256; i++) {
        float x = float(h1[i]);
        h1[i] = float16_t(x * float(x > 0.0));
    }

    // W2 chunk 1: hidden[256:512] @ W2[256:512, :] → 128-dim partial output
    coopvecNV<float16_t, 128> out1;
    coopVecMatMulAddNV(out1, h1, gl_ComponentTypeFloat16NV,
        weights, (w2_offset + 256) * 2, gl_ComponentTypeFloat16NV,
        weights, b2_offset * 2, gl_ComponentTypeFloat16NV,
        128, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 512*2);

    // Accumulate + residual
    coopvecNV<float16_t, 128> result;
    for (int i = 0; i < 128; i++) {
        result[i] = float16_t(float(out0[i]) + float(out1[i])
                    - float(features[b2_offset + i])  // bias added 2x, want 1x
                    + float(inp[i]));
    }

    coopVecStoreNV(result, features, (output_offset + tid * dim) * 2);
}
