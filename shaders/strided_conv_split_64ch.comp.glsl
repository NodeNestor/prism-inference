#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// OPTIMIZED Strided Conv3x3 64->128 (stride=2)
// Splits 64 input channels into 2 groups of 32.
// Each group: gather 32×9 = 288 elements → coopvec<288> → partial output.
// Avoids the coopvec<576> register pressure cliff.
//
// Drop-in replacement for strided_conv_coopvec_64ch.spv

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int in_channels;    // 64
    int out_channels;   // 128
    int in_width, in_height;
    int out_width, out_height;
    int weight_offset;  // [128, 576] row-major
    int bias_offset;    // [128]
    int input_offset;   // CHW
    int output_offset;  // CHW
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int out_pixels = out_width * out_height;
    if (idx >= out_pixels) return;

    int oy = idx / out_width;
    int ox = idx % out_width;
    int cx = ox * 2;
    int cy = oy * 2;
    int in_pixels = in_width * in_height;

    // Group 0: channels 0-31
    coopvecNV<float16_t, 288> pv0;
    {
        int p = 0;
        for (int ic = 0; ic < 32; ic++) {
            int ch_base = input_offset + ic * in_pixels;
            for (int ky = -1; ky <= 1; ky++) {
                int iy = cy + ky;
                for (int kx = -1; kx <= 1; kx++) {
                    int ix = cx + kx;
                    float16_t val = float16_t(0.0);
                    if (iy >= 0 && iy < in_height && ix >= 0 && ix < in_width)
                        val = features[ch_base + iy * in_width + ix];
                    pv0[p++] = val;
                }
            }
        }
    }
    coopvecNV<float16_t, 128> out0;
    coopVecMatMulAddNV(out0, pv0, gl_ComponentTypeFloat16NV,
        weights, weight_offset * 2, gl_ComponentTypeFloat16NV,
        weights, bias_offset * 2, gl_ComponentTypeFloat16NV,
        128, 288, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 576 * 2);

    // Group 1: channels 32-63
    coopvecNV<float16_t, 288> pv1;
    {
        int p = 0;
        for (int ic = 32; ic < 64; ic++) {
            int ch_base = input_offset + ic * in_pixels;
            for (int ky = -1; ky <= 1; ky++) {
                int iy = cy + ky;
                for (int kx = -1; kx <= 1; kx++) {
                    int ix = cx + kx;
                    float16_t val = float16_t(0.0);
                    if (iy >= 0 && iy < in_height && ix >= 0 && ix < in_width)
                        val = features[ch_base + iy * in_width + ix];
                    pv1[p++] = val;
                }
            }
        }
    }
    coopvecNV<float16_t, 128> out1;
    coopVecMatMulAddNV(out1, pv1, gl_ComponentTypeFloat16NV,
        weights, (weight_offset + 288) * 2, gl_ComponentTypeFloat16NV,
        weights, bias_offset * 2, gl_ComponentTypeFloat16NV,
        128, 288, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 576 * 2);

    // Accumulate + ReLU + store
    for (int c = 0; c < 128; c++) {
        float v = float(out0[c]) + float(out1[c]) - float(weights[bias_offset + c]);  // bias correction
        if (v < 0.0) v = 0.0;
        features[output_offset + c * out_pixels + idx] = float16_t(v);
    }
}
