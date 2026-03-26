#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// OPTIMIZED Strided Conv3x3 128->128 (stride=2)
// Splits 128 input channels into 4 groups of 32.
// Each group: gather 32×9 = 288 elements → coopvec<288> → partial output.
// Avoids the coopvec<1152> register pressure cliff.
//
// Same weights, same output — purely an execution optimization.
// Drop-in replacement for strided_conv_coopvec_128ch.spv

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int in_channels;    // 128
    int out_channels;   // 128
    int in_width, in_height;
    int out_width, out_height;
    int weight_offset;  // [128, 1152] row-major
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

    // Process 4 groups of 32 input channels
    // Weight layout: [128, 1152] row-major
    // For group g: input channels g*32..(g+1)*32-1
    // im2col columns: g*32*9..(g+1)*32*9-1 = g*288..(g+1)*288-1
    // Weight submatrix for group g: W[:, g*288:(g+1)*288] = [128, 288] with row stride 1152

    coopvecNV<float16_t, 128> accum;
    // Load bias as initial accumulator
    coopVecLoadNV(accum, weights, bias_offset * 2);

    for (int g = 0; g < 4; g++) {
        int ch_start = g * 32;

        // Gather im2col patch for this group: 32 channels × 3×3 = 288 values
        coopvecNV<float16_t, 288> pv;
        int p = 0;
        for (int ic = ch_start; ic < ch_start + 32; ic++) {
            int ch_base = input_offset + ic * in_pixels;
            for (int ky = -1; ky <= 1; ky++) {
                int iy = cy + ky;
                for (int kx = -1; kx <= 1; kx++) {
                    int ix = cx + kx;
                    float16_t val = float16_t(0.0);
                    if (iy >= 0 && iy < in_height && ix >= 0 && ix < in_width)
                        val = features[ch_base + iy * in_width + ix];
                    pv[p++] = val;
                }
            }
        }

        // Matmul: [128, 288] × [288] → [128] partial output
        // Weight offset for this group: columns g*288 in the [128, 1152] matrix
        // Row stride = 1152 elements
        coopvecNV<float16_t, 128> partial;
        coopVecMatMulAddNV(partial, pv, gl_ComponentTypeFloat16NV,
            weights, (weight_offset + g * 288) * 2, gl_ComponentTypeFloat16NV,
            weights, bias_offset * 2, gl_ComponentTypeFloat16NV,
            128, 288, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 1152 * 2);

        // Accumulate (subtract extra bias for groups 1-3)
        if (g == 0) {
            accum = partial;
        } else {
            for (int i = 0; i < 128; i++)
                accum[i] = float16_t(float(accum[i]) + float(partial[i]) - float(weights[bias_offset + i]));
        }
    }

    // ReLU + store CHW
    for (int c = 0; c < 128; c++) {
        float16_t v = accum[c];
        if (v < float16_t(0.0)) v = float16_t(0.0);
        features[output_offset + c * out_pixels + idx] = v;
    }
}
