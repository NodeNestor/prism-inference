#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Strided Conv3x3 128->128 (stride=2) using cooperative vectors
// im2col per output pixel: gather 3x3 x 128 = 1152 values, coopVecMatMulAddNV -> 128 outputs
// Each thread = one OUTPUT pixel

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int in_channels;    // 128
    int out_channels;   // 128
    int in_width;
    int in_height;
    int out_width;
    int out_height;
    int weight_offset;  // [128 x 1152] row-major
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

    // Gather im2col: 128 * 9 = 1152 values
    coopvecNV<float16_t, 1152> patchVec;
    int p = 0;
    for (int ic = 0; ic < 128; ic++) {
        int ch_base = input_offset + ic * in_pixels;
        for (int ky = -1; ky <= 1; ky++) {
            int iy = cy + ky;
            for (int kx = -1; kx <= 1; kx++) {
                int ix = cx + kx;
                float16_t val = float16_t(0.0);
                if (iy >= 0 && iy < in_height && ix >= 0 && ix < in_width) {
                    val = features[ch_base + iy * in_width + ix];
                }
                patchVec[p] = val;
                p++;
            }
        }
    }

    // Tensor core matmul: [128 x 1152] x [1152] -> [128]
    coopvecNV<float16_t, 128> outp;
    coopVecMatMulAddNV(
        outp, patchVec,
        gl_ComponentTypeFloat16NV,
        weights, weight_offset * 2,
        gl_ComponentTypeFloat16NV,
        weights, bias_offset * 2,
        gl_ComponentTypeFloat16NV,
        128, 1152,
        gl_CooperativeVectorMatrixLayoutRowMajorNV,
        false, 1152 * 2
    );

    // ReLU + store CHW
    for (int c = 0; c < 128; c++) {
        float16_t v = outp[c];
        if (v < float16_t(0.0)) v = float16_t(0.0);
        features[output_offset + c * out_pixels + idx] = v;
    }
}
