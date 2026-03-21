#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Input Conv3x3 for V2: 9 channels -> 32 output channels
// im2col per pixel: gather 3x3 x 9 = 81 values, then coopvec matmul to 32 outputs
// Each thread = one pixel

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int in_channels;    // 9  (color3 + depth1 + motion2 + 3 extra = padded to 9)
    int out_channels;   // 32
    int width;
    int height;
    int weight_offset;  // [32 x 81] weights
    int bias_offset;    // [32] bias
    int input_offset;   // CHW
    int output_offset;  // CHW
    int relu;
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int pixels = width * height;
    if (idx >= pixels) return;

    int oy = idx / width;
    int ox = idx % width;

    // Gather im2col: in_channels * 9 = 81 values (for 9ch input)
    // We support up to 9 input channels -> 81 patch values
    coopvecNV<float16_t, 81> inputPatch;
    int p = 0;
    for (int ic = 0; ic < in_channels; ic++) {
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int iy = oy + ky;
                int ix = ox + kx;
                float16_t val = float16_t(0.0);
                if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                    val = features[input_offset + ic * pixels + iy * width + ix];
                }
                if (p < 81) inputPatch[p] = val;
                p++;
            }
        }
    }
    // Zero-pad remaining slots if in_channels < 9
    for (int i = p; i < 81; i++) {
        inputPatch[i] = float16_t(0.0);
    }

    // Tensor core matmul: [32 x 81] x [81] -> [32]
    coopvecNV<float16_t, 32> outp;
    coopVecMatMulAddNV(
        outp, inputPatch,
        gl_ComponentTypeFloat16NV,
        weights, weight_offset * 2,
        gl_ComponentTypeFloat16NV,
        weights, bias_offset * 2,
        gl_ComponentTypeFloat16NV,
        32, 81,
        gl_CooperativeVectorMatrixLayoutRowMajorNV,
        false, 81 * 2
    );

    // ReLU + store CHW
    for (int c = 0; c < 32; c++) {
        float16_t v = outp[c];
        if (relu == 1 && v < float16_t(0.0)) v = float16_t(0.0);
        features[output_offset + c * pixels + idx] = v;
    }
}
