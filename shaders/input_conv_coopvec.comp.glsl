#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Input Conv3x3 6→64 using cooperative vectors
// Gathers 3x3 neighborhood × 6 channels = 54 values, then matmul to 64 outputs
// This treats conv3x3 as im2col + matmul (per pixel)

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int in_channels;    // 6
    int out_channels;   // 64
    int width;
    int height;
    int weight_offset;  // [64 × 54] weights (out_ch × in_ch*9)
    int bias_offset;
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

    // Gather im2col: 6 channels × 9 positions = 54 values
    // Pack into cooperative vector
    coopvecNV<float16_t, 54> inputPatch;
    int p = 0;
    for (int ic = 0; ic < 6; ic++) {
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int iy = oy + ky;
                int ix = ox + kx;
                float16_t val = float16_t(0.0);
                if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                    val = features[input_offset + ic * pixels + iy * width + ix];
                }
                inputPatch[p++] = val;
            }
        }
    }

    // Tensor core matmul: [64×54] × [54] → [64]
    coopvecNV<float16_t, 64> outp;
    coopVecMatMulAddNV(
        outp, inputPatch,
        gl_ComponentTypeFloat16NV,
        weights, weight_offset * 2,
        gl_ComponentTypeFloat16NV,
        weights, bias_offset * 2,
        gl_ComponentTypeFloat16NV,
        64, 54,
        gl_CooperativeVectorMatrixLayoutRowMajorNV,
        false, 54 * 2
    );

    // ReLU + store CHW
    for (int c = 0; c < 64; c++) {
        float16_t v = outp[c];
        if (relu == 1 && v < float16_t(0.0)) v = float16_t(0.0);
        features[output_offset + c * pixels + idx] = v;
    }
}
