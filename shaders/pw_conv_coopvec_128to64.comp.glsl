#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Pointwise conv 128->64 using cooperative vectors (tensor cores)
// CHW format input/output. Each thread = one pixel.

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int in_channels;    // 128
    int out_channels;   // 64
    int width;
    int height;
    int weight_offset;  // [64 x 128] row-major
    int bias_offset;    // [64]
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

    // Gather 128 channels for this pixel from CHW layout
    coopvecNV<float16_t, 128> inp;
    for (int c = 0; c < 128; c++) {
        inp[c] = features[input_offset + c * pixels + idx];
    }

    // Tensor core matmul: [64 x 128] x [128] -> [64]
    coopvecNV<float16_t, 64> outp;
    coopVecMatMulAddNV(
        outp, inp,
        gl_ComponentTypeFloat16NV,
        weights, weight_offset * 2,
        gl_ComponentTypeFloat16NV,
        weights, bias_offset * 2,
        gl_ComponentTypeFloat16NV,
        64, 128,
        gl_CooperativeVectorMatrixLayoutRowMajorNV,
        false, 128 * 2
    );

    // ReLU + store CHW
    for (int c = 0; c < 64; c++) {
        float16_t v = outp[c];
        if (relu == 1 && v < float16_t(0.0)) v = float16_t(0.0);
        features[output_offset + c * pixels + idx] = v;
    }
}
