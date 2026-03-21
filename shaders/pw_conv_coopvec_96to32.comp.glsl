#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Pointwise conv 96->32 using cooperative vectors (tensor cores)
// CHW format input/output. Each thread = one pixel.
// Used in dec1: upsample(64ch) + skip(32ch) = 96ch -> 32ch

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int in_channels;    // 96
    int out_channels;   // 32
    int width;
    int height;
    int weight_offset;  // [32 x 96] row-major
    int bias_offset;    // [32]
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

    // Gather 96 channels for this pixel from CHW layout
    coopvecNV<float16_t, 96> inp;
    for (int c = 0; c < 96; c++) {
        inp[c] = features[input_offset + c * pixels + idx];
    }

    // Tensor core matmul: [32 x 96] x [96] -> [32]
    coopvecNV<float16_t, 32> outp;
    coopVecMatMulAddNV(
        outp, inp,
        gl_ComponentTypeFloat16NV,
        weights, weight_offset * 2,
        gl_ComponentTypeFloat16NV,
        weights, bias_offset * 2,
        gl_ComponentTypeFloat16NV,
        32, 96,
        gl_CooperativeVectorMatrixLayoutRowMajorNV,
        false, 96 * 2
    );

    // ReLU + store CHW
    for (int c = 0; c < 32; c++) {
        float16_t v = outp[c];
        if (relu == 1 && v < float16_t(0.0)) v = float16_t(0.0);
        features[output_offset + c * pixels + idx] = v;
    }
}
