#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Pointwise conv 32->12 using cooperative vectors (tensor cores)
// Output conv before pixelshuffle. NO ReLU (sigmoid applied in pixelshuffle).
// CHW format input/output. Each thread = one pixel.

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int in_channels;    // 32
    int out_channels;   // 12
    int width;
    int height;
    int weight_offset;  // [12 x 32] row-major
    int bias_offset;    // [12]
    int input_offset;   // CHW
    int output_offset;  // CHW
    int relu;           // 0 for output conv
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int pixels = width * height;
    if (idx >= pixels) return;

    // Gather 32 channels for this pixel from CHW layout
    coopvecNV<float16_t, 32> inp;
    for (int c = 0; c < 32; c++) {
        inp[c] = features[input_offset + c * pixels + idx];
    }

    // Tensor core matmul: [12 x 32] x [32] -> [12]
    coopvecNV<float16_t, 12> outp;
    coopVecMatMulAddNV(
        outp, inp,
        gl_ComponentTypeFloat16NV,
        weights, weight_offset * 2,
        gl_ComponentTypeFloat16NV,
        weights, bias_offset * 2,
        gl_ComponentTypeFloat16NV,
        12, 32,
        gl_CooperativeVectorMatrixLayoutRowMajorNV,
        false, 32 * 2
    );

    // Store CHW (no activation — sigmoid is in pixelshuffle)
    for (int c = 0; c < 12; c++) {
        features[output_offset + c * pixels + idx] = outp[c];
    }
}
