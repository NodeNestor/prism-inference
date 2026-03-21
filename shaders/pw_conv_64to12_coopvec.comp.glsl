#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// PW Conv 64→12 using cooperative vectors (for output layer)
// Input CHW, output CHW

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int in_channels;    // 64
    int out_channels;   // 12
    int width;
    int height;
    int weight_offset;
    int bias_offset;
    int input_offset;   // CHW
    int output_offset;  // CHW
    int relu;
    int residual_offset;
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int pixels = width * height;
    if (idx >= pixels) return;

    // Load 64 input channels from CHW
    coopvecNV<float16_t, 64> inp;
    for (int c = 0; c < 64; c++) {
        inp[c] = features[input_offset + c * pixels + idx];
    }

    // Tensor core matmul: [12×64] × [64] → [12]
    coopvecNV<float16_t, 12> outp;
    coopVecMatMulAddNV(
        outp, inp,
        gl_ComponentTypeFloat16NV,
        weights, weight_offset * 2,
        gl_ComponentTypeFloat16NV,
        weights, bias_offset * 2,
        gl_ComponentTypeFloat16NV,
        12, 64,
        gl_CooperativeVectorMatrixLayoutRowMajorNV,
        false, 64 * 2
    );

    // Store output CHW
    for (int c = 0; c < 12; c++) {
        features[output_offset + c * pixels + idx] = outp[c];
    }
}
