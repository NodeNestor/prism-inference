#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// FUSED dec1: NN-upsample(64ch) + concat(enc0 32ch) + PW 96→32 + ReLU
// One dispatch instead of 3. Each thread = one output pixel at r0 (540×960).

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int out_width, out_height;      // r0: 960, 540
    int in_width, in_height;        // r1: 480, 270
    int up_channels;                // 64 (from dec2 output)
    int skip_channels;              // 32 (from enc0)
    int weight_offset, bias_offset; // PW 96→32 weights
    int input_offset;               // dec2 output @ r1 (64ch CHW)
    int skip_offset;                // enc0 skip @ r0 (32ch CHW)
    int output_offset;              // output @ r0 (32ch CHW)
};

layout(set = 0, binding = 0) readonly buffer W { float16_t weights[]; };
layout(set = 0, binding = 1) buffer F { float16_t features[]; };

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int out_pixels = out_width * out_height;
    if (idx >= out_pixels) return;

    int oy = idx / out_width;
    int ox = idx % out_width;
    int iy = min(oy / 2, in_height - 1);
    int ix = min(ox / 2, in_width - 1);
    int in_pixels = in_width * in_height;
    int in_spatial = iy * in_width + ix;

    // Gather 96 channels: 64 upsampled + 32 skip
    coopvecNV<float16_t, 96> inp;
    for (int c = 0; c < 64; c++)
        inp[c] = features[input_offset + c * in_pixels + in_spatial];
    for (int c = 0; c < 32; c++)
        inp[64 + c] = features[skip_offset + c * out_pixels + idx];

    // PW 96→32
    coopvecNV<float16_t, 32> outp;
    coopVecMatMulAddNV(outp, inp, gl_ComponentTypeFloat16NV,
        weights, weight_offset * 2, gl_ComponentTypeFloat16NV,
        weights, bias_offset * 2, gl_ComponentTypeFloat16NV,
        32, 96, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 96 * 2);

    for (int c = 0; c < 32; c++) {
        float16_t v = outp[c];
        if (v < float16_t(0.0)) v = float16_t(0.0);
        features[output_offset + c * out_pixels + idx] = v;
    }
}
