#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// FUSED dec3: NN-upsample(128ch) + concat(enc2 128ch) + PW 256→128 + ReLU
// One dispatch instead of 3. Eliminates 2 barriers + 2 intermediate buffers.
// Each thread handles one OUTPUT pixel at r2 (135×240).
// Reads directly from r3 (68×120) via NN-upsample logic.

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int out_width, out_height;      // r2: 240, 135
    int in_width, in_height;        // r3: 120, 68
    int skip_channels;              // 128 (from enc2)
    int weight_offset, bias_offset; // PW 256→128 weights
    int input_offset;               // transformer output @ r3 (128ch CHW)
    int skip_offset;                // enc2 skip @ r2 (128ch CHW)
    int output_offset;              // output @ r2 (128ch CHW)
};

layout(set = 0, binding = 0) readonly buffer W { float16_t weights[]; };
layout(set = 0, binding = 1) buffer F { float16_t features[]; };

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int out_pixels = out_width * out_height;
    if (idx >= out_pixels) return;

    int oy = idx / out_width;
    int ox = idx % out_width;

    // NN-upsample: map output pixel to input pixel (floor division by 2)
    int iy = oy / 2;
    int ix = ox / 2;
    if (iy >= in_height) iy = in_height - 1;
    if (ix >= in_width) ix = in_width - 1;
    int in_pixels = in_width * in_height;
    int in_spatial = iy * in_width + ix;

    // Gather 256 channels: 128 from upsampled input + 128 from skip
    coopvecNV<float16_t, 256> inp;
    // First 128: upsampled transformer output (NN-upsample from r3)
    for (int c = 0; c < 128; c++)
        inp[c] = features[input_offset + c * in_pixels + in_spatial];
    // Next 128: skip connection from enc2 (already at r2)
    for (int c = 0; c < 128; c++)
        inp[128 + c] = features[skip_offset + c * out_pixels + idx];

    // PW 256→128 matmul
    coopvecNV<float16_t, 128> outp;
    coopVecMatMulAddNV(outp, inp, gl_ComponentTypeFloat16NV,
        weights, weight_offset * 2, gl_ComponentTypeFloat16NV,
        weights, bias_offset * 2, gl_ComponentTypeFloat16NV,
        128, 256, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 256 * 2);

    // ReLU + store
    for (int c = 0; c < 128; c++) {
        float16_t v = outp[c];
        if (v < float16_t(0.0)) v = float16_t(0.0);
        features[output_offset + c * out_pixels + idx] = v;
    }
}
