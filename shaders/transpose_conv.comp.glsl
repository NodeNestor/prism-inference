#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

// ConvTranspose2d k=4, stride=2, pad=1 — upsamples 2x
// Naive per-output-pixel loop. Runs at low res (68x120 -> 135x240 etc) so perf is fine.
// Each thread writes one output pixel for one output channel.
//
// ConvTranspose2d relationship: for output pixel (ox, oy):
//   ox = ix * stride - pad + kx  =>  ix = (ox + pad - kx) / stride
//   Only contributes when (ox + pad - kx) % stride == 0
//
// With stride=2, pad=1, k=4:
//   For each (kx, ky) in [0..3]x[0..3]:
//     ix = (ox + 1 - kx) / 2,  only if (ox + 1 - kx) % 2 == 0 and ix in range
//     iy = (oy + 1 - ky) / 2,  only if (oy + 1 - ky) % 2 == 0 and iy in range

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PC {
    int in_channels;    // e.g. 128
    int out_channels;   // e.g. 128
    int in_width;       // input spatial
    int in_height;
    int out_width;      // output spatial (2x input)
    int out_height;
    int weight_offset;  // [in_ch, out_ch, 4, 4] fp16 elements
    int bias_offset;
    int input_offset;   // CHW
    int output_offset;  // CHW
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int ox = int(gl_GlobalInvocationID.x);
    int oy = int(gl_GlobalInvocationID.y);
    int oc = int(gl_GlobalInvocationID.z);

    if (ox >= out_width || oy >= out_height || oc >= out_channels) return;

    int in_pixels = in_width * in_height;
    int out_pixels = out_width * out_height;

    float sum = 0.0;

    // stride=2, pad=1, kernel=4
    for (int ic = 0; ic < in_channels; ic++) {
        for (int ky = 0; ky < 4; ky++) {
            int tmp_y = oy + 1 - ky;  // oy + pad - ky
            if (tmp_y % 2 != 0) continue;
            int iy = tmp_y / 2;
            if (iy < 0 || iy >= in_height) continue;

            for (int kx = 0; kx < 4; kx++) {
                int tmp_x = ox + 1 - kx;
                if (tmp_x % 2 != 0) continue;
                int ix = tmp_x / 2;
                if (ix < 0 || ix >= in_width) continue;

                float val = float(features[input_offset + ic * in_pixels + iy * in_width + ix]);
                // Weight layout: [in_ch, out_ch, ky, kx]
                int widx = weight_offset + ((ic * out_channels + oc) * 4 + ky) * 4 + kx;
                sum += val * float(weights[widx]);
            }
        }
    }

    sum += float(weights[bias_offset + oc]);
    features[output_offset + oc * out_pixels + oy * out_width + ox] = float16_t(sum);
}
