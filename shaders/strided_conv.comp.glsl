#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Strided 3x3 Conv (stride=2) using cooperative vectors
// Downsamples spatial by 2x, changes channels
// im2col per output pixel: gather 3x3 neighborhood from input, matmul to output
// Each thread = one OUTPUT pixel

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int in_channels;    // e.g. 32
    int out_channels;   // e.g. 64
    int in_width;
    int in_height;
    int out_width;      // in_width / 2
    int out_height;     // in_height / 2
    int weight_offset;  // [out_ch, in_ch*9] fp16 elements
    int bias_offset;
    int input_offset;   // CHW format
    int output_offset;  // CHW format
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int out_pixels = out_width * out_height;
    if (idx >= out_pixels) return;

    int oy = idx / out_width;
    int ox = idx % out_width;

    // Input center pixel (stride 2)
    int cx = ox * 2;
    int cy = oy * 2;
    int in_pixels = in_width * in_height;

    // Gather im2col: in_channels * 9 values
    // Max: 128 * 9 = 1152 — too big for one coopvec
    // Process with a loop and accumulate

    // For each output channel, compute dot product with input patch
    for (int oc = 0; oc < out_channels; oc++) {
        float sum = 0.0;
        int w_row = weight_offset + oc * in_channels * 9;

        for (int ic = 0; ic < in_channels; ic++) {
            for (int ky = -1; ky <= 1; ky++) {
                int iy = cy + ky;
                if (iy < 0 || iy >= in_height) continue;
                for (int kx = -1; kx <= 1; kx++) {
                    int ix = cx + kx;
                    if (ix < 0 || ix >= in_width) continue;
                    float val = float(features[input_offset + ic * in_pixels + iy * in_width + ix]);
                    int widx = w_row + ic * 9 + (ky+1) * 3 + (kx+1);
                    sum += val * float(weights[widx]);
                }
            }
        }
        sum += float(weights[bias_offset + oc]);
        if (sum < 0.0) sum = 0.0; // ReLU
        features[output_offset + oc * out_pixels + idx] = float16_t(sum);
    }
}
