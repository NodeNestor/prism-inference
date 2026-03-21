#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

// Generic Pointwise (1x1) Convolution — no input channel cap
// Handles any in_channels -> any out_channels
// CHW format. Each thread = one pixel, one output channel (z-dispatch).
// Slower than register-cached version but works for any channel count.

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PC {
    int in_channels;
    int out_channels;
    int width;
    int height;
    int weight_offset;
    int bias_offset;
    int input_offset;
    int output_offset;
    int relu;
    int residual_offset;
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int ox = int(gl_GlobalInvocationID.x);
    int oy = int(gl_GlobalInvocationID.y);
    int oc = int(gl_GlobalInvocationID.z);

    if (ox >= width || oy >= height || oc >= out_channels) return;

    int pixel_idx = oy * width + ox;
    int pixels = height * width;

    float sum = 0.0;
    int w_base = weight_offset + oc * in_channels;

    for (int ic = 0; ic < in_channels; ic++) {
        float val = float(features[input_offset + ic * pixels + pixel_idx]);
        sum += val * float(weights[w_base + ic]);
    }

    sum += float(weights[bias_offset + oc]);

    if (residual_offset >= 0) {
        sum += float(features[residual_offset + oc * pixels + pixel_idx]);
    }

    if (relu == 1 && sum < 0.0) sum = 0.0;
    features[output_offset + oc * pixels + pixel_idx] = float16_t(sum);
}
