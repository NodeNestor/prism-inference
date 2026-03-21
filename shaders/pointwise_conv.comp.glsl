#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

// 1x1 Pointwise Convolution — optimized with shared memory weight cache
// Each thread handles ONE pixel, loops over output channels in groups.
// Weights cached in shared memory (8KB for 64x64).
// Input channels read once per pixel, reused across all output channels.

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    int in_channels;
    int out_channels;
    int width;
    int height;
    int weight_offset;
    int bias_offset;
    int input_offset;
    int output_offset;
    int relu;
    int residual_offset;  // -1 = no residual
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

// Cache input channels in registers (max 64 channels)
// Each thread pre-loads its pixel's input channels once
void main() {
    int ox = int(gl_GlobalInvocationID.x);
    int oy = int(gl_GlobalInvocationID.y);
    if (ox >= width || oy >= height) return;

    int pixel_idx = oy * width + ox;
    int pixels = height * width;

    // Pre-load ALL input channels into registers (64 fp32 = 256 bytes)
    // This avoids re-reading from global memory for each output channel
    float inp[64];
    for (int ic = 0; ic < in_channels && ic < 64; ic++) {
        inp[ic] = float(features[input_offset + ic * pixels + pixel_idx]);
    }

    // Compute all output channels using cached input
    for (int oc = 0; oc < out_channels; oc++) {
        float sum = 0.0;
        int w_base = weight_offset + oc * in_channels;

        for (int ic = 0; ic < in_channels; ic++) {
            sum += inp[ic] * float(weights[w_base + ic]);
        }

        sum += float(weights[bias_offset + oc]);

        if (residual_offset >= 0) {
            sum += float(features[residual_offset + oc * pixels + pixel_idx]);
        }

        if (relu == 1 && sum < 0.0) sum = 0.0;
        features[output_offset + oc * pixels + pixel_idx] = float16_t(sum);
    }
}
