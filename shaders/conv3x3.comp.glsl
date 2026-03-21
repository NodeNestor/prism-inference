#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

// 3x3 Convolution — z-dimension parallelizes across output channels
// Each thread computes ONE output pixel for ONE output channel.

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
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int ox = int(gl_GlobalInvocationID.x);
    int oy = int(gl_GlobalInvocationID.y);
    int oc = int(gl_GlobalInvocationID.z);

    if (ox >= width || oy >= height || oc >= out_channels) return;

    int pixels = height * width;
    float sum = 0.0;

    for (int ic = 0; ic < in_channels; ic++) {
        // Load 3x3 kernel
        int w_base = weight_offset + (oc * in_channels + ic) * 9;
        for (int ky = -1; ky <= 1; ky++) {
            int iy = oy + ky;
            if (iy < 0 || iy >= height) continue;
            for (int kx = -1; kx <= 1; kx++) {
                int ix = ox + kx;
                if (ix < 0 || ix >= width) continue;
                float val = float(features[input_offset + ic * pixels + iy * width + ix]);
                sum += val * float(weights[w_base + (ky+1) * 3 + (kx+1)]);
            }
        }
    }

    sum += float(weights[bias_offset + oc]);
    if (relu == 1 && sum < 0.0) sum = 0.0;
    features[output_offset + oc * pixels + oy * width + ox] = float16_t(sum);
}
