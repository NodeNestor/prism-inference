#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

// Depthwise 3x3 Convolution — one kernel per channel
// z-dimension = channel index for full parallelism

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    int channels;
    int width;
    int height;
    int weight_offset;
    int bias_offset;    // -1 = no bias
    int input_offset;
    int output_offset;
    int relu;
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int ox = int(gl_GlobalInvocationID.x);
    int oy = int(gl_GlobalInvocationID.y);
    int ch = int(gl_GlobalInvocationID.z);

    if (ox >= width || oy >= height || ch >= channels) return;

    int pixels = height * width;
    float sum = 0.0;

    // Load 3x3 kernel weights into registers
    float w[9];
    for (int i = 0; i < 9; i++) {
        w[i] = float(weights[weight_offset + ch * 9 + i]);
    }

    // 3x3 convolution with zero-padding
    for (int ky = -1; ky <= 1; ky++) {
        int iy = oy + ky;
        if (iy < 0 || iy >= height) continue;
        for (int kx = -1; kx <= 1; kx++) {
            int ix = ox + kx;
            if (ix < 0 || ix >= width) continue;
            float val = float(features[input_offset + ch * pixels + iy * width + ix]);
            sum += val * w[(ky+1) * 3 + (kx+1)];
        }
    }

    if (bias_offset >= 0) sum += float(weights[bias_offset + ch]);
    if (relu == 1 && sum < 0.0) sum = 0.0;
    features[output_offset + ch * pixels + oy * width + ox] = float16_t(sum);
}
