#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

// Depthwise 3x3 Conv — HWC format
// Features stored as [H*W][channels], so channels are contiguous per pixel.
// Each thread processes ONE pixel, ALL channels.

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int channels;
    int width;
    int height;
    int weight_offset;
    int bias_offset;    // -1 = no bias
    int input_offset;   // HWC offset (fp16 elements)
    int output_offset;  // HWC offset
    int relu;
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int pixels = width * height;
    if (idx >= pixels) return;

    int oy = idx / width;
    int ox = idx % width;

    for (int ch = 0; ch < channels; ch++) {
        float sum = 0.0;

        // Load 3x3 kernel weights for this channel
        int w_base = weight_offset + ch * 9;

        for (int ky = -1; ky <= 1; ky++) {
            int iy = oy + ky;
            if (iy < 0 || iy >= height) continue;
            for (int kx = -1; kx <= 1; kx++) {
                int ix = ox + kx;
                if (ix < 0 || ix >= width) continue;
                // HWC: pixel (ix,iy), channel ch
                float val = float(features[input_offset + (iy * width + ix) * channels + ch]);
                sum += val * float(weights[w_base + (ky+1) * 3 + (kx+1)]);
            }
        }

        if (bias_offset >= 0) sum += float(weights[bias_offset + ch]);
        if (relu == 1 && sum < 0.0) sum = 0.0;

        features[output_offset + idx * channels + ch] = float16_t(sum);
    }
}
