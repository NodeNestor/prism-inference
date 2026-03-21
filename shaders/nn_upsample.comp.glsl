#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

// Nearest-neighbor 2x upsample in CHW format
// Each thread copies one input pixel to a 2x2 block in output
// Input: [channels, in_h, in_w], Output: [channels, out_h, out_w]

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int channels;
    int in_width;
    int in_height;
    int out_width;      // in_width * 2
    int out_height;     // in_height * 2
    int input_offset;   // CHW
    int output_offset;  // CHW
};

layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int in_pixels = in_width * in_height;
    int total = channels * in_pixels;
    if (idx >= total) return;

    int ch = idx / in_pixels;
    int spatial = idx % in_pixels;
    int iy = spatial / in_width;
    int ix = spatial % in_width;

    float16_t val = features[input_offset + ch * in_pixels + iy * in_width + ix];

    int out_pixels = out_width * out_height;
    int out_base = output_offset + ch * out_pixels;
    int oy = iy * 2;
    int ox = ix * 2;

    // Write 2x2 block
    out_base += oy * out_width + ox;
    features[out_base] = val;
    features[out_base + 1] = val;
    features[out_base + out_width] = val;
    features[out_base + out_width + 1] = val;
}
