#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

// PixelShuffle(2) + Sigmoid — rearranges channels to spatial pixels
// Input:  [12, H, W] (3 * 2^2 channels at render res)
// Output: [3, 2H, 2W] (RGB at display res)
// Also applies sigmoid activation.

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    int render_width;
    int render_height;
    int display_width;
    int display_height;
    int scale;            // 2 or 3
    int input_offset;
    int output_offset;
};

layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

// Output texture (game's display buffer)
layout(set = 0, binding = 2) writeonly buffer OutputRGB { float16_t output_rgb[]; };

void main() {
    // Each thread writes one output pixel
    int ox = int(gl_GlobalInvocationID.x);
    int oy = int(gl_GlobalInvocationID.y);
    if (ox >= display_width || oy >= display_height) return;

    // Map display pixel to render pixel + sub-pixel offset
    int rx = ox / scale;
    int ry = oy / scale;
    int sx = ox % scale;  // sub-pixel x (0 or 1 for scale=2)
    int sy = oy % scale;  // sub-pixel y

    int render_pixels = render_width * render_height;

    // For each RGB channel
    for (int c = 0; c < 3; c++) {
        // PixelShuffle maps: channel (c * scale^2 + sy * scale + sx) at (ry, rx)
        int src_ch = c * scale * scale + sy * scale + sx;
        int src_idx = input_offset + src_ch * render_pixels + ry * render_width + rx;

        float16_t val = features[src_idx];

        // Sigmoid: 1 / (1 + exp(-x))
        float v = float(val);
        v = 1.0 / (1.0 + exp(-v));

        // Write to output (CHW format)
        int dst_idx = output_offset + c * display_width * display_height + oy * display_width + ox;
        output_rgb[dst_idx] = float16_t(v);
    }
}
