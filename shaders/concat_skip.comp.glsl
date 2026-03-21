#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

// Concatenate two CHW feature maps along channel dimension
// Output: [ch_a + ch_b, H, W] from input A [ch_a, H, W] and B [ch_b, H, W]
// Used for U-Net skip connections: decoder output + encoder skip

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int ch_a;           // channels from source A (decoder upsample output)
    int ch_b;           // channels from source B (encoder skip)
    int width;
    int height;
    int offset_a;       // CHW offset of source A
    int offset_b;       // CHW offset of source B
    int output_offset;  // CHW offset of output
};

layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int pixels = width * height;
    int total_ch = ch_a + ch_b;
    int total_elements = total_ch * pixels;
    if (idx >= total_elements) return;

    int ch = idx / pixels;
    int spatial = idx % pixels;

    float16_t val;
    if (ch < ch_a) {
        val = features[offset_a + ch * pixels + spatial];
    } else {
        val = features[offset_b + (ch - ch_a) * pixels + spatial];
    }
    features[output_offset + ch * pixels + spatial] = val;
}
