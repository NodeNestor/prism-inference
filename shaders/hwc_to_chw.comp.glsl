#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

// Convert HWC → CHW format

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int channels;
    int width;
    int height;
    int input_offset;   // HWC source (fp16 elements)
    int output_offset;  // CHW destination (fp16 elements)
};

layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int pixels = width * height;
    if (idx >= pixels) return;

    for (int c = 0; c < channels; c++) {
        float16_t val = features[input_offset + idx * channels + c];
        features[output_offset + c * pixels + idx] = val;
    }
}
