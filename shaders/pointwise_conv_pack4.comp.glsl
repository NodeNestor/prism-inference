#version 450
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

// 1x1 Pointwise Convolution — PACK4 format
// Features stored as [ch/4][H*W][4] fp16 — 4 channels interleaved
// Each f16vec4 load reads 4 channels at one pixel in 8 bytes (coalesced!)
// Weights stored as [out_ch/4][in_ch/4][4][4] fp16

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    int in_channels;   // always multiple of 4
    int out_channels;  // always multiple of 4
    int width;
    int height;
    int weight_offset; // into weight buffer (in fp16 elements)
    int bias_offset;
    int input_offset;  // into feature buffer (in f16vec4 elements)
    int output_offset;
    int relu;
    int residual_offset; // -1 = no residual, else f16vec4 offset
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { f16vec4 features[]; };

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int pixels = width * height;
    if (idx >= pixels) return;

    int in_g = in_channels / 4;   // input channel groups
    int out_g = out_channels / 4;  // output channel groups

    // Load all input channel groups for this pixel (16 vec4s for 64ch)
    f16vec4 inp[16];
    for (int g = 0; g < in_g; g++) {
        inp[g] = features[input_offset + g * pixels + idx];
    }

    // Compute output in groups of 4
    for (int og = 0; og < out_g; og++) {
        vec4 sum = vec4(0.0);  // 4 output channels, accumulated in fp32

        // Each output channel dot-products with all input channels
        // Weight layout: [out_g][in_g][4][4] — for each out group × in group, 4×4 matrix
        int w_base = weight_offset + (og * in_g) * 16;

        for (int ig = 0; ig < in_g; ig++) {
            // Load 4×4 weight sub-matrix for this (out_group, in_group) pair
            // w[oc_local][ic_local] where oc_local, ic_local ∈ [0,3]
            int w_off = w_base + ig * 16;

            vec4 iv = vec4(inp[ig]);  // 4 input channels

            // For each of 4 output channels in this group:
            sum.x += dot(iv, vec4(
                float(weights[w_off + 0]), float(weights[w_off + 1]),
                float(weights[w_off + 2]), float(weights[w_off + 3])));
            sum.y += dot(iv, vec4(
                float(weights[w_off + 4]), float(weights[w_off + 5]),
                float(weights[w_off + 6]), float(weights[w_off + 7])));
            sum.z += dot(iv, vec4(
                float(weights[w_off + 8]), float(weights[w_off + 9]),
                float(weights[w_off + 10]), float(weights[w_off + 11])));
            sum.w += dot(iv, vec4(
                float(weights[w_off + 12]), float(weights[w_off + 13]),
                float(weights[w_off + 14]), float(weights[w_off + 15])));
        }

        // Add bias
        sum.x += float(weights[bias_offset + og * 4 + 0]);
        sum.y += float(weights[bias_offset + og * 4 + 1]);
        sum.z += float(weights[bias_offset + og * 4 + 2]);
        sum.w += float(weights[bias_offset + og * 4 + 3]);

        // Optional residual add
        if (residual_offset >= 0) {
            f16vec4 res = features[residual_offset + og * pixels + idx];
            sum += vec4(res);
        }

        // ReLU
        if (relu == 1) sum = max(sum, vec4(0.0));

        features[output_offset + og * pixels + idx] = f16vec4(sum);
    }
}
