#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// FUSED DW3x3 + PW1x1 for 128-channel layers (decoder bottleneck)
// Same structure as fused_dw_pw.comp.glsl but with 128-channel coopvecs

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int channels;       // 128
    int width;
    int height;
    int dw_weight_offset;
    int pw_weight_offset;
    int pw_bias_offset;
    int input_offset;
    int output_offset;
    int relu;
    int residual_offset;
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int pixels = width * height;
    if (idx >= pixels) return;

    int oy = idx / width;
    int ox = idx % width;

    // DW 3x3 -> result in registers
    coopvecNV<float16_t, 128> dw_out;
    for (int ch = 0; ch < channels; ch++) {
        float sum = 0.0;
        int w_base = dw_weight_offset + ch * 9;
        for (int ky = -1; ky <= 1; ky++) {
            int iy = oy + ky;
            if (iy < 0 || iy >= height) continue;
            for (int kx = -1; kx <= 1; kx++) {
                int ix = ox + kx;
                if (ix < 0 || ix >= width) continue;
                float val = float(features[input_offset + ch * pixels + iy * width + ix]);
                sum += val * float(weights[w_base + (ky+1) * 3 + (kx+1)]);
            }
        }
        dw_out[ch] = float16_t(sum);
    }

    // PW 1x1 via tensor cores
    coopvecNV<float16_t, 128> pw_out;
    coopVecMatMulAddNV(
        pw_out, dw_out,
        gl_ComponentTypeFloat16NV,
        weights, pw_weight_offset * 2,
        gl_ComponentTypeFloat16NV,
        weights, pw_bias_offset * 2,
        gl_ComponentTypeFloat16NV,
        128, 128,
        gl_CooperativeVectorMatrixLayoutRowMajorNV,
        false, 128 * 2
    );

    // Residual + ReLU + store CHW
    for (int ch = 0; ch < channels; ch++) {
        float v = float(pw_out[ch]);
        if (residual_offset >= 0) {
            v += float(features[residual_offset + ch * pixels + idx]);
        }
        if (relu == 1 && v < 0.0) v = 0.0;
        features[output_offset + ch * pixels + idx] = float16_t(v);
    }
}
