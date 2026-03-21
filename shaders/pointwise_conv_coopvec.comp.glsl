#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// 1x1 Pointwise Convolution using COOPERATIVE VECTORS (tensor cores)
// Features in HWC format: [H*W][channels] — contiguous channels per pixel
// One thread per pixel. coopVecMatMulAddNV runs on tensor cores.

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int in_channels;
    int out_channels;
    int width;
    int height;
    int weight_offset;   // fp16 element offset into weight buffer
    int bias_offset;     // fp16 element offset
    int input_offset;    // fp16 element offset into feature buffer (HWC)
    int output_offset;   // fp16 element offset (HWC)
    int relu;
    int residual_offset; // -1 = no residual, else fp16 element offset (HWC)
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int pixels = width * height;
    if (idx >= pixels) return;

    // Load input vector — contiguous in HWC format!
    coopvecNV<float16_t, 64> inp;
    coopVecLoadNV(inp, features, (input_offset + idx * in_channels) * 2);

    // Tensor core matrix-vector multiply + bias
    coopvecNV<float16_t, 64> outp;
    coopVecMatMulAddNV(
        outp,
        inp,
        gl_ComponentTypeFloat16NV,
        weights,
        weight_offset * 2,
        gl_ComponentTypeFloat16NV,
        weights,
        bias_offset * 2,
        gl_ComponentTypeFloat16NV,
        64,   // M = out_channels (compile-time constant required)
        64,   // K = in_channels
        gl_CooperativeVectorMatrixLayoutRowMajorNV,
        false,
        64 * 2  // stride = K * sizeof(fp16)
    );

    // Residual + ReLU + store (HWC)
    int out_base = output_offset + idx * out_channels;
    if (residual_offset >= 0) {
        int res_base = residual_offset + idx * out_channels;
        for (int c = 0; c < out_channels; c++) {
            float16_t v = outp[c] + features[res_base + c];
            if (relu == 1 && v < float16_t(0.0)) v = float16_t(0.0);
            features[out_base + c] = v;
        }
    } else {
        if (relu == 1) {
            coopvecNV<float16_t, 64> zero = coopvecNV<float16_t, 64>(float16_t(0.0));
            outp = max(outp, zero);
        }
        coopVecStoreNV(outp, features, out_base * 2);
    }
}
