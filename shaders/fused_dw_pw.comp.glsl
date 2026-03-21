#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// FUSED Depthwise 3x3 Conv + Pointwise 1x1 Conv
// The DW output stays in REGISTERS — zero intermediate memory traffic!
// This is the key optimization that DLSS/FSR use.
//
// Each thread processes ONE pixel:
//   1. Gather 3x3 neighborhood for all 64 channels (DW conv) → 64 values in registers
//   2. Feed directly to cooperative vector matmul (PW conv) → tensor cores
//   3. Write final output only
//
// CHW format for input/output (spatial locality for DW conv gather)

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int channels;       // both in and out channels (same for DSC)
    int width;
    int height;
    int dw_weight_offset;   // DW 3x3 kernel weights (fp16 elements)
    int pw_weight_offset;   // PW 1x1 weight matrix (fp16 elements)
    int pw_bias_offset;     // PW bias (fp16 elements)
    int input_offset;       // CHW input (fp16 elements)
    int output_offset;      // CHW output (fp16 elements)
    int relu;
    int residual_offset;    // -1 = none, else CHW offset for residual add
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    int pixels = width * height;
    if (idx >= pixels) return;

    int oy = idx / width;
    int ox = idx % width;

    // === STEP 1: Depthwise 3x3 Conv (result stays in registers!) ===
    coopvecNV<float16_t, 64> dw_out;

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

    // === STEP 2: Pointwise 1x1 Conv using tensor cores ===
    // dw_out is already in a cooperative vector — feed directly to matmul!
    coopvecNV<float16_t, 64> pw_out;
    coopVecMatMulAddNV(
        pw_out,
        dw_out,
        gl_ComponentTypeFloat16NV,
        weights,
        pw_weight_offset * 2,
        gl_ComponentTypeFloat16NV,
        weights,
        pw_bias_offset * 2,
        gl_ComponentTypeFloat16NV,
        64,   // M = out_channels
        64,   // K = in_channels
        gl_CooperativeVectorMatrixLayoutRowMajorNV,
        false,
        64 * 2
    );

    // === STEP 3: Optional residual add + ReLU + write output (CHW) ===
    for (int ch = 0; ch < channels; ch++) {
        float v = float(pw_out[ch]);
        if (residual_offset >= 0) {
            v += float(features[residual_offset + ch * pixels + idx]);
        }
        if (relu == 1 && v < 0.0) v = 0.0;
        features[output_offset + ch * pixels + idx] = float16_t(v);
    }
}
