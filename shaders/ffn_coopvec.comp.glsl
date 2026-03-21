#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Feed-Forward Network: Linear(dim->dim*4) -> GELU -> Linear(dim*4->dim)
// Using cooperative vectors for both linear layers (tensor cores)
// Each thread processes one token

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int n_tokens;
    int dim;          // 128
    int hidden_dim;   // 512 (dim * 4)
    int w1_offset;    // [hidden, dim] weight fp16 elements
    int b1_offset;    // [hidden] bias
    int w2_offset;    // [dim, hidden] weight
    int b2_offset;    // [dim] bias
    int input_offset; // [N, dim] tokens
    int output_offset;
    int residual;     // 1 = add input as residual
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int idx = int(gl_GlobalInvocationID.x);
    if (idx >= n_tokens) return;

    // Load input token
    coopvecNV<float16_t, 128> inp;
    coopVecLoadNV(inp, features, (input_offset + idx * dim) * 2);

    // Linear 1: dim -> hidden_dim (128 -> 512)
    coopvecNV<float16_t, 512> hidden;
    coopVecMatMulAddNV(hidden, inp, gl_ComponentTypeFloat16NV,
        weights, w1_offset * 2, gl_ComponentTypeFloat16NV,
        weights, b1_offset * 2, gl_ComponentTypeFloat16NV,
        512, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128 * 2);

    // GELU activation (approximate)
    for (int i = 0; i < 512; i++) {
        float x = float(hidden[i]);
        // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float x3 = x * x * x;
        hidden[i] = float16_t(0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x3))));
    }

    // Linear 2: hidden_dim -> dim (512 -> 128)
    coopvecNV<float16_t, 128> outp;
    coopVecMatMulAddNV(outp, hidden, gl_ComponentTypeFloat16NV,
        weights, w2_offset * 2, gl_ComponentTypeFloat16NV,
        weights, b2_offset * 2, gl_ComponentTypeFloat16NV,
        128, 512, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 512 * 2);

    // Residual add
    if (residual == 1) {
        for (int i = 0; i < 128; i++) {
            outp[i] = outp[i] + inp[i];
        }
    }

    coopVecStoreNV(outp, features, (output_offset + idx * dim) * 2);
}
