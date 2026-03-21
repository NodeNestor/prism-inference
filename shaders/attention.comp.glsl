#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Self-Attention using cooperative vectors (tensor cores)
// Input: [N, dim] tokens in HWC-like format (contiguous per token)
// Steps: Q,K,V projections -> attention -> output projection
// All matrix multiplies on tensor cores.

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int n_tokens;      // N (e.g., 8160 = 68*120)
    int dim;           // embedding dim (128)
    int n_heads;       // attention heads (4)
    int head_dim;      // dim / n_heads (32)
    int qkv_w_offset;  // weight for QKV projection [3*dim, dim] fp16 elements
    int qkv_b_offset;  // bias [3*dim] fp16 elements
    int out_w_offset;  // output projection weight [dim, dim]
    int out_b_offset;  // output projection bias [dim]
    int input_offset;  // token input [N, dim] fp16 elements
    int output_offset; // token output [N, dim] fp16 elements
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

// Shared memory for K and V of current head's chunk
// We process attention in chunks to fit in shared memory
shared float scores[256]; // attention scores for one query

void main() {
    int token_idx = int(gl_GlobalInvocationID.x);
    if (token_idx >= n_tokens) return;

    // === QKV Projection: [dim] -> [3*dim] using tensor cores ===
    // Load input token
    coopvecNV<float16_t, 128> inp;
    coopVecLoadNV(inp, features, (input_offset + token_idx * dim) * 2);

    // QKV = W_qkv @ inp + b_qkv  (produces [3*dim] = [384] for dim=128)
    // Split into Q[128], K[128], V[128]
    // Since coopVecMatMulAddNV needs compile-time M, do 3 separate projections

    // Q projection
    coopvecNV<float16_t, 128> Q;
    coopVecMatMulAddNV(Q, inp, gl_ComponentTypeFloat16NV,
        weights, qkv_w_offset * 2, gl_ComponentTypeFloat16NV,
        weights, qkv_b_offset * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128 * 2);

    // K projection (weight offset += dim*dim = 128*128 = 16384)
    coopvecNV<float16_t, 128> K;
    coopVecMatMulAddNV(K, inp, gl_ComponentTypeFloat16NV,
        weights, (qkv_w_offset + 128*128) * 2, gl_ComponentTypeFloat16NV,
        weights, (qkv_b_offset + 128) * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128 * 2);

    // V projection
    coopvecNV<float16_t, 128> V;
    coopVecMatMulAddNV(V, inp, gl_ComponentTypeFloat16NV,
        weights, (qkv_w_offset + 2*128*128) * 2, gl_ComponentTypeFloat16NV,
        weights, (qkv_b_offset + 2*128) * 2, gl_ComponentTypeFloat16NV,
        128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128 * 2);

    // Store Q, K, V to feature buffer for attention computation
    // Layout: after input tokens, store Q[N,dim], K[N,dim], V[N,dim]
    int qkv_base = output_offset + n_tokens * dim; // temp storage after output
    coopVecStoreNV(Q, features, (qkv_base + token_idx * dim) * 2);
    coopVecStoreNV(K, features, (qkv_base + (n_tokens + token_idx) * dim) * 2);
    coopVecStoreNV(V, features, (qkv_base + (2 * n_tokens + token_idx) * dim) * 2);
}
