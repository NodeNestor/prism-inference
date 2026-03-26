#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Flash Linear Attention — Fused Reduce + Q + FFN (2-dispatch approach)
//
// This shader does everything EXCEPT KV projection in a single dispatch:
//   Phase 1: Cooperatively compute S and z from φ(K), V in L2 cache
//   Phase 2: Q projection + attention output + FFN per token
//
// Every workgroup redundantly computes S (only 8KB — fits in shared memory).
// The φ(K) and V data (4MB) lives in L2 cache from the KV dispatch.
// 32 workgroups × ~71MB L2 reads = fast at 4 TB/s internal bandwidth.
//
// dim=128, FFN=512 variant. 4 heads × 32 head_dim.

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int n_tokens;       // 8160
    int dim;            // 128
    int n_heads;        // 4
    int head_dim;       // 32
    int w_q_offset;
    int b_q_offset;
    int w_o_offset;
    int b_o_offset;
    int ffn_w1_offset;  // [512, 128]
    int ffn_b1_offset;
    int ffn_w2_offset;  // [128, 512]
    int ffn_b2_offset;
    int input_offset;
    int output_offset;
    int k_offset;       // φ(K) in features buffer (from KV dispatch)
    int v_offset;       // V in features buffer
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

// S matrix and z vector in shared memory
shared float S[4][32][32];   // 16 KB fp32
shared float Z[4][32];       // 512 bytes fp32

void main() {
    int lid = int(gl_LocalInvocationID.x);   // 0..255
    int wg = int(gl_WorkGroupID.x);

    // ================================================================
    // PHASE 1: Compute S and z by reading φ(K), V from L2 cache
    //
    // 4096 S entries + 128 z entries = 4224 total
    // 256 threads → 16 S entries + ~0.5 z entries per thread
    //
    // For S[h][i][j] = Σ_t φ(K_t[h*32+i]) * V_t[h*32+j]
    // Assign 16 consecutive j values per thread within same (h, i):
    //   thread 0:   S[0][0][0..15]
    //   thread 1:   S[0][0][16..31]
    //   thread 2:   S[0][1][0..15]
    //   ...
    //   thread 255: S[3][31][16..31]
    // ================================================================

    int s_base = lid * 16;  // base index into flattened S[4096]
    int h = s_base / 1024;
    int local_idx = s_base % 1024;
    int i = local_idx / 32;
    int j_start = local_idx % 32;  // always 0 or 16

    float partial_s[16];
    for (int x = 0; x < 16; x++) partial_s[x] = 0.0;
    float partial_z = 0.0;

    int ki_addr = h * 32 + i;  // which K dimension to read
    int vj_base = h * 32 + j_start;  // base V dimension

    for (int t = 0; t < n_tokens; t++) {
        float phi_k = float(features[k_offset + t * dim + ki_addr]);

        for (int x = 0; x < 16; x++) {
            float v = float(features[v_offset + t * dim + vj_base + x]);
            partial_s[x] += phi_k * v;
        }
        partial_z += phi_k;
    }

    // Write S entries to shared memory
    for (int x = 0; x < 16; x++) {
        S[h][i][j_start + x] = partial_s[x];
    }

    // z: 128 entries. Threads with j_start == 0 contribute z (128 of 256 threads)
    if (j_start == 0) {
        Z[h][i] = partial_z;
    }

    barrier();

    // ================================================================
    // PHASE 2: Q projection + linear attention output + FFN
    // ================================================================

    // Each workgroup handles a chunk of tokens
    int tokens_per_wg = (n_tokens + int(gl_NumWorkGroups.x) - 1) / int(gl_NumWorkGroups.x);
    int t_start = wg * tokens_per_wg;
    int t_end = min(t_start + tokens_per_wg, n_tokens);

    for (int t = t_start + lid; t < t_end; t += 256) {
        // Load input
        coopvecNV<float16_t, 128> inp;
        coopVecLoadNV(inp, features, (input_offset + t * dim) * 2);

        // Q projection (tensor cores)
        coopvecNV<float16_t, 128> Q;
        coopVecMatMulAddNV(Q, inp, gl_ComponentTypeFloat16NV,
            weights, w_q_offset * 2, gl_ComponentTypeFloat16NV,
            weights, b_q_offset * 2, gl_ComponentTypeFloat16NV,
            128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

        // Apply φ(Q) = ELU+1
        float phi_q[128];
        for (int d = 0; d < 128; d++) {
            float x = float(Q[d]);
            phi_q[d] = x >= 0.0 ? x + 1.0 : exp(x);
        }

        // Linear attention: φ(Q) @ S per head, from shared memory (fast!)
        coopvecNV<float16_t, 128> attn_out;
        for (int hh = 0; hh < 4; hh++) {
            int hoff = hh * 32;

            float out_h[32];
            for (int j = 0; j < 32; j++) {
                float sum = 0.0;
                for (int ii = 0; ii < 32; ii++) {
                    sum += phi_q[hoff + ii] * S[hh][ii][j];
                }
                out_h[j] = sum;
            }

            float norm = 0.0;
            for (int ii = 0; ii < 32; ii++) {
                norm += phi_q[hoff + ii] * Z[hh][ii];
            }
            norm = max(norm, 1e-6);

            for (int j = 0; j < 32; j++) {
                attn_out[hoff + j] = float16_t(out_h[j] / norm);
            }
        }

        // Output projection (tensor cores)
        coopvecNV<float16_t, 128> proj_out;
        coopVecMatMulAddNV(proj_out, attn_out, gl_ComponentTypeFloat16NV,
            weights, w_o_offset * 2, gl_ComponentTypeFloat16NV,
            weights, b_o_offset * 2, gl_ComponentTypeFloat16NV,
            128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

        // Residual 1
        for (int d = 0; d < 128; d++) proj_out[d] = proj_out[d] + inp[d];

        // FFN: 128 → 512 → 128 (tensor cores)
        coopvecNV<float16_t, 512> ffn_hidden;
        coopVecMatMulAddNV(ffn_hidden, proj_out, gl_ComponentTypeFloat16NV,
            weights, ffn_w1_offset * 2, gl_ComponentTypeFloat16NV,
            weights, ffn_b1_offset * 2, gl_ComponentTypeFloat16NV,
            512, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

        for (int d = 0; d < 512; d++) {
            float x = float(ffn_hidden[d]);
            ffn_hidden[d] = float16_t(0.5 * x * (1.0 + tanh(0.7978846 * (x + 0.044715 * x*x*x))));
        }

        coopvecNV<float16_t, 128> ffn_out;
        coopVecMatMulAddNV(ffn_out, ffn_hidden, gl_ComponentTypeFloat16NV,
            weights, ffn_w2_offset * 2, gl_ComponentTypeFloat16NV,
            weights, ffn_b2_offset * 2, gl_ComponentTypeFloat16NV,
            128, 512, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 512*2);

        for (int d = 0; d < 128; d++) ffn_out[d] = ffn_out[d] + proj_out[d];

        coopVecStoreNV(ffn_out, features, (output_offset + t * dim) * 2);
    }
}
