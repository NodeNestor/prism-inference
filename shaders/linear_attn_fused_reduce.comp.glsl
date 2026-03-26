#version 460
#extension GL_NV_cooperative_vector : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_KHR_shader_subgroup_arithmetic : enable

// Flash Linear Attention — Fused KV Projection + S Reduction
//
// Fuses KV projection and state matrix reduction into ONE dispatch.
// No intermediate K,V buffers in global memory — everything stays in
// registers and shared memory.
//
// Each workgroup handles one row of the S matrix (128 workgroups total).
// For each token, projects K and V on-the-fly using tensor cores,
// extracts φ(K[row]) and V_head, accumulates into partial S.
//
// Tensor cores do the heavy lifting (128→128 projection), we just
// extract 1 scalar from K and 32 values from V. "Wasteful" but
// the tensor core ops are so fast it doesn't matter — and we avoid
// ~4MB of global memory round-trips per block.
//
// Combined with the Q+FFN dispatch, this gives 2 dispatches per block
// instead of 3, cutting overhead by ~33%.

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int n_tokens;       // 8160
    int dim;            // 128
    int n_heads;        // 4
    int head_dim;       // 32
    int w_k_offset;     // K projection weights [dim, dim]
    int b_k_offset;     // K bias [dim]
    int w_v_offset;     // V projection weights [dim, dim]
    int b_v_offset;     // V bias [dim]
    int input_offset;   // input features
    int s_offset;       // output S [n_heads, head_dim, head_dim]
    int z_offset;       // output z [n_heads * head_dim]
};

layout(set = 0, binding = 0) readonly buffer Weights { float16_t weights[]; };
layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

// Shared memory for cross-subgroup reduction
shared float shared_s[8][32];  // 8 subgroups × 32 head_dim = 1KB
shared float shared_z[8];      // 8 subgroups

void main() {
    int wg_id = int(gl_WorkGroupID.x);     // 0..127: which row of S
    int lid = int(gl_LocalInvocationID.x);  // 0..255
    int sg_id = lid / 32;
    int sg_lane = lid % 32;

    int head = wg_id / head_dim;            // 0..3
    int row = wg_id % head_dim;             // 0..31
    int head_start = head * head_dim;

    // Thread-local accumulators
    float partial_s[32];
    for (int j = 0; j < 32; j++) partial_s[j] = 0.0;
    float partial_z = 0.0;

    // Each thread processes its chunk of tokens
    int tpt = (n_tokens + 255) / 256;
    int t_start = lid * tpt;
    int t_end = min(t_start + tpt, n_tokens);

    for (int t = t_start; t < t_end; t++) {
        // Load input token (from feature buffer)
        coopvecNV<float16_t, 128> inp;
        coopVecLoadNV(inp, features, (input_offset + t * dim) * 2);

        // Project K using tensor cores (full 128→128, extract row element)
        coopvecNV<float16_t, 128> K_full;
        coopVecMatMulAddNV(K_full, inp, gl_ComponentTypeFloat16NV,
            weights, w_k_offset * 2, gl_ComponentTypeFloat16NV,
            weights, b_k_offset * 2, gl_ComponentTypeFloat16NV,
            128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

        // Apply φ(K) = ELU+1, extract only K[head_start + row]
        float k_val = float(K_full[head_start + row]);
        float phi_k = k_val >= 0.0 ? k_val + 1.0 : exp(k_val);

        // Project V using tensor cores (full 128→128, extract head slice)
        coopvecNV<float16_t, 128> V_full;
        coopVecMatMulAddNV(V_full, inp, gl_ComponentTypeFloat16NV,
            weights, w_v_offset * 2, gl_ComponentTypeFloat16NV,
            weights, b_v_offset * 2, gl_ComponentTypeFloat16NV,
            128, 128, gl_CooperativeVectorMatrixLayoutRowMajorNV, false, 128*2);

        // Accumulate outer product contribution: S[row][j] += φ(K[row]) * V[j]
        for (int j = 0; j < 32; j++) {
            partial_s[j] += phi_k * float(V_full[head_start + j]);
        }
        partial_z += phi_k;
    }

    // === Subgroup reduction ===
    for (int j = 0; j < 32; j++) {
        partial_s[j] = subgroupAdd(partial_s[j]);
    }
    partial_z = subgroupAdd(partial_z);

    // Write subgroup results to shared memory
    if (sg_lane == 0) {
        for (int j = 0; j < 32; j++) shared_s[sg_id][j] = partial_s[j];
        shared_z[sg_id] = partial_z;
    }
    barrier();

    // Final reduction across subgroups (thread 0 only)
    if (lid == 0) {
        float final_s[32];
        float final_z = 0.0;
        for (int j = 0; j < 32; j++) final_s[j] = 0.0;

        for (int sg = 0; sg < 8; sg++) {
            for (int j = 0; j < 32; j++) final_s[j] += shared_s[sg][j];
            final_z += shared_z[sg];
        }

        // Write to global S and z
        int s_row_off = s_offset + head * head_dim * head_dim + row * head_dim;
        for (int j = 0; j < 32; j++) {
            features[s_row_off + j] = float16_t(final_s[j]);
        }
        features[z_offset + head_start + row] = float16_t(final_z);
    }
}
