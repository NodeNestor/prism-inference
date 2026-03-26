#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_KHR_shader_subgroup_arithmetic : enable

// Linear Attention — Phase 2: Global KV State Reduction
//
// Computes per-head state matrices:
//   S_h[i][j] = Σ_t φ(K_h[t][i]) * V_h[t][j]   (head_dim × head_dim per head)
//   z_h[i]    = Σ_t φ(K_h[t][i])                  (head_dim per head)
//
// One workgroup per row of S (n_heads * head_dim = 128 workgroups).
// Each workgroup reduces over all tokens via subgroup + shared memory.
//
// S total: 4 heads × 32 × 32 = 4096 fp16 = 8 KB (fits in L2 cache!)
// z total: 4 × 32 = 128 fp16 = 256 bytes
//
// This is the magic of linear attention: we compress 8160 tokens into
// a tiny 8KB state matrix, then every query just reads from cache.

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int n_tokens;       // total tokens (8160)
    int dim;            // 128
    int n_heads;        // 4
    int head_dim;       // 32
    int k_offset;       // φ(K) data [n_tokens, dim]
    int v_offset;       // V data [n_tokens, dim]
    int s_offset;       // output S [n_heads, head_dim, head_dim]
    int z_offset;       // output z [n_heads, head_dim]
};

layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

// Shared memory for cross-subgroup reduction
// 8 subgroups (256 threads / 32 warp), each with head_dim partial sums
shared float shared_s[8][32];
shared float shared_z[8];

void main() {
    int wg_id = int(gl_WorkGroupID.x);     // 0..127
    int lid = int(gl_LocalInvocationID.x);  // 0..255
    int sg_id = lid / 32;                   // subgroup index (0..7)
    int sg_lane = lid % 32;                 // lane within subgroup

    // Which head and row this workgroup handles
    int head = wg_id / head_dim;            // 0..3
    int row = wg_id % head_dim;             // 0..31

    int head_start = head * head_dim;       // channel offset for this head

    // Each thread accumulates over its chunk of tokens
    float partial_s[32];  // partial sum for S[head][row][0..31]
    for (int j = 0; j < 32; j++) partial_s[j] = 0.0;
    float partial_z = 0.0;

    // Divide tokens across 256 threads
    int tokens_per_thread = (n_tokens + 255) / 256;
    int t_start = lid * tokens_per_thread;
    int t_end = min(t_start + tokens_per_thread, n_tokens);

    for (int t = t_start; t < t_end; t++) {
        // Load φ(K_h[t][row]) — single scalar
        float phi_k = float(features[k_offset + t * dim + head_start + row]);

        // Accumulate outer product contribution and z
        for (int j = 0; j < 32; j++) {
            float v_j = float(features[v_offset + t * dim + head_start + j]);
            partial_s[j] += phi_k * v_j;
        }
        partial_z += phi_k;
    }

    // Subgroup reduction (within 32-thread warp)
    for (int j = 0; j < 32; j++) {
        partial_s[j] = subgroupAdd(partial_s[j]);
    }
    partial_z = subgroupAdd(partial_z);

    // Write subgroup result to shared memory (only lane 0 of each subgroup)
    if (sg_lane == 0) {
        for (int j = 0; j < 32; j++) {
            shared_s[sg_id][j] = partial_s[j];
        }
        shared_z[sg_id] = partial_z;
    }
    barrier();

    // Final reduction: thread 0 sums across 8 subgroups
    if (lid == 0) {
        float final_s[32];
        float final_z = 0.0;
        for (int j = 0; j < 32; j++) final_s[j] = 0.0;

        int n_subgroups = (256 + 31) / 32;  // 8
        for (int sg = 0; sg < n_subgroups; sg++) {
            for (int j = 0; j < 32; j++) {
                final_s[j] += shared_s[sg][j];
            }
            final_z += shared_z[sg];
        }

        // Write to global buffer
        int s_row_offset = s_offset + head * head_dim * head_dim + row * head_dim;
        for (int j = 0; j < 32; j++) {
            features[s_row_offset + j] = float16_t(final_s[j]);
        }
        features[z_offset + head * head_dim + row] = float16_t(final_z);
    }
}
