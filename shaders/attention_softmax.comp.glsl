#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Attention: softmax(Q @ K^T / sqrt(d)) @ V for ONE query token
// Each thread processes one query token, computes attention over ALL keys
// For 8K tokens this is 8K dot products + softmax + 8K weighted sums
// Split by heads: each head has head_dim channels

layout(local_size_x = 128) in;

layout(push_constant) uniform PC {
    int n_tokens;
    int dim;        // 128
    int n_heads;    // 4
    int head_dim;   // 32
    int q_offset;   // Q[N, dim] fp16 elements
    int k_offset;   // K[N, dim]
    int v_offset;   // V[N, dim]
    int out_offset; // attention output [N, dim]
};

layout(set = 0, binding = 1) buffer Features { float16_t features[]; };

void main() {
    int query_idx = int(gl_GlobalInvocationID.x);
    if (query_idx >= n_tokens) return;

    float scale = 1.0 / sqrt(float(head_dim));

    // Process each head independently
    for (int h = 0; h < n_heads; h++) {
        int ch_start = h * head_dim;

        // Load this query's Q vector for this head
        float q[32];
        for (int d = 0; d < head_dim; d++) {
            q[d] = float(features[q_offset + query_idx * dim + ch_start + d]);
        }

        // Compute attention scores: dot(Q, K_j) for all j
        // Also find max for numerical stability
        float max_score = -1e9;
        for (int j = 0; j < n_tokens; j++) {
            float score = 0.0;
            for (int d = 0; d < head_dim; d++) {
                score += q[d] * float(features[k_offset + j * dim + ch_start + d]);
            }
            score *= scale;
            if (score > max_score) max_score = score;
        }

        // Softmax: exp(score - max) / sum(exp)
        float sum_exp = 0.0;
        // Compute weighted sum of V in same pass
        float result[32];
        for (int d = 0; d < head_dim; d++) result[d] = 0.0;

        for (int j = 0; j < n_tokens; j++) {
            // Recompute score (avoid storing 8K floats)
            float score = 0.0;
            for (int d = 0; d < head_dim; d++) {
                score += q[d] * float(features[k_offset + j * dim + ch_start + d]);
            }
            float w = exp(score * scale - max_score);
            sum_exp += w;

            // Accumulate weighted V
            for (int d = 0; d < head_dim; d++) {
                result[d] += w * float(features[v_offset + j * dim + ch_start + d]);
            }
        }

        // Normalize and write
        float inv_sum = 1.0 / sum_exp;
        for (int d = 0; d < head_dim; d++) {
            features[out_offset + query_idx * dim + ch_start + d] =
                float16_t(result[d] * inv_sum);
        }
    }
}
