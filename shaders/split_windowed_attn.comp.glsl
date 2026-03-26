#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// Split Pipeline Pass 2: Windowed Attention Only
// 64 threads/WG = one 8x8 window. Reads Q/K/V from global, writes attn_out.
// No coopVec matmuls — pure scalar attention. Lower register pressure.

layout(local_size_x = 64) in;

layout(push_constant) uniform PC {
    int n_tokens, dim, n_heads, head_dim;
    int spatial_w, spatial_h, window_size;
    int q_offset, k_offset, v_offset, output_offset;
};

layout(set = 0, binding = 1) buffer F { float16_t features[]; };

shared float16_t shared_k[64 * 128];
shared float16_t shared_v[64 * 128];

void main() {
    int window_idx = int(gl_WorkGroupID.x);
    int local_idx = int(gl_LocalInvocationID.x);

    int windows_x = (spatial_w + window_size - 1) / window_size;
    int win_y = window_idx / windows_x;
    int win_x = window_idx % windows_x;
    int tok_y = local_idx / window_size;
    int tok_x = local_idx % window_size;
    int global_y = win_y * window_size + tok_y;
    int global_x = win_x * window_size + tok_x;
    bool valid = (global_y < spatial_h && global_x < spatial_w);
    int global_token = valid ? (global_y * spatial_w + global_x) : 0;

    // Load Q for this token
    float Q[128];
    if (valid) {
        for (int d = 0; d < 128; d++)
            Q[d] = float(features[q_offset + global_token * dim + d]);
    }

    // Load K and V into shared memory
    if (valid) {
        for (int d = 0; d < 128; d++) {
            shared_k[local_idx * 128 + d] = features[k_offset + global_token * dim + d];
            shared_v[local_idx * 128 + d] = features[v_offset + global_token * dim + d];
        }
    } else {
        for (int d = 0; d < 128; d++) {
            shared_k[local_idx * 128 + d] = float16_t(0.0);
            shared_v[local_idx * 128 + d] = float16_t(0.0);
        }
    }
    barrier();

    // Windowed attention per head
    float attn_out[128];
    for (int d = 0; d < 128; d++) attn_out[d] = 0.0;

    for (int h = 0; h < n_heads; h++) {
        int ch_off = h * head_dim;
        float scale = 1.0 / sqrt(float(head_dim));

        float scores[64];
        float max_s = -1e9;
        int wt = min(64, n_tokens);

        for (int j = 0; j < wt; j++) {
            float s = 0.0;
            for (int d = 0; d < head_dim; d++)
                s += Q[ch_off + d] * float(shared_k[j * 128 + ch_off + d]);
            scores[j] = s * scale;
            if (scores[j] > max_s) max_s = scores[j];
        }

        float sum_exp = 0.0;
        float result[32];
        for (int d = 0; d < head_dim; d++) result[d] = 0.0;

        for (int j = 0; j < wt; j++) {
            float w = exp(scores[j] - max_s);
            sum_exp += w;
            for (int d = 0; d < head_dim; d++)
                result[d] += w * float(shared_v[j * 128 + ch_off + d]);
        }

        float inv = 1.0 / max(sum_exp, 1e-6);
        for (int d = 0; d < head_dim; d++)
            attn_out[ch_off + d] = result[d] * inv;
    }

    // Write attention output
    if (valid) {
        for (int d = 0; d < 128; d++)
            features[output_offset + global_token * dim + d] = float16_t(attn_out[d]);
    }
}
