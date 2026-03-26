#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// MoE Unpack: gather expert outputs back to original token order

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int n_tokens, dim, n_experts;
    int expert_out_offset;      // packed expert output buffers
    int expert_buf_stride;      // n_tokens * dim per expert
    int output_offset;          // destination in original order
    int pack_idx_offset;        // original token indices per packed position
    int expert_count_offset;    // token counts per expert
};

layout(set = 0, binding = 1) buffer F { float16_t features[]; };
layout(set = 0, binding = 2) readonly buffer Aux { int aux[]; };

void main() {
    // Each thread handles one packed token across all experts
    int gid = int(gl_GlobalInvocationID.x);
    if (gid >= n_tokens) return;

    // Find which expert and position this global id maps to
    int running = 0;
    for (int e = 0; e < n_experts; e++) {
        int count = aux[expert_count_offset + e];
        if (gid < running + count) {
            int pos = gid - running;
            int orig_token = aux[pack_idx_offset + e * n_tokens + pos];
            int src = expert_out_offset + e * expert_buf_stride + pos * dim;
            int dst = output_offset + orig_token * dim;
            for (int i = 0; i < 256; i++)
                features[dst + i] = features[src + i];
            return;
        }
        running += count;
    }
}
