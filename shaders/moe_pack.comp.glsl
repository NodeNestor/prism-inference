#version 460
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

// MoE Pack: scatter tokens into per-expert buffers
// Each token is copied to its assigned expert's buffer using atomic counters.

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    int n_tokens, dim, n_experts;
    int input_offset;           // source features [n_tokens, dim] (CHW or token-major)
    int expert_buf_offset;      // dest: packed expert buffers [n_experts * n_tokens * dim] (worst case)
    int expert_buf_stride;      // n_tokens * dim (max tokens per expert buffer)
    int assignment_offset;      // expert id per token
    int pack_idx_offset;        // output: original token index per packed position
    int expert_count_offset;    // atomic counters per expert (must be zeroed before dispatch)
};

layout(set = 0, binding = 1) buffer F { float16_t features[]; };
layout(set = 0, binding = 2) buffer Aux { int aux[]; };

void main() {
    int tid = int(gl_GlobalInvocationID.x);
    if (tid >= n_tokens) return;

    int expert = aux[assignment_offset + tid];

    // Atomically get position in expert's buffer
    int pos = atomicAdd(aux[expert_count_offset + expert], 1);

    // Copy token to expert's packed buffer
    int src = input_offset + tid * dim;
    int dst = expert_buf_offset + expert * expert_buf_stride + pos * dim;
    for (int i = 0; i < 256; i++)
        features[dst + i] = features[src + i];

    // Store original index for unpacking
    aux[pack_idx_offset + expert * n_tokens + pos] = tid;
}
