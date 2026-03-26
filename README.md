# Prism — Neural Rendering Engine

> Real-time neural style transfer for games. **139M parameter MoE model at 66 FPS** on RTX 5060 Ti, 540p to 1080p.

Prism intercepts G-buffer data (color, depth, motion vectors) through [OptiScaler](https://github.com/cdozdil/OptiScaler) and runs a neural network for **style transfer** — transforming game graphics toward photorealism using Vulkan compute with NVIDIA cooperative vectors (tensor cores).

## Performance (RTX 5060 Ti, 540p -> 1080p)

| Config | Total Params | Active Params | End-to-End | FPS |
|--------|-------------|---------------|------------|-----|
| Baseline (dim=128, 4 blocks) | 2.5M | 2.5M | 5.6ms | 180 |
| Standard (dim=256, 12 blocks) | 9M | 9M | 15.3ms | 65 |
| MoE 16exp (dim=256, 16 blocks) | 37M | 6M | 13.4ms | 73 |
| MoE 32exp (dim=256, 22 blocks) | 98M | 8M | 18.0ms | 56 |
| **MoE 64exp (dim=256, 16 blocks)** | **139M** | **6M** | **15.1ms** | **66** |

All numbers are real GPU timestamps from Vulkan queries, end-to-end verified (encoder + transformer + decoder + output).

## Architecture

```
Game renders at 540p (G-buffer: color + depth + motion vectors)
  -> Conv Encoder (split coopvec): 540p -> 270p -> 135p -> 68p     [1.3ms]
  -> Projection: 128-dim -> 256-dim                                 [0.05ms]
  -> Transformer MoE blocks at 68p (windowed attention + expert FFN) [variable]
  -> Projection: 256-dim -> 128-dim                                 [0.05ms]
  -> Fused Decoder with skip connections: 68p -> 135p -> 270p -> 540p [0.6ms]
  -> PixelShuffle: 540p -> 1080p                                    [0.2ms]
```

### Mixture of Experts (MoE)

Each transformer block has a router that assigns each token (pixel region) to one of N expert FFN networks. Different experts specialize in different visual features — edges, textures, lighting, skin, foliage, sky, etc.

```
Token -> Attention (shared, 8 heads windowed) -> Router (256->N matmul, picks 1 expert)
     -> Expert FFN (256->256->256, only for assigned tokens)
     -> Residual -> next block
```

Adding experts is nearly free: going from 16 to 64 experts at 16 blocks costs only 1.7ms because each expert dispatch skips non-assigned tokens via early return (~94-98% of threads skip per expert).

### Key Optimizations

1. **Split cooperative vectors**: `coopvec<512>` causes 16x register pressure cliff. All operations split to `coopvec<256>` max. ([commit](../../commit/122afe7))
2. **Split encoder convolutions**: im2col at 128ch uses `coopvec<1152>` — split into 4 groups of `coopvec<288>`. 4x encoder speedup. ([commit](../../commit/da12f29))
3. **Fused decoder**: NN-upsample + concat + PW conv in single dispatch. 2.2x decoder speedup. ([commit](../../commit/035c0a9))
4. **MoE with early-skip routing**: `coopVecMatMulAddNV` allows per-thread divergence. Non-assigned tokens return early, making expert count nearly free. ([commit](../../commit/429044e))
5. **dim=256 with projection adapters**: wider transformer dim better utilizes tensor cores. 128->256 and 256->128 projections at bottleneck resolution are negligible cost. ([commit](../../commit/f793abf))

### What We Learned About Cooperative Vectors

- `coopvecNV<float16_t, N>` has a **hard performance cliff at N>256** due to register pressure destroying GPU occupancy
- Sequential data-dependent matmul chains scale **superlinearly** (12 chained calls = 92x slower than 12x, not 12x) because they can't pipeline
- Independent matmul calls (MoE experts) scale **linearly** — different experts reading different weights can overlap
- `coopVecMatMulAddNV` is per-thread, NOT a collective operation — warp divergence (early return) is allowed and efficient
- Fusing multiple transformer blocks into a single dispatch crashes the GPU — too much register pressure for the shader compiler

## Project Structure

```
prism-inference/
  shaders/                          # Vulkan compute shaders (GLSL + SPIR-V)
    attention_windowed.comp.glsl    # Original windowed attention (dim=128)
    attention_windowed_d256.comp.glsl # Optimized dim=256 with split FFN
    split2_qkv_attn_d256.comp.glsl # Attention-only pass (for MoE 2-split)
    moe_router.comp.glsl           # MoE token-to-expert routing
    moe_expert_skip.comp.glsl      # Expert FFN with early-skip routing
    strided_conv_split_*.comp.glsl  # Optimized encoder (split coopvec)
    dec*_fused.comp.glsl            # Fused decoder stages
    pw_project_*.comp.glsl          # Dimension projection adapters
    bench_*.comp.glsl               # Isolated component benchmarks
    *.spv                           # Compiled SPIR-V (included for convenience)

  vulkan_engine/
    prism_moe_bench.cpp             # End-to-end MoE pipeline benchmark
    prism_overhead_bench.cpp        # Component-level overhead analysis
    prism_scaling_bench.cpp         # Model size scaling benchmark
    prism_linear_bench.cpp          # Linear attention experiments
    prism_v3_bench.cpp              # Original V3 pipeline benchmark
    prism_vulkan.h/cpp              # V1 engine
    export_weights.py               # PyTorch -> flat fp16 weight export
    CMakeLists.txt
    deps/                           # volk + Vulkan headers
```

## Building

```bash
# Build all benchmarks
cd vulkan_engine/build
cmake .. -G "MinGW Makefiles"   # or "Visual Studio 17 2022"
cmake --build .

# Run from repo root (shaders loaded relative to cwd)
cd ../..
./vulkan_engine/build/moe_bench.exe       # End-to-end MoE pipeline
./vulkan_engine/build/overhead_bench.exe  # Component analysis
./vulkan_engine/build/scaling_bench.exe   # Size scaling comparison
```

### Compiling Shaders

Pre-compiled SPIR-V files are included. To recompile from GLSL:

```bash
# Requires glslangValidator (from Vulkan SDK or github.com/KhronosGroup/glslang)
glslangValidator --target-env vulkan1.3 -o shader.spv shader.comp.glsl
```

## Requirements

- NVIDIA GPU with Vulkan 1.3 + `VK_NV_cooperative_vector` (RTX 20xx+)
- MinGW or MSVC for building benchmarks
- [OptiScaler](https://github.com/cdozdil/OptiScaler) for game integration

## Training

The model is trained as a GAN:
- **Generator**: U-Net encoder/decoder + Transformer MoE (this engine)
- **Discriminator**: Multi-scale PatchGAN
- **Data**: Real video + synthetic game data
- **Goal**: Style transfer — game graphics -> photorealistic

Training code is in a separate repository. The `export_weights.py` script converts PyTorch checkpoints to the flat fp16 binary format used by the Vulkan engine.

## License

MIT

## Credits

- [OptiScaler](https://github.com/cdozdil/OptiScaler) — DLSS/FSR/XeSS interception
- [NVIDIA RTXNS](https://github.com/NVIDIA-RTX/RTXNS) — cooperative vector reference
- [volk](https://github.com/zeux/volk) — Vulkan meta-loader
