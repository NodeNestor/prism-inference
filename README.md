# Prism — Neural Rendering Engine

> **Status: Architecture Validated** — Inference engine achieves **12.4ms at 1080p** (80 FPS) on RTX 5060 Ti using Vulkan compute with cooperative vectors (tensor cores). Model training starting now.

Prism is a neural renderer that makes games look photorealistic. It intercepts G-buffer data (color, depth, motion vectors) through [OptiScaler](https://github.com/cdozdil/OptiScaler) and runs a trained neural network for **style transfer** — not just upscaling, but transforming game graphics toward photorealism.

Think DLSS 5 Ray Reconstruction, but open source and GAN-trained on real video.

## Architecture: U-Net + Transformer

```
Game renders at 540p (G-buffer: color + depth + motion vectors)
  -> Conv Encoder: 540p -> 270p -> 135p -> 68p (progressive downsampling)
  -> Transformer Blocks at 68p (global scene understanding on tensor cores)
  -> Conv Decoder with skip connections: 68p -> 135p -> 270p -> 540p
  -> PixelShuffle: 540p -> 1080p photorealistic output
```

**Key design choices:**
- **U-Net with aggressive downsampling** — expensive transformer blocks run at 1/8 resolution (8K tokens instead of 518K pixels). This is how FSR Neural achieves <1ms.
- **Windowed self-attention** (8x8 windows) — each token attends to 64 neighbors, not 8K. Global context through stacked blocks.
- **Cooperative vectors** (`VK_NV_cooperative_vector`) — all matrix operations run on tensor cores. 173x speedup over regular compute shaders.
- **Single command buffer** — all dispatches chained with pipeline barriers. Zero CPU overhead during inference.
- **GAN-trained** — PatchGAN discriminator trained on real video footage. The network learns photorealistic style, not just sharpening.

## Confirmed Performance (RTX 5060 Ti, 540p -> 1080p)

| Stage | GPU Time |
|-------|----------|
| Input conv (9->32ch) | 0.31ms |
| Encoder (3 strided convs) | 5.05ms |
| Transformer (4 blocks, tensor cores) | 3.68ms |
| Decoder (3 upsample + skip + pw conv) | 3.23ms |
| Output + PixelShuffle | 0.19ms |
| **Total** | **12.4ms (80 FPS)** |

All numbers are real GPU timestamps from Vulkan queries, not estimates.

### DLSS Preset Scaling

| Preset | Render Res | Tokens | Expected Speed |
|--------|-----------|--------|---------------|
| Ultra Performance (3x) | 360p | ~1K | ~4ms |
| Performance (2x) | 540p | ~8K | ~12ms |
| Quality (1.5x) | 720p | ~14K | ~18ms |
| DLAA (1x) | 1080p | ~32K | ~30ms |

## Model Presets

| Preset | Params | Transformer | Target Use |
|--------|--------|-------------|------------|
| fast | ~400K | 2 blocks | Real-time at 144Hz |
| balanced | ~800K | 4 blocks | Quality at 60Hz |
| quality | ~1.2M | 6 blocks | Maximum quality |

## How It Works

1. **OptiScaler** intercepts DLSS/FSR/XeSS calls from any game
2. Game provides: color, depth, motion vectors (already on GPU)
3. Prism runs the neural network via Vulkan compute shaders
4. Photorealistic output replaces the game's frame
5. Temporal GRU maintains coherence between frames (at bottleneck resolution — nearly free)

## Vulkan Compute Engine

The inference engine uses native Vulkan compute shaders with NVIDIA cooperative vectors for tensor core acceleration:

- `shaders/attention_windowed.comp.glsl` — Fused transformer block (QKV + windowed attention + FFN)
- `shaders/conv3x3_coopvec_*.comp.glsl` — im2col + tensor core convolution
- `shaders/strided_conv_coopvec_*.comp.glsl` — Strided encoder convolutions
- `shaders/pw_conv_coopvec_*.comp.glsl` — Pointwise decoder convolutions
- `shaders/nn_upsample.comp.glsl` — Nearest-neighbor upsampling
- `shaders/pixelshuffle_sigmoid.comp.glsl` — PixelShuffle output

All shaders use `GL_NV_cooperative_vector` for tensor core matrix operations via `coopVecMatMulAddNV`.

## Project Structure

```
prism-inference/
  shaders/              # Vulkan compute shaders (GLSL + compiled SPIR-V)
  vulkan_engine/        # C++ Vulkan inference engine + benchmarks
    prism_vulkan.h/cpp  # V1 engine (DSC blocks)
    prism_v3_bench.cpp  # V2 end-to-end benchmark (U-Net + Transformer)
    deps/               # volk.h + Vulkan headers
  training/             # In prism-optiscaler repo
    model_v2.py         # PyTorch model definition
    benchmark_v2.py     # PyTorch speed testing
```

## Training (In Progress)

The model is trained as a GAN:
- **Generator**: U-Net + Transformer (this architecture)
- **Discriminator**: Multi-scale PatchGAN
- **Data**: Real video (nature, urban, medieval, etc.) + synthetic game data (TartanAir)
- **Goal**: Style transfer — game graphics -> photorealistic

Training runs on A100 with the dataset ready. Production models coming soon.

## Requirements

- NVIDIA GPU with Vulkan 1.3 + `VK_NV_cooperative_vector` (RTX 20xx+)
- [OptiScaler](https://github.com/cdozdil/OptiScaler) for game integration

## License

MIT

## Credits

- [OptiScaler](https://github.com/cdozdil/OptiScaler) — DLSS/FSR/XeSS interception
- [FSR Neural](https://gpuopen.com/amd-fsr-sdk/) — architectural inspiration (U-Net, ML2CODE)
- [NVIDIA RTXNS](https://github.com/NVIDIA-RTX/RTXNS) — cooperative vector reference
