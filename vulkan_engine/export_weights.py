#!/usr/bin/env python3
"""Export ncnn model weights to a flat fp16 binary for the Vulkan compute engine.

Usage: python export_weights.py <param_file> <bin_file> <output.weights>

The .weights file is a flat fp16 buffer with all weights packed contiguously.
A companion .json file is written with offsets and layer info.
"""
import sys, struct, json, numpy as np
from pathlib import Path

def parse_ncnn_param(param_path):
    """Parse ncnn .param file to get layer info and weight sizes."""
    layers = []
    with open(param_path) as f:
        magic = f.readline().strip()
        assert magic == "7767517", f"Bad magic: {magic}"
        layer_count, blob_count = map(int, f.readline().split())
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            layer_type = parts[0]
            layer_name = parts[1]
            n_in = int(parts[2])
            n_out = int(parts[3])
            # Parse key=value params
            params = {}
            idx = 4 + n_in + n_out
            while idx < len(parts):
                kv = parts[idx].split("=")
                params[int(kv[0])] = int(kv[1])
                idx += 1
            layers.append({
                "type": layer_type,
                "name": layer_name,
                "params": params,
            })
    return layers

def calc_weight_sizes(layers):
    """Calculate weight data sizes (in fp32 values) for each layer."""
    result = []
    for layer in layers:
        t = layer["type"]
        p = layer["params"]
        if t == "Convolution":
            out_ch = p[0]
            kw = p.get(1, 1)
            kh = p.get(11, kw)
            weight_count = p.get(6, 0)  # explicit weight count
            has_bias = p.get(5, 0)
            bias_count = out_ch if has_bias else 0
            result.append({
                "name": layer["name"],
                "type": "conv",
                "weight_count": weight_count,
                "bias_count": bias_count,
                "total_fp32": weight_count + bias_count,
            })
        elif t == "ConvolutionDepthWise":
            channels = p[0]
            weight_count = p.get(6, 0)
            has_bias = p.get(5, 0)
            bias_count = channels if has_bias else 0
            result.append({
                "name": layer["name"],
                "type": "dw_conv",
                "weight_count": weight_count,
                "bias_count": bias_count,
                "total_fp32": weight_count + bias_count,
            })
        # Other layer types (ReLU, Split, BinaryOp, etc.) have no weights
    return result

def main():
    if len(sys.argv) < 4:
        print("Usage: export_weights.py <param> <bin> <output.weights>")
        sys.exit(1)

    param_path = sys.argv[1]
    bin_path = sys.argv[2]
    out_path = sys.argv[3]

    layers = parse_ncnn_param(param_path)
    weight_info = calc_weight_sizes(layers)

    # Read ncnn .bin (all fp32 packed sequentially)
    bin_data = np.fromfile(bin_path, dtype=np.float32)
    print(f"Total fp32 values in .bin: {len(bin_data)}")

    # Verify total matches
    total_expected = sum(w["total_fp32"] for w in weight_info)
    print(f"Expected from param: {total_expected}")

    # Convert to fp16 and pack
    fp16_data = bin_data[:total_expected].astype(np.float16)

    # Calculate offsets in fp16 elements
    offset = 0
    layer_offsets = []
    for w in weight_info:
        w["weight_offset_fp16"] = offset
        offset += w["weight_count"]
        w["bias_offset_fp16"] = offset
        offset += w["bias_count"]
        layer_offsets.append(w)

    # Write fp16 binary
    fp16_data.tofile(out_path)
    print(f"Wrote {out_path}: {len(fp16_data)*2} bytes ({len(fp16_data)} fp16 values)")

    # Write JSON descriptor
    json_path = out_path.replace(".weights", ".json")
    with open(json_path, "w") as f:
        json.dump({
            "total_fp16_values": len(fp16_data),
            "total_bytes": len(fp16_data) * 2,
            "layers": layer_offsets,
        }, f, indent=2)
    print(f"Wrote {json_path}")

if __name__ == "__main__":
    main()
