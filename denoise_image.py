#!/usr/bin/env python3
"""Denoise one RGB image with overlap-tiled TriTD-Half inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.io_utils import resolve_device, resolve_dtype
from train import build_model


def positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def pad_to_tile(image: np.ndarray, tile: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    pad_h, pad_w = max(tile - height, 0), max(tile - width, 0)
    if pad_h or pad_w:
        mode = "reflect" if height > 1 and width > 1 else "edge"
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    return image, (height, width)


@torch.no_grad()
def tiled_denoise(model, image: np.ndarray, tile_size: int = 64, overlap: int = 16,
                  batch_size: int = 4, device: torch.device = torch.device("cpu"),
                  dtype: torch.dtype = torch.float32) -> tuple[np.ndarray, np.ndarray]:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be RGB HxWx3")
    if tile_size < 3 or not 0 <= overlap < tile_size:
        raise ValueError("require tile_size>=3 and 0<=overlap<tile_size")
    padded, original_shape = pad_to_tile(image, tile_size)
    stride = tile_size - overlap
    locations = [(y, x) for y in positions(padded.shape[0], tile_size, stride)
                 for x in positions(padded.shape[1], tile_size, stride)]
    low_sum = np.zeros_like(padded, dtype=np.float64)
    sparse_sum = np.zeros_like(padded, dtype=np.float64)
    weight = np.zeros(padded.shape[:2] + (1,), dtype=np.float64)
    for start in range(0, len(locations), batch_size):
        chunk = locations[start:start + batch_size]
        patches = np.stack([padded[y:y + tile_size, x:x + tile_size] for y, x in chunk])
        tensor = torch.from_numpy(patches).to(device=device, dtype=dtype)
        output = model(tensor)
        lows = output["L"].clamp(0.0, 1.0).cpu().numpy()
        sparsities = output["S"].cpu().numpy()
        for (y, x), low, sparse in zip(chunk, lows, sparsities):
            low_sum[y:y + tile_size, x:x + tile_size] += low
            sparse_sum[y:y + tile_size, x:x + tile_size] += sparse
            weight[y:y + tile_size, x:x + tile_size] += 1.0
    height, width = original_shape
    return ((low_sum / weight)[:height, :width].astype(np.float32),
            (sparse_sum / weight)[:height, :width].astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="noisy RGB image")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="denoised PNG/JPEG path")
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    args = parser.parse_args()
    device, dtype = resolve_device(args.device), resolve_dtype(args.dtype)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"]).to(device=device, dtype=dtype)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with Image.open(args.input) as source:
        image = np.asarray(source.convert("RGB"), dtype=np.float32) / 255.0
    low, sparse = tiled_denoise(model, image, args.tile_size, args.overlap,
                                args.batch_size, device, dtype)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(np.clip(low, 0, 1) * 255).astype(np.uint8), "RGB").save(output)
    np.save(output.with_name(output.stem + "_noise.npy"), sparse)
    noise_view = np.round(np.clip(0.5 + sparse, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(noise_view, "RGB").save(output.with_name(output.stem + "_noise.png"))
    print(f"saved denoised image to {output}")


if __name__ == "__main__":
    main()

