#!/usr/bin/env python3
"""Download/validate DIV2K and create lossless RGB patch manifests."""

from __future__ import annotations

import argparse
import csv
import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


URLS = {
    "train": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "valid": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise ValueError(f"unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def download_div2k(source: Path) -> None:
    source.mkdir(parents=True, exist_ok=True)
    for split, url in URLS.items():
        archive = source / Path(url).name
        target = source / f"DIV2K_{'train' if split == 'train' else 'valid'}_HR"
        if target.exists():
            continue
        if not archive.exists():
            print(f"downloading {url}")
            with urllib.request.urlopen(url) as response, archive.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        safe_extract(archive, source)


def positions(length: int, patch: int, stride: int) -> list[int]:
    if length < patch:
        return []
    values = list(range(0, length - patch + 1, stride))
    if values[-1] != length - patch:
        values.append(length - patch)
    return values


def select_grid(grid: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    if limit <= 0 or len(grid) <= limit:
        return grid
    indices = np.linspace(0, len(grid) - 1, num=limit, dtype=int)
    return [grid[index] for index in indices]


def process_images(images: list[Path], split: str, output: Path, patch: int,
                   stride: int, max_per_image: int) -> list[dict]:
    patch_dir = output / "patches" / split
    patch_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for image_path in images:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            width, height = image.size
            grid = [(y, x) for y in positions(height, patch, stride)
                    for x in positions(width, patch, stride)]
            for y, x in select_grid(grid, max_per_image):
                patch_id = f"{image_path.stem}_y{y:04d}_x{x:04d}"
                relative = Path("patches") / split / f"{patch_id}.png"
                image.crop((x, y, x + patch, y + patch)).save(output / relative, compress_level=3)
                records.append({"split": split, "image_id": image_path.stem,
                                "patch_id": patch_id, "path": relative.as_posix(),
                                "x": x, "y": y, "width": patch, "height": patch})
        print(f"{split}: {image_path.name}")
    return records


def write_manifest(output: Path, split: str, records: list[dict]) -> None:
    manifest = output / "manifests" / f"{split}.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    fields = ["split", "image_id", "patch_id", "path", "x", "y", "width", "height"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="directory containing DIV2K_train_HR/valid_HR")
    parser.add_argument("--out", required=True, help="processed dataset directory")
    parser.add_argument("--download", action="store_true", help="download official HR archives if missing")
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--max-patches-per-image", type=int, default=32,
                        help="0 keeps the complete grid")
    parser.add_argument("--val-images", type=int, default=50,
                        help="first N official validation images form val; remaining form test")
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.out)
    if args.patch_size < 3 or args.stride < 1:
        raise ValueError("patch-size must be >=3 and stride positive")
    if args.download:
        download_div2k(source)
    train_images = sorted((source / "DIV2K_train_HR").glob("*.png"))
    valid_images = sorted((source / "DIV2K_valid_HR").glob("*.png"))
    if not train_images or not valid_images:
        raise FileNotFoundError("expected DIV2K_train_HR and DIV2K_valid_HR PNG files")
    if not 1 <= args.val_images < len(valid_images):
        raise ValueError("val-images must leave at least one image for test")
    splits = {"train": train_images, "val": valid_images[:args.val_images],
              "test": valid_images[args.val_images:]}
    output.mkdir(parents=True, exist_ok=True)
    for split, images in splits.items():
        records = process_images(images, split, output, args.patch_size, args.stride,
                                 args.max_patches_per_image)
        write_manifest(output, split, records)
        print(f"wrote {len(records)} {split} patches")


if __name__ == "__main__":
    main()
