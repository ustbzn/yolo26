#!/usr/bin/env python3
"""Prepare traffic-light timer images in 50k batches and optionally upload to COS."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

DEFAULT_LISTS = [
    # Path("/pcpt/pcpt/data/2D-Det/COS/TXT/yolo26_fiter_data/train_data_filter_0525.txt"),
    Path("/pcpt/pcpt/data/2D-Det/COS/TXT/yolo26_fiter_data/val_data_filter.txt"),
]
DEFAULT_SOURCE_ROOT = Path("/pcpt/pcpt/data/2D-Det")
DEFAULT_OUTPUT_ROOT = Path("data_2")
DEFAULT_COS_PREFIX = "newbag/2d_datas_remark_0604/"


def read_image_paths(list_paths: Iterable[Path], dedup: bool) -> list[Path]:
    image_paths: list[Path] = []
    seen = set()

    for list_path in list_paths:
        if not list_path.is_file():
            raise FileNotFoundError(f"list file not found: {list_path}")
        with list_path.open("r", encoding="utf-8") as f:
            for line in f:
                raw_path = line.strip()
                if not raw_path:
                    continue
                if dedup and raw_path in seen:
                    continue
                seen.add(raw_path)
                image_paths.append(Path(raw_path))

    return image_paths


def relative_image_path(image_path: Path, source_root: Path) -> Path:
    try:
        return image_path.relative_to(source_root)
    except ValueError:
        return Path(*image_path.parts[1:])


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if mode in {"hardlink", "auto"}:
        try:
            os.link(src, dst)
            return
        except OSError:
            if mode == "hardlink":
                raise

    if mode in {"copy", "auto"}:
        shutil.copy2(src, dst)
        return

    if mode == "symlink":
        os.symlink(src, dst)
        return

    raise ValueError(f"unsupported mode: {mode}")


def prepare_batches(args: argparse.Namespace) -> tuple[int, int, int]:
    image_paths = read_image_paths(args.list, args.dedup)
    missing_count = 0
    prepared_count = 0

    args.output_root.mkdir(parents=True, exist_ok=True)

    for index, image_path in enumerate(image_paths):
        if not image_path.is_file():
            missing_count += 1
            if len(args.missing_examples) < args.stats_limit:
                args.missing_examples.append(str(image_path))
            continue

        batch_id = prepared_count // args.batch_size + 1
        batch_dir = args.output_root / f"{args.batch_prefix}_{batch_id:06d}"
        dst = batch_dir / relative_image_path(image_path, args.source_root)
        link_or_copy(image_path, dst, args.mode)
        prepared_count += 1

        if args.progress_interval > 0 and prepared_count % args.progress_interval == 0:
            print(f"[INFO] prepared {prepared_count}/{len(image_paths)} images")

    return len(image_paths), prepared_count, missing_count


def upload_to_cos(output_root: Path, cos_prefix: str) -> None:
    cmd = ["coscmd", "upload", "-r", str(output_root), cos_prefix]
    print("[INFO] upload command:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        type=Path,
        nargs="+",
        default=DEFAULT_LISTS,
        help="Image list txt files. Default uses train_data_filter_0525.txt and val_data_filter.txt.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Local upload root. Default: data",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Source root stripped from destination paths. Default: /pcpt/pcpt/data/2D-Det",
    )
    parser.add_argument("--batch-size", type=int, default=50000, help="Images per batch folder.")
    parser.add_argument("--batch-prefix", default="batch", help="Batch folder prefix.")
    parser.add_argument(
        "--mode",
        choices=("auto", "hardlink", "copy", "symlink"),
        default="auto",
        help="File materialization mode. Default hardlinks first, then copies.",
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="Deduplicate identical image paths before batching.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Run: coscmd upload -r data badcase-data-1255510688/newbag/2d_datas_remark_0604/",
    )
    parser.add_argument(
        "--cos-prefix",
        default=DEFAULT_COS_PREFIX,
        help="COS destination prefix. Default: badcase-data-1255510688/newbag/2d_datas_remark_0604/",
    )
    parser.add_argument("--stats-limit", type=int, default=20, help="Max missing image examples to print.")
    parser.add_argument("--progress-interval", type=int, default=10000, help="Progress print interval.")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.upload and args.output_root != DEFAULT_OUTPUT_ROOT:
        print(f"[WARN] upload command will use custom output root: {args.output_root}")

    args.missing_examples = []
    return args


def main() -> None:
    args = parse_args()
    total_count, prepared_count, missing_count = prepare_batches(args)

    batch_count = (prepared_count + args.batch_size - 1) // args.batch_size
    print("[DONE]")
    print(f"  input images:     {total_count}")
    print(f"  prepared images:  {prepared_count}")
    print(f"  missing images:   {missing_count}")
    print(f"  batch folders:    {batch_count}")
    print(f"  output root:      {args.output_root}")

    if args.missing_examples:
        print("  first missing images:")
        for image_path in args.missing_examples:
            print(f"    {image_path}")

    if args.upload:
        upload_to_cos(args.output_root, args.cos_prefix)
    else:
        print("  upload skipped:   add --upload to run coscmd")


if __name__ == "__main__":
    main()
