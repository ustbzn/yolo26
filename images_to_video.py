#!/usr/bin/env python3
"""Convert an image sequence in a folder to a video file."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2


def natural_key(path: Path):
    """Sort paths like 1, 2, 10 instead of 1, 10, 2."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def parse_size(size_text: str):
    match = re.fullmatch(r"(\d+)x(\d+)", size_text.strip().lower())
    if not match:
        raise ValueError("--size format must be WIDTHxHEIGHT, for example: 1920x1080")
    w, h = int(match.group(1)), int(match.group(2))
    if w <= 0 or h <= 0:
        raise ValueError("width and height must be > 0")
    return w, h


def collect_images(input_dir: Path, exts: tuple[str, ...], recursive: bool):
    pattern = "**/*" if recursive else "*"
    files = [p for p in input_dir.glob(pattern) if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=natural_key)


def build_parser():
    parser = argparse.ArgumentParser(description="Convert images in a folder to a video")
    parser.add_argument("--input", required=True, help="Input image folder")
    parser.add_argument("--output", default="output.mp4", help="Output video path, e.g. out.mp4")
    parser.add_argument("--fps", type=float, default=25.0, help="Output video FPS (default: 25)")
    parser.add_argument(
        "--size",
        default="",
        help="Output size WIDTHxHEIGHT. If omitted, use first image size (e.g. 1280x720)",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec, e.g. mp4v (MP4), XVID (AVI). Default: mp4v",
    )
    parser.add_argument(
        "--ext",
        default=".jpg,.jpeg,.png,.bmp,.webp",
        help="Comma-separated image extensions to include",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan subfolders")
    return parser


def main():
    args = build_parser().parse_args()

    if args.fps <= 0:
        print("Error: --fps must be > 0", file=sys.stderr)
        sys.exit(1)

    input_dir = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: input folder not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    exts = tuple(ext.strip().lower() for ext in args.ext.split(",") if ext.strip())
    if not exts:
        print("Error: --ext is empty", file=sys.stderr)
        sys.exit(1)

    images = collect_images(input_dir, exts, args.recursive)
    if not images:
        print(f"Error: no images found in {input_dir} with extensions {exts}", file=sys.stderr)
        sys.exit(1)

    first_frame = cv2.imread(str(images[0]))
    if first_frame is None:
        print(f"Error: failed to read first image: {images[0]}", file=sys.stderr)
        sys.exit(1)

    if args.size:
        width, height = parse_size(args.size)
    else:
        height, width = first_frame.shape[:2]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(args.codec) != 4:
        print("Error: --codec must be exactly 4 characters", file=sys.stderr)
        sys.exit(1)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*args.codec),
        args.fps,
        (width, height),
    )

    if not writer.isOpened():
        print("Error: failed to open video writer. Try another codec like XVID or MJPG.", file=sys.stderr)
        sys.exit(1)

    written = 0
    for idx, img_path in enumerate(images, 1):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"Warning: skip unreadable image: {img_path}", file=sys.stderr)
            continue

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        writer.write(frame)
        written += 1

        if idx % 50 == 0 or idx == len(images):
            print(f"Progress: {idx}/{len(images)}")

    writer.release()

    if written == 0:
        print("Error: no valid images were written", file=sys.stderr)
        sys.exit(1)

    print(f"Done: wrote {written} frames to {output_path}")


if __name__ == "__main__":
    main()
