#!/usr/bin/env python3
"""Filter image paths with labels_6cls GT boxes, then split train/val.

Default behavior:
- input: dataset/bzp/dataset_paths_Q1_Q4.txt
- filter: remove samples without labels_6cls or without valid GT boxes
- split: train/val = 9:1 (val_ratio=0.1)
- output: dataset/bzp/train_paths_Q1_Q4.txt and dataset/bzp/val_paths_Q1_Q4.txt
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple


def read_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")


def label_path_from_image(image_path: str) -> Path:
    label_path = image_path.replace("/images/", "/labels_10cls/")
    return Path(label_path).with_suffix(".txt")


def has_gt_box(label_path: Path) -> bool:
    try:
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    int(parts[0])
                    [float(v) for v in parts[1:5]]
                except ValueError:
                    continue
                return True
    except OSError:
        return False
    return False


def filter_lines_with_gt(lines: List[str]) -> Tuple[List[str], List[Tuple[str, Path]], List[Tuple[str, Path]]]:
    kept: List[str] = []
    missing_labels: List[Tuple[str, Path]] = []
    empty_labels: List[Tuple[str, Path]] = []

    for line in lines:
        label_path = label_path_from_image(line)
        if not label_path.is_file():
            missing_labels.append((line, label_path))
            continue
        if not has_gt_box(label_path):
            empty_labels.append((line, label_path))
            continue
        kept.append(line)

    return kept, missing_labels, empty_labels


def split_shuffled(lines: List[str], val_ratio: float, seed: int) -> tuple[List[str], List[str]]:
    items = list(lines)
    random.Random(seed).shuffle(items)

    n = len(items)
    if n == 0:
        return [], []

    ratio = min(max(val_ratio, 0.0), 1.0)
    val_n = int(round(n * ratio))
    if n > 1 and ratio > 0:
        val_n = max(1, val_n)
    val_n = min(val_n, n)

    split_idx = n - val_n
    train = items[:split_idx]
    val = items[split_idx:]
    return train, val


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "dataset" / "timer" / "dataset_timer.txt",
        help="Input txt path.",
    )
    parser.add_argument(
        "--train-out",
        type=Path,
        default=root / "dataset" / "timer" / "train_timer.txt",
        help="Train output txt path.",
    )
    parser.add_argument(
        "--val-out",
        type=Path,
        default=root / "dataset" / "timer" / "val_timer.txt",
        help="Val output txt path.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio in [0,1], default 0.1.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dedup", action="store_true", help="Deduplicate lines before shuffle.")
    parser.add_argument(
        "--no-filter-gt",
        action="store_true",
        help="Disable labels_6cls GT filtering before shuffle.",
    )
    parser.add_argument(
        "--stats-limit",
        type=int,
        default=30,
        help="Max missing/empty labels_6cls examples to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    lines = read_lines(args.input)
    total_before = len(lines)

    if args.dedup:
        seen = set()
        deduped = []
        for x in lines:
            if x in seen:
                continue
            seen.add(x)
            deduped.append(x)
        lines = deduped

    missing_labels: List[Tuple[str, Path]] = []
    empty_labels: List[Tuple[str, Path]] = []
    if not args.no_filter_gt:
        lines, missing_labels, empty_labels = filter_lines_with_gt(lines)

    train, val = split_shuffled(lines, args.val_ratio, args.seed)

    write_lines(args.train_out, train)
    write_lines(args.val_out, val)

    print(f"Input: {args.input}")
    print(f"Total before dedup: {total_before}")
    print(f"Total used: {len(lines)}")
    print(f"Filter labels_6cls GT: {not args.no_filter_gt}")
    print(f"Missing labels_6cls: {len(missing_labels)}")
    print(f"Empty labels_6cls: {len(empty_labels)}")
    print(f"Shuffle seed: {args.seed}")
    print(f"Val ratio: {args.val_ratio}")
    print(f"Train: {len(train)} -> {args.train_out}")
    print(f"Val:   {len(val)} -> {args.val_out}")

    if missing_labels:
        print("First missing labels_6cls entries:")
        for image_path, label_path in missing_labels[: args.stats_limit]:
            print(f"  {image_path}\t{label_path}")

    if empty_labels:
        print("First empty labels_6cls entries:")
        for image_path, label_path in empty_labels[: args.stats_limit]:
            print(f"  {image_path}\t{label_path}")


if __name__ == "__main__":
    main()
