#!/usr/bin/env python3
"""Build all/train/val txt files from filtered traffic-light LabelMe JSON files.

The script keeps images whose shapes contain the target label, resolves each
image basename through dataset/index.json, writes all matched paths to one txt,
then removes samples without labels_6cls GT boxes before shuffling and splitting
train/val.

Example:
  python tools/make_filter_trafficlights_train_val.py \
      --overwrite \
      --val-ratio 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence
from urllib.parse import urlparse


class Sample(NamedTuple):
    image_path: str
    resolved_path: str
    group: str
    json_path: Path


def iter_json_paths(data_dir: Path) -> Iterable[Path]:
    return sorted(p for p in data_dir.rglob("*.json") if p.is_file())


def print_progress(current: int, total: int, width: int = 36) -> None:
    if total <= 0:
        return
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    sys.stderr.write(f"\rScanning JSON [{bar}] {current}/{total} {ratio:6.2%}")
    sys.stderr.flush()
    if current >= total:
        sys.stderr.write("\n")
        sys.stderr.flush()


def image_basename(image_path: str) -> str:
    parsed = urlparse(image_path)
    path = parsed.path if parsed.scheme else image_path
    return os.path.basename(path)


def image_group(image_path: str) -> str:
    parsed = urlparse(image_path)
    path = parsed.path if parsed.scheme else image_path
    parent = os.path.basename(os.path.dirname(path))
    return parent or "__unknown__"


def label_path_from_image(image_path: str) -> Path:
    label_path = image_path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels_6cls{os.sep}")
    return Path(os.path.splitext(label_path)[0] + ".txt")


def load_index(index_path: Path) -> dict[str, str]:
    with index_path.open("r", encoding="utf-8") as f:
        index = json.load(f)
    if not isinstance(index, dict):
        raise TypeError(f"Index must be a JSON object: {index_path}")
    return {str(k): str(v) for k, v in index.items()}


def read_labelme(path: Path) -> tuple[str, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    image_path = data.get("imagePath", "")
    shapes = data.get("shapes", [])
    if not isinstance(image_path, str):
        image_path = ""
    if not isinstance(shapes, list):
        shapes = []

    labels = []
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = shape.get("label")
        if isinstance(label, str):
            labels.append(label)
    return image_path, labels


def read_labelme_fast(path: Path) -> tuple[str, list[str]]:
    """Extract imagePath and labels without parsing the large imageData string."""
    chunks = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            marker_pos = chunk.find('"imageData"')
            if marker_pos >= 0:
                chunks.append(chunk[:marker_pos])
                break
            chunks.append(chunk)

    prefix = "".join(chunks)
    image_match = re.search(r'"imagePath"\s*:\s*"((?:\\.|[^"\\])*)"', prefix)
    image_path = json.loads(f'"{image_match.group(1)}"') if image_match else ""
    labels = [json.loads(f'"{match}"') for match in re.findall(r'"label"\s*:\s*"((?:\\.|[^"\\])*)"', prefix)]
    return image_path, labels


def collect_samples(
    data_dir: Path,
    index: dict[str, str],
    target_label: str,
    max_json: int,
    fast_extract: bool,
    show_progress: bool,
) -> tuple[list[Sample], Counter, Counter, Counter, list[tuple[Path, str]], list[Path], int]:
    samples: list[Sample] = []
    label_counter: Counter = Counter()
    image_group_counter: Counter = Counter()
    selected_group_counter: Counter = Counter()
    missing_index: list[tuple[Path, str]] = []
    bad_json: list[Path] = []

    seen_resolved = set()
    json_paths = list(iter_json_paths(data_dir))
    if max_json > 0:
        json_paths = json_paths[:max_json]

    total_json = len(json_paths)
    if show_progress:
        print_progress(0, total_json)

    scanned = 0
    for json_path in json_paths:
        scanned += 1
        try:
            if fast_extract:
                image_path, labels = read_labelme_fast(json_path)
            else:
                image_path, labels = read_labelme(json_path)
        except (json.JSONDecodeError, OSError):
            bad_json.append(json_path)
            if show_progress and (scanned == total_json or scanned % 100 == 0):
                print_progress(scanned, total_json)
            continue

        label_counter.update(labels)
        if image_path:
            image_group_counter[image_group(image_path)] += 1

        if target_label not in labels:
            if show_progress and (scanned == total_json or scanned % 100 == 0):
                print_progress(scanned, total_json)
            continue

        basename = image_basename(image_path)
        resolved_path = index.get(basename)
        if not resolved_path:
            missing_index.append((json_path, basename))
            if show_progress and (scanned == total_json or scanned % 100 == 0):
                print_progress(scanned, total_json)
            continue

        if resolved_path in seen_resolved:
            if show_progress and (scanned == total_json or scanned % 100 == 0):
                print_progress(scanned, total_json)
            continue
        seen_resolved.add(resolved_path)

        group = image_group(image_path)
        selected_group_counter[group] += 1
        samples.append(Sample(image_path, resolved_path, group, json_path))

        if show_progress and (scanned == total_json or scanned % 100 == 0):
            print_progress(scanned, total_json)

    if show_progress and scanned < total_json:
        print_progress(scanned, total_json)

    return samples, label_counter, image_group_counter, selected_group_counter, missing_index, bad_json, scanned


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


def filter_samples_with_gt(
    samples: Sequence[Sample],
) -> tuple[list[Sample], list[tuple[Sample, Path]], list[tuple[Sample, Path]]]:
    kept: list[Sample] = []
    missing_labels: list[tuple[Sample, Path]] = []
    empty_labels: list[tuple[Sample, Path]] = []

    for sample in samples:
        label_path = label_path_from_image(sample.resolved_path)
        if not label_path.is_file():
            missing_labels.append((sample, label_path))
            continue
        if not has_gt_box(label_path):
            empty_labels.append((sample, label_path))
            continue
        kept.append(sample)

    return kept, missing_labels, empty_labels


def select_val_indices_interleaved(n: int, val_n: int) -> set[int]:
    """Pick validation indices evenly across a sorted sequence."""
    if val_n <= 0:
        return set()
    if val_n >= n:
        return set(range(n))

    idxs: set[int] = set()
    step = n / val_n
    for k in range(val_n):
        idx = int((k + 0.5) * step)
        idxs.add(max(0, min(n - 1, idx)))

    if len(idxs) < val_n:
        for i in range(n):
            if i not in idxs:
                idxs.add(i)
                if len(idxs) == val_n:
                    break
    return idxs


def allocate_group_val_counts(
    by_group: dict[str, list[Sample]],
    val_count: int,
    seed: int,
) -> dict[str, int]:
    """Allocate exact validation count proportionally to groups."""
    total = sum(len(group_samples) for group_samples in by_group.values())
    if val_count <= 0:
        return {group: 0 for group in by_group}
    if val_count >= total:
        return {group: len(group_samples) for group, group_samples in by_group.items()}

    counts: dict[str, int] = {}
    remainders: list[tuple[float, float, str]] = []
    rng = random.Random(seed)
    for group, group_samples in by_group.items():
        exact = len(group_samples) * val_count / total
        count = int(exact)
        counts[group] = count
        remainders.append((exact - count, rng.random(), group))

    remaining = val_count - sum(counts.values())
    for _, _, group in sorted(remainders, key=lambda x: (-x[0], x[1], x[2]))[:remaining]:
        counts[group] += 1

    return counts


def split_shuffled(samples: Sequence[Sample], val_ratio: float, seed: int) -> tuple[list[str], list[str], list[str]]:
    total_paths = [sample.resolved_path for sample in samples]
    if not total_paths:
        return [], [], []

    shuffled = list(total_paths)
    random.Random(seed).shuffle(shuffled)

    ratio = min(max(val_ratio, 0.0), 1.0)
    val_count = round(len(shuffled) * ratio)
    if len(shuffled) > 1 and ratio > 0:
        val_count = max(1, val_count)
    val_count = min(val_count, len(shuffled))

    split_idx = len(shuffled) - val_count
    train = shuffled[:split_idx]
    val = shuffled[split_idx:]
    return total_paths, train, val


def write_lines(path: Path, lines: Sequence[str], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output exists, pass --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if lines:
            f.write("\n".join(lines) + "\n")


def print_counter(title: str, counter: Counter, limit: int) -> None:
    print(title)
    if not counter:
        print("  <empty>")
        return
    for key, count in counter.most_common(limit):
        print(f"  {key}: {count}")
    hidden = len(counter) - limit
    if hidden > 0:
        print(f"  ... {hidden} more")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=root / "filter_trafficlights_data")
    parser.add_argument("--index", type=Path, default=root / "dataset" / "index.json")
    parser.add_argument("--all-out", type=Path, default=root / "dataset/filter" / "all_data_filter.txt")
    parser.add_argument("--train-out", type=Path, default=root / "dataset/filter" / "train_data_filter.txt")
    parser.add_argument("--val-out", type=Path, default=root / "dataset/filter" / "val_data_filter.txt")
    parser.add_argument("--target-label", default="留")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats-limit", type=int, default=30)
    parser.add_argument("--max-json", type=int, default=0, help="Only scan first N JSON files; 0 means no limit")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    parser.add_argument(
        "--full-json",
        action="store_true",
        help="Parse full JSON files instead of fast-extracting imagePath/label before imageData",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only print stats; do not write output files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = load_index(args.index)
    samples, label_counter, group_counter, selected_group_counter, missing_index, bad_json, scanned = collect_samples(
        args.data_dir,
        index,
        args.target_label,
        args.max_json,
        not args.full_json,
        not args.no_progress,
    )

    total = [sample.resolved_path for sample in samples]
    train_val_samples, missing_labels, empty_labels = filter_samples_with_gt(samples)
    _, train, val = split_shuffled(train_val_samples, args.val_ratio, args.seed)

    print(f"JSON dir: {args.data_dir}")
    print(f"Index: {args.index}")
    print(f"Target label: {args.target_label}")
    print(f"Scanned JSON: {scanned}")
    print(f"Matched images: {len(samples)}")
    print(f"All: {len(total)}")
    print(f"Train/val source after labels_6cls GT filter: {len(train_val_samples)}")
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Missing labels_6cls: {len(missing_labels)}")
    print(f"Empty labels_6cls: {len(empty_labels)}")
    print(f"Missing index: {len(missing_index)}")
    print(f"Bad JSON: {len(bad_json)}")
    print_counter("Label stats:", label_counter, args.stats_limit)
    print_counter("imagePath group stats:", group_counter, args.stats_limit)
    print_counter("Selected group stats:", selected_group_counter, args.stats_limit)

    if missing_index:
        print("First missing index entries:")
        for json_path, basename in missing_index[: args.stats_limit]:
            print(f"  {basename}\t{json_path}")

    if missing_labels:
        print("First missing labels_6cls entries:")
        for sample, label_path in missing_labels[: args.stats_limit]:
            print(f"  {sample.resolved_path}\t{label_path}")

    if empty_labels:
        print("First empty labels_6cls entries:")
        for sample, label_path in empty_labels[: args.stats_limit]:
            print(f"  {sample.resolved_path}\t{label_path}")

    if args.dry_run:
        return

    write_lines(args.all_out, total, args.overwrite)
    write_lines(args.train_out, train, args.overwrite)
    write_lines(args.val_out, val, args.overwrite)
    print(f"Wrote all: {args.all_out}")
    print(f"Wrote train: {args.train_out}")
    print(f"Wrote val: {args.val_out}")


if __name__ == "__main__":
    main()
