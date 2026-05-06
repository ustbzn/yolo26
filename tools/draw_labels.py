"""
draw_labels.py — 批量可视化 YOLO 格式标注
用法:
  python tools/draw_labels.py --txt /path/to/list.txt --output /tmp/vis
  python tools/draw_labels.py --txt /path/to/list.txt --output /tmp/vis --limit 500 --workers 8
"""

import os
import json
import argparse
from multiprocessing import Pool
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont

INPUT_TXT   = "/pcpt/pcpt/data/2D-Det/COS/data_clean_zn.txt"
OUTPUT_DIR  = "/pcpt/pcpt/project/atlb/data/TrafficLight/data"
DATASET_DIR = "/pcpt/pcpt/project/atlb/vla/ultralytics/dataset"
BATCH_SIZE  = 150000

CLASS_COLORS = {
    0: (255, 0,   0),    # RedLight           — 红
    1: (0,   255, 0),    # GreenLight         — 绿
    2: (255, 255, 0),    # YellowLight        — 黄
    3: (0,   0,   0),    # BlackLight         — 黑
    4: (0,   0,   255),  # MovingTrafficLight — 蓝
    5: (128, 0,   128),  # MultiSemanticTrafficLights — 紫
}
CLASS_NAMES = {
    0: "红灯",
    1: "绿灯",
    2: "黄灯",
    3: "黑灯",
    4: "运动交通灯",
    5: "多语义交通灯",
}
LINE_WIDTH = 1

try:
    FONT = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 18)
except Exception:
    FONT = ImageFont.load_default()

LEGEND_SWATCH = 18
LEGEND_PAD    = 8
LEGEND_GAP    = 6
LEGEND_MARGIN = 8


def draw_legend(draw, img_w, img_h):
    line_h = LEGEND_SWATCH + LEGEND_PAD
    n = len(CLASS_COLORS)
    text_w = max(FONT.getlength(CLASS_NAMES[c]) for c in CLASS_NAMES)
    box_w = int(LEGEND_PAD * 2 + LEGEND_SWATCH + LEGEND_GAP + text_w)
    box_h = LEGEND_PAD + n * line_h

    x0 = LEGEND_MARGIN
    y0 = img_h - LEGEND_MARGIN - box_h

    draw.rectangle([x0, y0, x0 + box_w, y0 + box_h], fill=(40, 40, 40))

    for i, cls in enumerate(sorted(CLASS_COLORS)):
        cy = y0 + LEGEND_PAD + i * line_h
        sx = x0 + LEGEND_PAD
        draw.rectangle([sx, cy, sx + LEGEND_SWATCH, cy + LEGEND_SWATCH],
                       fill=CLASS_COLORS[cls], outline=(200, 200, 200), width=1)
        draw.text((sx + LEGEND_SWATCH + LEGEND_GAP, cy), CLASS_NAMES[cls],
                  fill=(255, 255, 255), font=FONT)


def get_label_path(img_path):
    p = img_path.replace("/images/", "/labels_6cls/")
    return os.path.splitext(p)[0] + ".txt"


def draw_one(args):
    img_path, out_path = args
    try:
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        draw = ImageDraw.Draw(img)

        label_path = get_label_path(img_path)
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    color = CLASS_COLORS.get(cls, (255, 255, 0))
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=LINE_WIDTH)

        draw_legend(draw, w, h)
        img.save(out_path, quality=85)
        return True, img_path
    except Exception as e:
        return False, f"{img_path}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt",     default=INPUT_TXT)
    parser.add_argument("--output",  default=OUTPUT_DIR)
    parser.add_argument("--dataset", default=DATASET_DIR, help="全局 JSON 输出目录")
    parser.add_argument("--limit",   type=int, default=0, help="最多处理 N 张，0=全部")
    parser.add_argument("--start",   type=int, default=0, help="从第 N 张开始")
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    with open(args.txt) as f:
        paths = [l.strip() for l in f if l.strip()]

    paths = paths[args.start:]
    if args.limit > 0:
        paths = paths[:args.limit]

    total = len(paths)
    print(f"[INFO] 共 {total} 张，输出目录: {args.output}，进程数: {args.workers}")

    # 分 batch，每批 BATCH_SIZE 张，预建目录
    batch_dirs = []
    for i in range(0, total, BATCH_SIZE):
        d = os.path.join(args.output, f"batch{i // BATCH_SIZE}")
        os.makedirs(d, exist_ok=True)
        batch_dirs.append(d)

    # 为每张图分配输出路径，记录 filename -> 原始路径 映射
    all_index = {}       # 全局: {filename: src_path}
    index_map = {}       # 按 batch: {batch_dir: {filename: src_path}}
    tasks = []
    for i, src in enumerate(paths):
        batch_dir = batch_dirs[i // BATCH_SIZE]
        fname = os.path.basename(src)
        out_path = os.path.join(batch_dir, fname)
        tasks.append((src, out_path))
        index_map.setdefault(batch_dir, {})[fname] = src
        all_index[fname] = src

    # 每个 batch 单独的 JSON
    for batch_dir, mapping in index_map.items():
        with open(os.path.join(batch_dir, "index.json"), "w") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)

    # 全局唯一 JSON 存到 dataset 目录
    os.makedirs(args.dataset, exist_ok=True)
    global_json = os.path.join(args.dataset, "index.json")
    with open(global_json, "w") as f:
        json.dump(all_index, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 共 {len(batch_dirs)} 个 batch，全局 JSON: {global_json}")

    ok = err = 0
    with Pool(args.workers) as pool:
        with tqdm(total=total, unit="张", dynamic_ncols=True) as bar:
            for success, msg in pool.imap_unordered(draw_one, tasks, chunksize=20):
                if success:
                    ok += 1
                else:
                    err += 1
                    tqdm.write(f"[WARN] {msg}")
                bar.set_postfix(ok=ok, err=err)
                bar.update(1)

    print(f"[DONE] ok={ok}  err={err}  输出: {args.output}")


if __name__ == "__main__":
    main()
