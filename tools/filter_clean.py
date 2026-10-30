"""
filter_clean.py — 筛选有效、不重复、有标注的数据
用法:
  python tools/filter_clean.py
  python tools/filter_clean.py --txt /path/to/list.txt --output /path/to/clean.txt --workers 16.
"""

import argparse
import hashlib
import os
from multiprocessing import Pool

from tqdm import tqdm

INPUT_TXT = "/pcpt/pcpt/data/2D-Det/COS/data_needclean.txt"
OUTPUT_TXT = "/pcpt/pcpt/data/2D-Det/COS/data_clean.txt"


def get_label_path(img_path):
    p = img_path.replace("/images/", "/labels_6cls/")
    return os.path.splitext(p)[0] + ".txt"


def md5_file(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def check_one(img_path):
    """返回 (img_path, md5) 或 (img_path, None) 表示无效."""
    if not os.path.isfile(img_path):
        return img_path, None, "图片不存在"
    label_path = get_label_path(img_path)
    if not os.path.isfile(label_path):
        return img_path, None, "label不存在"
    try:
        digest = md5_file(img_path)
        return img_path, digest, ""
    except Exception as e:
        return img_path, None, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--txt", default=INPUT_TXT)
    parser.add_argument("--output", default=OUTPUT_TXT)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    with open(args.txt) as f:
        paths = [l.strip() for l in f if l.strip()]

    total = len(paths)
    print(f"[INFO] 原始条目: {total}，进程数: {args.workers}")

    no_img = 0
    no_label = 0
    err = 0
    dup = 0
    seen_md5 = set()
    clean = []

    with Pool(args.workers) as pool, tqdm(total=total, unit="张", dynamic_ncols=True) as bar:
        for img_path, digest, reason in pool.imap_unordered(check_one, paths, chunksize=50):
            bar.update(1)
            if digest is None:
                if "图片" in reason:
                    no_img += 1
                elif "label" in reason:
                    no_label += 1
                else:
                    err += 1
                bar.set_postfix(clean=len(clean), dup=dup, no_img=no_img, no_lbl=no_label)
                continue
            if digest in seen_md5:
                dup += 1
                bar.set_postfix(clean=len(clean), dup=dup, no_img=no_img, no_lbl=no_label)
                continue
            seen_md5.add(digest)
            clean.append(img_path)
            bar.set_postfix(clean=len(clean), dup=dup, no_img=no_img, no_lbl=no_label)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(clean) + "\n")

    print(f"""
[DONE]
  原始总数:   {total}
  图片不存在: {no_img}
  label不存在:{no_label}
  内容重复:   {dup}
  其他错误:   {err}
  ─────────────────
  有效保留:   {len(clean)}
  输出文件:   {args.output}
""")


if __name__ == "__main__":
    main()
