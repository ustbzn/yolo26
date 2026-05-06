"""
数据预处理脚本（不修改原始数据）：
1. 检查损坏/无法读取的图片，过滤后生成新的 txt 列表
2. 将标签文件去重后复制到新目录 (labels_6cls_clean)
3. 统计每个类别的实例数量

.cache 文件由训练首次启动时自动按目录生成，预处理不再生成。

用法: python preprocess_mogo_6cls.py
"""

import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.pool import ThreadPool
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm


# ---------- 配置 ----------
TXT_FILES = [
    '/pcpt/pcpt/data/2D-Det/COS/TXT/6cls/train_paths_day_6cls_filtercity_0212addQ1_0323.txt',
    '/pcpt/pcpt/data/2D-Det/COS/TXT/6cls/val_paths_day_6cls_filtercity.txt',
]
OUTPUT_DIR = '/pcpt/pcpt/data/2D-Det/COS/TXT/6cls/cleaned'
CLASS_NAMES = {
    0: 'RedLight',
    1: 'GreenLight',
    2: 'YellowLight',
    3: 'BlackLight',
    4: 'MovingTrafficLight',
    5: 'MultiSemanticTrafficLights',
}
NUM_WORKERS = 32
IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp"}
# ---------------------------


def img_to_label_path(img_path: str) -> str:
    """images/xxx.jpg -> labels_6cls/xxx.txt"""
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels_6cls{os.sep}'
    return sb.join(img_path.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt'


def img_to_clean_label_path(img_path: str) -> str:
    """images/xxx.jpg -> labels_6cls_clean/xxx.txt"""
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels_6cls_clean{os.sep}'
    return sb.join(img_path.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt'


def check_image(img_path: str) -> tuple:
    """与 ultralytics verify_image_label 一致的图片校验，返回 (img_path, is_ok)"""
    try:
        im = Image.open(img_path)
        im.verify()
        shape = exif_size(im)
        assert (shape[0] > 9) & (shape[1] > 9), f"image size {shape} <10 pixels"
        assert im.format.lower() in IMG_FORMATS, f"invalid image format {im.format}"
        if im.format.lower() in {"jpg", "jpeg"}:
            with open(img_path, "rb") as f:
                f.seek(-2, 2)
                if f.read() != b"\xff\xd9":
                    return img_path, False  # corrupt JPEG
        return img_path, True
    except Exception:
        return img_path, False


def copy_and_dedup_label(args: tuple) -> int:
    """复制标签文件到新路径，同时去重。返回删除的重复行数。"""
    src_label, dst_label = args
    if not os.path.isfile(src_label):
        return 0
    with open(src_label, 'r') as f:
        lines = f.readlines()
    unique = list(dict.fromkeys(lines))  # 保序去重
    with open(dst_label, 'w') as f:
        f.writelines(unique)
    return len(lines) - len(unique)


def exif_size(img: Image.Image) -> tuple:
    """Return exif-corrected PIL size."""
    s = img.size  # (width, height)
    if img.format == "JPEG":
        try:
            if exif := img.getexif():
                rotation = exif.get(274, None)
                if rotation in {6, 8}:
                    s = s[1], s[0]
        except Exception:
            pass
    return s


def count_classes(label_files: list) -> defaultdict:
    """统计标签文件中各类别实例数量。"""
    class_counts = defaultdict(int)
    for lb_file in label_files:
        if not os.path.isfile(lb_file):
            continue
        with open(lb_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        class_counts[int(float(parts[0]))] += 1
                    except (ValueError, IndexError):
                        pass
    return class_counts


def process_txt(txt_file: str):
    print(f'\n{"="*60}')
    print(f'Processing: {txt_file}')
    print(f'{"="*60}')

    with open(txt_file, 'r') as f:
        img_paths = [line.strip() for line in f if line.strip()]
    total = len(img_paths)
    print(f'Total images: {total}')

    # 1. 检查损坏图片
    print('\n[1/3] Checking corrupt images...')
    corrupt = set()
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(check_image, p): p for p in img_paths}
        for fut in tqdm(as_completed(futures), total=total, desc='Checking'):
            img_path, is_ok = fut.result()
            if not is_ok:
                corrupt.add(img_path)

    valid_paths = [p for p in img_paths if p not in corrupt]
    print(f'  Corrupt: {len(corrupt)}, Valid: {len(valid_paths)}')
    if corrupt:
        print('  Corrupt files (not deleted, only excluded):')
        for p in sorted(corrupt)[:20]:
            print(f'    {p}')
        if len(corrupt) > 20:
            print(f'    ... and {len(corrupt) - 20} more')

    # 2. 复制标签并去重（多线程）
    print('\n[2/3] Copying labels with dedup (-> labels_6cls_clean)...')
    src_labels = [img_to_label_path(p) for p in valid_paths]
    dst_labels = [img_to_clean_label_path(p) for p in valid_paths]
    # 预创建所有目标目录
    dst_dirs = set(os.path.dirname(d) for d in dst_labels)
    for d in dst_dirs:
        os.makedirs(d, exist_ok=True)
    total_dup = 0
    dup_files = 0
    with ThreadPool(NUM_WORKERS) as pool:
        results_dedup = list(tqdm(
            pool.imap(copy_and_dedup_label, zip(src_labels, dst_labels), chunksize=256),
            total=len(valid_paths), desc='Copy & dedup',
        ))
    for removed in results_dedup:
        if removed > 0:
            total_dup += removed
            dup_files += 1
    print(f'  Deduped {total_dup} lines in {dup_files} files.')

    # 3. 生成新 txt
    txt_name = Path(txt_file).stem + '_cleaned.txt'
    new_txt = os.path.join(OUTPUT_DIR, txt_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(new_txt, 'w') as f:
        f.write('\n'.join(valid_paths) + '\n')
    print(f'  New txt: {new_txt} ({len(valid_paths)} images)')

    # 4. 统计类别分布
    print('\n[3/3] Counting class distribution...')
    clean_label_files = [img_to_clean_label_path(p) for p in valid_paths]
    class_counts = count_classes(clean_label_files)

    return new_txt, len(corrupt), total_dup, dup_files, class_counts


def main():
    results = []
    total_class_counts = defaultdict(int)

    for txt_file in TXT_FILES:
        if not os.path.isfile(txt_file):
            print(f'WARNING: {txt_file} not found, skipping.')
            continue
        result = process_txt(txt_file)
        results.append(result)
        for cls_id, cnt in result[4].items():
            total_class_counts[cls_id] += cnt

    print(f'\n{"="*60}')
    print('Summary:')
    for new_txt, n_corrupt, n_dup, n_dup_files, class_counts in results:
        split = 'train' if 'train' in new_txt else 'val'
        print(f'\n  [{split}] {new_txt}')
        print(f'    corrupt excluded: {n_corrupt}, dup lines removed: {n_dup} ({n_dup_files} files)')
        print(f'    class distribution:')
        for cls_id in sorted(class_counts.keys()):
            name = CLASS_NAMES.get(cls_id, f'class_{cls_id}')
            print(f'      {cls_id}: {name:30s} {class_counts[cls_id]:>10,}')

    print(f'\n  [total] class distribution:')
    for cls_id in sorted(total_class_counts.keys()):
        name = CLASS_NAMES.get(cls_id, f'class_{cls_id}')
        print(f'    {cls_id}: {name:30s} {total_class_counts[cls_id]:>10,}')

    total_instances = sum(total_class_counts.values())
    print(f'    {"":30s}  ----------')
    print(f'    {"Total":30s} {total_instances:>10,}')

    print(f'\nOriginal data NOT modified.')
    print(f'Clean labels: labels_6cls_clean/')
    print(f'Clean txt: {OUTPUT_DIR}/')
    print(f'.cache files will be auto-generated on first training run.')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
