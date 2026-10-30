"""
Ground Truth Annotation Viewer  —  多人协同 Web UI.
===================================================
用法:  python tools/gt_viewer.py [--port 8888]
浏览器: http://<server-ip>:8888

快捷键:
  ← / →   上一帧 / 下一帧
  Q        Accept（自动跳下一未审帧）
  E        Reject（停留当前帧）
  N        跳到下一个未审帧
  Delete   删除选中框
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, request
from PIL import Image, ImageDraw

# ──────────────────────────────────────────────────────────────
# 数据集配置
# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASE_DATA_ROOT = ROOT / "case_data_mau"
# 参考 infer.py：10 类 timer 模型，先用 yaml 配置建模型再 load 权重
DEFAULT_MODEL_CFG = ROOT / "ultralytics/cfg/models/26/yolo26m-timer.yaml"
DEFAULT_WEIGHTS = Path(
    "/pcpt/pcpt/project/atlb/vla/ultralytics/runs/detect/trafficlights_logsim_finetune_timer/3600frames_lr1e-45/weights/best.pt"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# TRAIN_TXT = "/pcpt/pcpt/data/2D-Det/COS/data_clean.txt"
DEFAULT_TRAIN_TXT = "/pcpt/pcpt/data/2D-Det/COS/2026_Q2/0418/2026_Q2_0418_train_paths.txt"
DEFAULT_VAL_TXT = ""
TRAIN_TXT = DEFAULT_TRAIN_TXT
VAL_TXT = DEFAULT_VAL_TXT

CLASS_NAMES = {
    0: "RedLight",
    1: "GreenLight",
    2: "YellowLight",
    3: "BlackLight",
    4: "MovingTrafficLight",
    5: "MultiSemanticTrafficLights",
    6: "BlackTimingCard",
    7: "GreenTimingCard",
    8: "RedTimingCard",
    9: "YellowTimingCard",
}
CLASS_COLORS = {
    0: (220, 50, 50),
    1: (50, 200, 50),
    2: (220, 180, 0),
    3: (130, 130, 130),
    4: (50, 150, 255),
    5: (200, 50, 200),
    6: (80, 80, 80),
    7: (0, 160, 90),
    8: (180, 40, 40),
    9: (240, 200, 40),
}
USER_COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#f06292", "#ce93d8", "#80cbc4", "#fff176", "#ffcc80"]

PROGRESS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gt_viewer_progress.json")
DEFAULT_ACCEPTED_TXT = str(ROOT / "dataset/accepted.txt")
RESTORED_USER = "accepted.txt"


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────
def get_label_path(img_path: str) -> str:
    p = img_path.replace("/images/", "/labels_10cls/")
    return os.path.splitext(p)[0] + ".txt"


def load_txt(path: str) -> list[str]:
    if not os.path.exists(path):
        print(f"[WARN] 文件不存在: {path}")
        return []
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def normalize_path(path: str) -> str:
    """统一为绝对路径，避免数据插入导致下标进度漂移。."""
    return str(Path(path).expanduser().resolve())


def discover_image_paths(data_root: str) -> list[str]:
    """扫描 case 数据根目录下 images 目录中的图像。."""
    root = Path(data_root).expanduser().resolve()
    if not root.exists():
        print(f"[WARN] 数据目录不存在: {root}")
        return []
    paths = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and p.parent.name == "images":
            paths.append(str(p))
    return sorted(paths)


def source_name(img_path: str) -> str:
    """返回图像所属 case 名称；普通 txt 数据源保持 train/val。."""
    p = Path(img_path)
    if p.parent.name == "images" and p.parent.parent.name:
        return p.parent.parent.name
    return "data"


def run_model_infer(
    weights: str,
    image_paths: list[str],
    model_cfg: str = "",
    imgsz: int = 1600,
    conf: float = 0.3,
    iou: float = 0.7,
    device: str = "0",
    batch: int = 16,
    overwrite: bool = False,
) -> None:
    """使用 Ultralytics 模型推理并写入 labels_10cls YOLO 标注。.

    参考 infer.py：若提供 model_cfg，则先用 yaml 配置构建模型再 load 权重， 以保证 10 类 timer 模型结构正确。
    """
    todo = []
    for img_path in image_paths:
        label_path = get_label_path(img_path)
        if overwrite or not os.path.exists(label_path):
            todo.append(img_path)
    if not todo:
        print("[INFO] 所有 label 已存在，跳过模型推理")
        return

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from ultralytics import YOLO

    print(f"[INFO] 开始模型推理: cfg={model_cfg or '-'}, weights={weights}, images={len(todo)}, overwrite={overwrite}")
    if model_cfg:
        model = YOLO(model_cfg)
        model.load(weights)
    else:
        model = YOLO(weights)
    done = 0
    batch = max(1, int(batch))
    for start in range(0, len(todo), batch):
        chunk = todo[start : start + batch]
        results = model.predict(
            source=chunk,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            batch=batch,
            stream=False,
            verbose=False,
        )
        for img_path, result in zip(chunk, results):
            label_path = get_label_path(img_path)
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            with open(label_path, "w") as f:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
                    xywhn = boxes.xywhn.detach().cpu().numpy()
                    f.writelines(
                        f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"
                        for cls_id, (cx, cy, bw, bh) in zip(cls_ids, xywhn)
                    )
            done += 1
        if done % 50 == 0 or done == len(todo):
            print(f"[INFO] 推理写标注进度: {done}/{len(todo)}")


@lru_cache(maxsize=30)
def render_image(img_path: str) -> tuple[str, int, int, int, int]:
    """返回原始图像（不画框）+ 原始尺寸 + 显示尺寸."""
    if not os.path.exists(img_path):
        img = Image.new("RGB", (640, 360), (30, 30, 30))
        ImageDraw.Draw(img).text((20, 160), f"图像不存在:\n{img_path}", fill=(255, 80, 80))
    else:
        img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode(), orig_w, orig_h, orig_w, orig_h


def read_boxes(label_path: str, img_w: int, img_h: int) -> list:
    """读取 YOLO 标注，返回像素坐标 + 归一化坐标."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            name = CLASS_NAMES.get(cls_id, f"cls{cls_id}")
            boxes.append(
                {
                    "cls_id": cls_id,
                    "name": name,
                    "color": "#{:02x}{:02x}{:02x}".format(*color),
                    "cx": cx,
                    "cy": cy,
                    "bw": bw,
                    "bh": bh,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
    return boxes


# ──────────────────────────────────────────────────────────────
# 全局状态
# ──────────────────────────────────────────────────────────────
train_paths: list[str] = []
val_paths: list[str] = []
records: list[tuple[str, str]] = []
record_index: dict = {}
total = 0
# status: {user_name: {img_path(str): {"status": "accept"/"reject", "ts": float}}}
status: dict = {}
users: dict = {}  # name -> {color, idx, last_seen}
lock = threading.Lock()


def configure_dataset(
    train_txt: str = DEFAULT_TRAIN_TXT, val_txt: str = DEFAULT_VAL_TXT, data_root: str = "", progress_file: str = ""
):
    """初始化数据列表；data_root 优先，未传则沿用 train/val txt。."""
    global TRAIN_TXT, VAL_TXT, PROGRESS_FILE, train_paths, val_paths, records, record_index, total, status, users
    TRAIN_TXT = train_txt
    VAL_TXT = val_txt
    if progress_file:
        PROGRESS_FILE = progress_file

    status = {}
    users = {}
    if data_root:
        train_paths = discover_image_paths(data_root)
        val_paths = []
        records = [(p, source_name(p)) for p in train_paths]
    else:
        train_paths = [normalize_path(p) for p in load_txt(TRAIN_TXT)]
        val_paths = [normalize_path(p) for p in load_txt(VAL_TXT)]
        records = [(p, "train") for p in train_paths] + [(p, "val") for p in val_paths]
    total = len(records)
    record_index = {p: i for i, (p, _) in enumerate(records)}
    render_image.cache_clear()
    load_progress()
    print(f"[INFO] 加载完成: train={len(train_paths)}, val={len(val_paths)}, total={total}")


def _user_stats(user_name: str):
    """返回某用户的 accepted/rejected/reviewed 数."""
    ustat = status.get(user_name, {})
    accepted = sum(1 for p, v in ustat.items() if p in record_index and v["status"] == "accept")
    rejected = sum(1 for p, v in ustat.items() if p in record_index and v["status"] == "reject")
    return accepted, rejected, accepted + rejected


def _global_stats():
    """返回全局（所有用户合并，同一帧多人标了取最新那条）的统计."""
    merged = {}  # img_path -> {status, ts}
    for ustat in status.values():
        for img_path, v in ustat.items():
            if img_path not in record_index:
                continue
            if img_path not in merged or v["ts"] > merged[img_path]["ts"]:
                merged[img_path] = v
    accepted = sum(1 for v in merged.values() if v["status"] == "accept")
    rejected = sum(1 for v in merged.values() if v["status"] == "reject")
    return accepted, rejected, accepted + rejected


def _path_from_progress_key(key: str) -> str:
    """兼容旧下标进度；新格式直接使用图像绝对路径。."""
    if isinstance(key, int):
        key = str(key)
    if key.isdigit():
        idx = int(key)
        return records[idx][0] if 0 <= idx < total else ""
    img_path = normalize_path(key)
    return img_path if img_path in record_index else ""


def restore_accepted_txt_progress(target_users=None):
    """按 accepted.txt 恢复已验收路径，避免旧 idx 进度被新增数据挤乱。."""
    accepted_paths = load_txt(DEFAULT_ACCEPTED_TXT)
    if not accepted_paths:
        return

    accepted_paths = [normalize_path(p) for p in accepted_paths]
    accepted_paths = [p for p in accepted_paths if p in record_index]
    if not accepted_paths:
        print(f"[WARN] {DEFAULT_ACCEPTED_TXT} 中没有命中当前数据集的路径")
        return

    users_to_restore = [u for u in (target_users or []) if u]
    if not users_to_restore:
        users_to_restore = list(status.keys()) or [RESTORED_USER]

    ts = os.path.getmtime(DEFAULT_ACCEPTED_TXT)
    restored = 0
    for uname in users_to_restore:
        ustat = status.setdefault(uname, {})
        for img_path in accepted_paths:
            old = ustat.get(img_path)
            if old is None or old.get("status") != "accept":
                ustat[img_path] = {"status": "accept", "ts": max(ts, old.get("ts", 0.0) if old else 0.0)}
                restored += 1
    print(f"[INFO] 按 accepted.txt 恢复进度: users={users_to_restore}, paths={len(accepted_paths)}, updated={restored}")


def load_progress():
    global status
    raw_users = []
    if not os.path.exists(PROGRESS_FILE):
        restore_accepted_txt_progress()
        return
    try:
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
        raw = data.get("status", {})
        raw_users = list(raw.keys())
        # 兼容旧格式：{str(idx): {status,user,ts}}  →  迁移到新格式
        # 新格式：{user_name: {img_path: {status,ts}}}
        if raw and all(k.isdigit() for k in raw):
            # 旧格式，迁移
            for k, v in raw.items():
                if isinstance(v, dict) and "user" in v:
                    uname = v.get("user", "unknown")
                    img_path = _path_from_progress_key(k)
                    if img_path:
                        status.setdefault(uname, {})[img_path] = {"status": v["status"], "ts": v.get("ts", 0.0)}
            print("[INFO] 已从旧格式迁移进度")
        else:
            # 用户分组格式；key 可能是旧 idx，也可能是新绝对路径
            for uname, ustat in raw.items():
                if not isinstance(ustat, dict):
                    continue
                digit_keys = [k for k in ustat if str(k).isdigit()]
                if digit_keys and len(digit_keys) == len(ustat) and os.path.exists(DEFAULT_ACCEPTED_TXT):
                    # 纯 idx 进度在数据前插后不可信，交给 accepted.txt 按绝对路径恢复。
                    status.setdefault(uname, {})
                    continue
                for k, v in ustat.items():
                    img_path = _path_from_progress_key(str(k))
                    if img_path and isinstance(v, dict) and v.get("status") in ("accept", "reject"):
                        status.setdefault(uname, {})[img_path] = {"status": v["status"], "ts": v.get("ts", 0.0)}
        restore_accepted_txt_progress(raw_users)
        total_frames = sum(len(u) for u in status.values())
        print(f"[INFO] 恢复进度: {len(status)} 位用户，共标注 {total_frames} 帧次")
    except Exception as e:
        print(f"[WARN] 恢复进度失败: {e}")
        restore_accepted_txt_progress(raw_users)


def save_progress():
    try:
        serial = {uname: {p: v for p, v in ustat.items() if p in record_index} for uname, ustat in status.items()}
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"status": serial}, f, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] 进度写盘失败: {e}")


def next_unreviewed(from_idx: int, user_name: str = "") -> int:
    """找该用户下一个未审帧；若 user_name 为空则看全局."""
    if user_name:
        ustat = status.get(user_name, {})
        for i in range(from_idx, total):
            if records[i][0] not in ustat:
                return i
    else:
        # 全局：所有用户都审过才算审过
        for i in range(from_idx, total):
            img_path = records[i][0]
            if not any(img_path in ustat for ustat in status.values()):
                return i
    return total - 1


# ──────────────────────────────────────────────────────────────
# Flask
# ──────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/join", methods=["POST"])
def join():
    name = (request.json or {}).get("name", "").strip()
    if not name:
        return jsonify({"ok": False, "msg": "用户名不能为空"}), 400
    with lock:
        if name not in users:
            color = USER_COLORS[len(users) % len(USER_COLORS)]
            users[name] = {"color": color, "idx": next_unreviewed(0, name), "last_seen": time.time()}
        else:
            users[name]["last_seen"] = time.time()
        u = users[name]
        accepted, rejected, reviewed = _user_stats(name)
        g_accepted, g_rejected, g_reviewed = _global_stats()
    return jsonify(
        {
            "ok": True,
            "name": name,
            "color": u["color"],
            "idx": u["idx"],
            "total": total,
            "accepted": accepted,
            "rejected": rejected,
            "reviewed": reviewed,
            "g_accepted": g_accepted,
            "g_rejected": g_rejected,
            "g_reviewed": g_reviewed,
        }
    )


@app.route("/frame")
def get_frame():
    if total <= 0:
        return jsonify({"ok": False, "msg": "没有可用图像，请检查数据路径"}), 404
    idx = int(request.args.get("idx", 0))
    name = request.args.get("user", "")
    prefetch = request.args.get("prefetch", "0") == "1"
    idx = max(0, min(idx, total - 1))
    if not prefetch:
        with lock:
            if name in users:
                users[name]["idx"] = idx
                users[name]["last_seen"] = time.time()
    img_path, source = records[idx]
    label_path = get_label_path(img_path)
    b64, orig_w, orig_h, disp_w, disp_h = render_image(img_path)
    boxes = read_boxes(label_path, orig_w, orig_h)
    with lock:
        # 当前用户对该帧的标注
        ustat = status.get(name, {})
        my_st = ustat.get(img_path)
        # 所有用户对该帧的标注（用于显示谁审过）
        frame_marks = {u: s.get(img_path) for u, s in status.items() if img_path in s}
        accepted, rejected, reviewed = _user_stats(name)
        g_accepted, g_rejected, g_reviewed = _global_stats()
        now = time.time()
        online = {n: {"color": u["color"], "idx": u["idx"]} for n, u in users.items() if now - u["last_seen"] < 60}
    return jsonify(
        {
            "idx": idx,
            "total": total,
            "source": source,
            "img_path": img_path,
            "label_path": label_path,
            "label_exists": os.path.exists(label_path),
            # 当前用户自己的标注状态
            "status": my_st["status"] if my_st else None,
            "marked_by": name if my_st else None,
            # 所有人对该帧的标注
            "frame_marks": {u: {"status": v["status"], "ts": v["ts"]} for u, v in frame_marks.items()},
            "image_b64": b64,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "disp_w": disp_w,
            "disp_h": disp_h,
            "boxes": boxes,
            "accepted": accepted,
            "rejected": rejected,
            "reviewed": reviewed,
            "g_accepted": g_accepted,
            "g_rejected": g_rejected,
            "g_reviewed": g_reviewed,
            "online_users": online,
        }
    )


@app.route("/touch", methods=["POST"])
def touch():
    """预加载命中后，轻量更新用户位置并返回最新状态（无图片）."""
    data = request.json or {}
    name = data.get("user", "")
    idx = int(data.get("idx", 0))
    idx = max(0, min(idx, total - 1)) if total else 0
    img_path = records[idx][0] if total else ""
    with lock:
        if name in users:
            users[name]["idx"] = idx
            users[name]["last_seen"] = time.time()
        ustat = status.get(name, {})
        my_st = ustat.get(img_path)
        frame_marks = {u: s.get(img_path) for u, s in status.items() if img_path in s}
        accepted, rejected, reviewed = _user_stats(name)
        g_accepted, g_rejected, g_reviewed = _global_stats()
        now = time.time()
        online = {n: {"color": u["color"], "idx": u["idx"]} for n, u in users.items() if now - u["last_seen"] < 60}
    return jsonify(
        {
            "status": my_st["status"] if my_st else None,
            "frame_marks": {u: {"status": v["status"], "ts": v["ts"]} for u, v in frame_marks.items()},
            "accepted": accepted,
            "rejected": rejected,
            "reviewed": reviewed,
            "g_accepted": g_accepted,
            "g_rejected": g_rejected,
            "g_reviewed": g_reviewed,
            "online_users": online,
        }
    )


@app.route("/mark", methods=["POST"])
def mark():
    data = request.json or {}
    idx = int(data.get("idx", 0))
    st = data.get("status")
    name = data.get("user", "?")
    if idx < 0 or idx >= total:
        return jsonify({"ok": False, "msg": "idx out of range"}), 400
    img_path = records[idx][0]
    with lock:
        if name not in status:
            status[name] = {}
        if st in ("accept", "reject"):
            status[name][img_path] = {"status": st, "ts": time.time()}
        elif st is None:
            status[name].pop(img_path, None)
        save_progress()
    return jsonify({"ok": True, "idx": idx, "status": st, "user": name})


@app.route("/next_unreviewed")
def api_next_unreviewed():
    from_idx = int(request.args.get("from", 0))
    user_name = request.args.get("user", "")
    with lock:
        nxt = next_unreviewed(from_idx, user_name)
    return jsonify({"idx": nxt})


@app.route("/sync")
def sync():
    since = float(request.args.get("since", 0))
    user_name = request.args.get("user", "")
    with lock:
        # 只推送该用户自己的变更
        ustat = status.get(user_name, {})
        changes = {str(record_index[p]): v for p, v in ustat.items() if p in record_index and v["ts"] > since}
        # 也推送其他人对同一帧的标注（让前端可以显示他人状态）
        frame_changes = {}
        for uname, ustats in status.items():
            for img_path, v in ustats.items():
                if img_path in record_index and v["ts"] > since:
                    sk = str(record_index[img_path])
                    if sk not in frame_changes:
                        frame_changes[sk] = {}
                    frame_changes[sk][uname] = {"status": v["status"], "ts": v["ts"]}
        now = time.time()
        online = {n: {"color": u["color"], "idx": u["idx"]} for n, u in users.items() if now - u["last_seen"] < 60}
        accepted, rejected, reviewed = _user_stats(user_name)
        g_accepted, g_rejected, g_reviewed = _global_stats()
    return jsonify(
        {
            "ts": time.time(),
            "changes": changes,
            "frame_changes": frame_changes,
            "online_users": online,
            "accepted": accepted,
            "rejected": rejected,
            "reviewed": reviewed,
            "g_accepted": g_accepted,
            "g_rejected": g_rejected,
            "g_reviewed": g_reviewed,
        }
    )


@app.route("/reset", methods=["POST"])
def reset_all():
    """清除所有标注进度，重新开始."""
    global status
    user = (request.json or {}).get("user", "")
    with lock:
        if user:
            # 只清除指定用户
            status.pop(user, None)
            msg = f"已清除 {user} 的所有标注"
        else:
            # 清除所有人
            status = {}
            msg = "已清除所有用户的标注记录"
        save_progress()
    print(f"[INFO] {msg}")
    return jsonify({"ok": True, "msg": msg})


@app.route("/save_label", methods=["POST"])
def save_label():
    """保存修改后的标注框到 label 文件."""
    data = request.json or {}
    idx = int(data.get("idx", 0))
    boxes = data.get("boxes", [])
    if idx < 0 or idx >= total:
        return jsonify({"ok": False, "msg": "idx out of range"}), 400
    img_path, _ = records[idx]
    label_path = get_label_path(img_path)
    try:
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        with open(label_path, "w") as f:
            f.writelines(f"{b['cls_id']} {b['cx']} {b['cy']} {b['bw']} {b['bh']}\n" for b in boxes)
        return jsonify({"ok": True, "label_path": label_path, "saved_boxes": len(boxes)})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)}), 500


@app.route("/save", methods=["POST"])
def save():
    data = request.json or {}
    out_path = data.get("out_path", "/pcpt/pcpt/project/atlb/vla/ultralytics/dataset/accepted.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base, ext = os.path.splitext(out_path)

    saved = []
    all_accepted = []
    all_rejected = []
    with lock:
        # 先按用户各自输出
        for uname, ustat in status.items():
            u_accepted = [p for p, _ in records if p in ustat and ustat[p]["status"] == "accept"]
            u_rejected = [p for p, _ in records if p in ustat and ustat[p]["status"] == "reject"]
            safe_name = uname.replace("/", "_").replace(" ", "_")
            if u_accepted:
                p = f"{base}_{safe_name}{ext}"
                with open(p, "w") as f:
                    f.write("\n".join(u_accepted) + "\n")
                saved.append(f"accept[{uname}]: {len(u_accepted)} 条 → {p}")
            if u_rejected:
                p = f"{base.replace('accepted', 'rejected')}_{safe_name}{ext}"
                with open(p, "w") as f:
                    f.write("\n".join(u_rejected) + "\n")
                saved.append(f"reject[{uname}]: {len(u_rejected)} 条 → {p}")

        # 也生成一份合并文件（同一帧多人都标了取最新）
        merged = {}
        for ustat in status.values():
            for img_path, v in ustat.items():
                if img_path not in record_index:
                    continue
                if img_path not in merged or v["ts"] > merged[img_path]["ts"]:
                    merged[img_path] = v
        all_accepted = [p for p, _ in records if p in merged and merged[p]["status"] == "accept"]
        all_rejected = [p for p, _ in records if p in merged and merged[p]["status"] == "reject"]
        if all_accepted:
            with open(out_path, "w") as f:
                f.write("\n".join(all_accepted) + "\n")
            saved.append(f"accept[合并]: {len(all_accepted)} 条 → {out_path}")
        reject_path = base.replace("accepted", "rejected") + ext
        if all_rejected:
            with open(reject_path, "w") as f:
                f.write("\n".join(all_rejected) + "\n")
            saved.append(f"reject[合并]: {len(all_rejected)} 条 → {reject_path}")

    return jsonify({"ok": True, "saved": saved, "count": len(all_accepted) + len(all_rejected)})


# ──────────────────────────────────────────────────────────────
# HTML
# ──────────────────────────────────────────────────────────────
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>GT Viewer — 10cls</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#1a1a1a; color:#ddd; font-family:'Courier New',monospace;
       height:100vh; display:flex; flex-direction:column; overflow:hidden; }

/* 登录 */
#login-mask {
  position:fixed; inset:0; background:rgba(0,0,0,.8);
  display:flex; align-items:center; justify-content:center; z-index:999;
}
#login-box {
  background:#252526; border:1px solid #444; border-radius:10px;
  padding:36px 40px; min-width:320px; text-align:center;
}
#login-box h2 { margin-bottom:20px; font-size:18px; color:#eee; }
#login-name {
  width:100%; padding:10px 14px; font-size:15px;
  background:#3a3d41; color:#eee; border:1px solid #555;
  border-radius:6px; margin-bottom:14px;
}
#login-btn {
  width:100%; padding:10px; font-size:15px; font-weight:bold;
  background:#005f87; color:#fff; border:none; border-radius:6px; cursor:pointer;
}
#login-btn:hover { background:#0077aa; }

/* 顶栏 */
#topbar {
  background:#252526; padding:5px 10px;
  display:flex; align-items:center; gap:8px; flex-shrink:0;
  border-bottom:1px solid #333;
}
#user-badge { font-size:12px; font-weight:bold; padding:3px 10px;
  border-radius:12px; background:#333; white-space:nowrap; border:1px solid; }
#src-badge  { font-size:12px; font-weight:bold; padding:3px 8px;
  border-radius:4px; background:#3a3d41; color:#ffaa00; white-space:nowrap; }
#path-info  { flex:1; font-size:11px; color:#999; overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap; }
#stat-info  { font-size:12px; color:#88cc88; white-space:nowrap; }

/* 在线用户 */
#users-bar {
  background:#1e1e2e; padding:3px 10px; display:flex;
  align-items:center; gap:8px; flex-shrink:0; min-height:26px;
  border-bottom:1px solid #2a2a3a; flex-wrap:wrap;
}
.u-chip { font-size:11px; padding:2px 8px; border-radius:10px;
  background:#2a2a3a; border:1px solid; }
.u-dot  { display:inline-block; width:8px; height:8px;
  border-radius:50%; margin-right:4px; vertical-align:middle; }

/* 进度条 */
#progress-wrap { height:3px; background:#333; flex-shrink:0; }
#progress-bar  { height:100%; background:#005f87; transition:width .2s; }

/* 主体 */
#main { flex:1; display:flex; overflow:hidden; }

#canvas-wrap {
  flex:1; position:relative; background:#111; overflow:hidden;
}
#main-canvas { display:block; width:100%; height:100%; cursor:default; }
#main-canvas.selectable { cursor:pointer; }

/* 状态角标 */
#overlay {
  position:absolute; top:10px; right:10px;
  font-size:16px; font-weight:bold; padding:5px 14px;
  border-radius:6px; display:none; pointer-events:none;
}
.ov-accept { background:rgba(0,140,0,.85); color:#aaffaa; }
.ov-reject { background:rgba(140,0,0,.85); color:#ffaaaa; }
#no-label {
  position:absolute; top:10px; left:10px;
  background:rgba(180,100,0,.85); color:#fff; padding:4px 12px;
  border-radius:6px; font-size:13px; display:none; pointer-events:none;
}
#modified-badge {
  position:absolute; bottom:10px; left:10px;
  background:rgba(200,140,0,.9); color:#fff; padding:4px 12px;
  border-radius:6px; font-size:12px; display:none; pointer-events:none;
}

/* 侧栏 */
#sidebar {
  width:270px; background:#252526; display:flex; flex-direction:column;
  border-left:1px solid #333; flex-shrink:0; overflow:hidden;
}
#sidebar-top { flex:1; overflow-y:auto; padding:10px; }
#sidebar-bottom { padding:8px 10px; border-top:1px solid #333; flex-shrink:0; }
#sidebar h3 { font-size:12px; color:#888; margin-bottom:6px; }

.box-item {
  border-left:4px solid #555; padding:5px 8px; margin-bottom:5px;
  background:#2d2d2d; border-radius:3px; font-size:11px; line-height:1.6;
  cursor:pointer; transition:background .1s;
}
.box-item:hover   { background:#363636; }
.box-item.selected { background:#3a3020; outline:1px solid #ffaa00; }

.box-edit {
  margin-top:6px; padding:6px; background:#1e1e1e;
  border-radius:4px; display:flex; flex-direction:column; gap:5px;
}
.cls-select {
  width:100%; padding:4px 6px; background:#3a3d41; color:#eee;
  border:1px solid #555; border-radius:4px; font-size:12px;
}
.btn-del {
  padding:4px 0; background:#7e1e1e; color:#fff; border:none;
  border-radius:4px; cursor:pointer; font-size:12px; font-weight:bold; width:100%;
}
.btn-del:hover { background:#dc3545; }
.btn-save-label {
  width:100%; padding:6px; background:#2d5a1a; color:#aaffaa;
  border:none; border-radius:4px; cursor:pointer;
  font-size:12px; font-weight:bold; margin-top:4px;
}
.btn-save-label:hover { background:#3a7a22; }
.btn-save-label.modified { background:#7a5a00; color:#ffe; }
.btn-save-label.modified:hover { background:#a07800; }

.path-block { color:#666; word-break:break-all; font-size:10px;
  line-height:1.5; margin-top:4px; }

/* 底部 */
#bottombar {
  background:#252526; padding:6px 10px; display:flex;
  align-items:center; gap:6px; flex-shrink:0; border-top:1px solid #333;
  flex-wrap:wrap;
}
button {
  padding:5px 12px; border:none; border-radius:4px; cursor:pointer;
  font-size:13px; font-weight:bold; transition:opacity .15s;
}
button:hover { opacity:.85; }
.btn-nav      { background:#3a3d41; color:#ddd; }
.btn-accept   { background:#1a6e2e; color:#fff; }
.btn-reject   { background:#6e1a1a; color:#fff; }
.btn-clear    { background:#4a3800; color:#ffcc55; border:1px solid #7a6000; display:none; }
.btn-clear:hover { background:#6a5200 !important; opacity:1 !important; }
.btn-next-new { background:#2a4a2a; color:#9fc; }
.btn-save-res { background:#005f87; color:#fff; }
.btn-reset    { background:#5a1a1a; color:#ffaaaa; border:1px solid #8a2a2a; }
.btn-active-accept { background:#28a745!important; box-shadow:0 0 0 2px #5dde7e; }
.btn-active-reject { background:#dc3545!important; box-shadow:0 0 0 2px #ff8888; }

#jump-wrap { display:flex; align-items:center; gap:4px; }
#jump-input {
  width:64px; background:#3a3d41; color:#ddd; border:1px solid #555;
  border-radius:4px; padding:4px 6px; font-size:13px;
}
#out-path-input {
  flex:1; min-width:180px; background:#3a3d41; color:#aaa;
  border:1px solid #555; border-radius:4px; padding:4px 7px; font-size:11px;
}

#toast {
  position:fixed; bottom:56px; left:50%; transform:translateX(-50%);
  background:#333; color:#eee; padding:7px 18px; border-radius:6px;
  font-size:13px; display:none; z-index:999; pointer-events:none;
}
</style>
</head>
<body>

<!-- 登录 -->
<div id="login-mask">
  <div id="login-box">
    <h2>🔍 GT Viewer — 请输入你的名字</h2>
    <input id="login-name" type="text" placeholder="例如：Alice" autofocus
           onkeydown="if(event.key==='Enter')doLogin()"/>
    <button id="login-btn" onclick="doLogin()">进入</button>
  </div>
</div>

<!-- 顶栏 -->
<div id="topbar">
  <span id="user-badge">-</span>
  <span id="src-badge">-</span>
  <span id="path-info">Loading...</span>
  <span id="stat-info">-</span>
</div>
<div id="users-bar"><span style="color:#555;font-size:11px">在线用户:</span></div>
<div id="progress-wrap"><div id="progress-bar" style="width:0"></div></div>

<!-- 主体 -->
<div id="main">
  <div id="canvas-wrap">
    <canvas id="main-canvas"></canvas>
    <div id="overlay"></div>
    <div id="no-label">⚠ label 文件不存在</div>
    <div id="modified-badge">● 标注已修改，未保存</div>
  </div>

  <div id="sidebar">
    <div id="sidebar-top">
      <h3 id="box-count">标注框 (0)</h3>
      <div id="box-list"></div>
      <hr style="border-color:#333;margin:10px 0">
      <h3>路径</h3>
      <div id="img-path"   class="path-block"></div>
      <div id="label-path" class="path-block"></div>
    </div>
    <div id="sidebar-bottom">
      <div style="margin-bottom:6px;">
        <div style="font-size:11px;color:#888;margin-bottom:4px;">拉框默认类别</div>
        <select id="new-cls-select" class="cls-select" onchange="defaultNewCls=parseInt(this.value)">
          <option value="0">0: RedLight</option>
          <option value="1">1: GreenLight</option>
          <option value="2">2: YellowLight</option>
          <option value="3">3: BlackLight</option>
          <option value="4">4: MovingTrafficLight</option>
          <option value="5">5: MultiSemanticTrafficLights</option>
          <option value="6">6: BlackTimingCard</option>
          <option value="7">7: GreenTimingCard</option>
          <option value="8">8: RedTimingCard</option>
          <option value="9">9: YellowTimingCard</option>
        </select>
      </div>
      <button class="btn-save-label" id="btn-save-label" onclick="saveLabelFile()">
        💾 保存当前帧标注
      </button>
    </div>
  </div>
</div>

<!-- 底部 -->
<div id="bottombar">
  <button class="btn-nav"      onclick="go(-1)">◀ 上一帧</button>
  <button class="btn-nav"      onclick="go(+1)">下一帧 ▶</button>
  <button class="btn-next-new" onclick="jumpNextNew()" title="N">⏭ 下一未审帧</button>
  <div id="jump-wrap">
    <span style="font-size:11px;color:#888">跳转</span>
    <input id="jump-input" type="number" min="1"
           onkeydown="if(event.key==='Enter')jumpTo()"/>
    <button class="btn-nav" onclick="jumpTo()">Go</button>
  </div>
  <button id="btn-accept" class="btn-accept" onclick="doMark('accept')">✔ Accept</button>
  <button id="btn-reject" class="btn-reject" onclick="doMark('reject')">✘ Reject</button>
  <button id="btn-clear"  class="btn-clear"  onclick="doClear()"       title="取消当前标注 (C)">✖ 取消</button>
  <span style="flex:1"></span>
  <input id="out-path-input" type="text"
         value="/pcpt/pcpt/project/atlb/vla/ultralytics/dataset/accepted.txt"/>
  <button class="btn-save-res" onclick="saveResults()">💾 保存结果</button>
  <button class="btn-reset"    onclick="resetRecords()" title="清除所有标注记录，重新开始">🗑 清除记录</button>
</div>

<div id="toast"></div>

<script>
// ── 常量（与后端 CLASS_NAMES / CLASS_COLORS 对应）──
const CLASS_NAMES_JS = {
  0:"RedLight", 1:"GreenLight", 2:"YellowLight",
  3:"BlackLight", 4:"MovingTrafficLight", 5:"MultiSemanticTrafficLights",
  6:"BlackTimingCard", 7:"GreenTimingCard", 8:"RedTimingCard",
  9:"YellowTimingCard"
};
const CLASS_COLORS_JS = {
  0:"#dc3232", 1:"#32c832", 2:"#dcb400",
  3:"#828282", 4:"#3296ff", 5:"#c832c8",
  6:"#505050", 7:"#00a05a", 8:"#b42828", 9:"#f0c828"
};

// ── 状态 ──
let ME = "", ME_COLOR = "";
let curIdx = 0, curTotal = 1;
let syncTs = 0;
let curFrameMarks = {};  // {user: {status, ts}} — 当前帧所有人的标注

const frameCache = new Map(); // idx -> Promise<frameData>

function prefetchFrame(idx) {
  if (idx < 0 || idx >= curTotal || frameCache.has(idx)) return;
  frameCache.set(idx,
    fetch(`/frame?idx=${idx}&user=${encodeURIComponent(ME)}&prefetch=1`)
      .then(r => r.json()).catch(() => null)
  );
}

// Canvas & 标注框
const canvas  = document.getElementById('main-canvas');
const ctx     = canvas.getContext('2d');
let imgEl     = new Image();
let origW=0, origH=0, dispW=0, dispH=0;
let canvasBoxes  = [];   // [{cls_id,name,color,cx,cy,bw,bh,x1,y1,x2,y2}]
let selectedBox  = -1;
let labelModified = false;

// 拉框状态
let drawing = false, drawStart = null, drawCurrent = null;
let defaultNewCls = 0;

// transform cache
let cvScale=1, cvOffX=0, cvOffY=0;

// ── Canvas 尺寸 ──────────────────────────────────────────────
function resizeCanvas() {
  const wrap = document.getElementById('canvas-wrap');
  canvas.width  = wrap.clientWidth;
  canvas.height = wrap.clientHeight;
  drawCanvas();
}
new ResizeObserver(resizeCanvas).observe(document.getElementById('canvas-wrap'));

// ── 绘制 ─────────────────────────────────────────────────────
function computeTransform() {
  const cw = canvas.width, ch = canvas.height;
  if (!dispW || !dispH) return;
  const s = Math.min(cw/dispW, ch/dispH);
  cvOffX  = (cw - dispW*s) / 2;
  cvOffY  = (ch - dispH*s) / 2;
  cvScale = s * dispW / origW;   // scale: orig-px → canvas-px
}

function drawCanvas() {
  const cw = canvas.width, ch = canvas.height;
  ctx.fillStyle = '#111';
  ctx.fillRect(0, 0, cw, ch);
  if (!imgEl.complete || !imgEl.src) return;

  computeTransform();
  const imgDrawW = dispW * (cvOffX === 0
    ? (ch - 0) / dispH
    : (cw - 0) / dispW);
  // 重新算（更稳健）
  const s2 = Math.min(cw/dispW, ch/dispH);
  const dw = dispW*s2, dh = dispH*s2;
  const ox = (cw-dw)/2, oy = (ch-dh)/2;
  ctx.drawImage(imgEl, ox, oy, dw, dh);
  cvScale = s2 * dispW / origW;
  cvOffX  = ox; cvOffY = oy;

  // 绘制框
  canvasBoxes.forEach((b, i) => {
    const bx1 = b.x1*cvScale + cvOffX;
    const by1 = b.y1*cvScale + cvOffY;
    const bx2 = b.x2*cvScale + cvOffX;
    const by2 = b.y2*cvScale + cvOffY;
    const sel = i === selectedBox;

    ctx.strokeStyle = sel ? '#ffdd00' : b.color;
    ctx.lineWidth   = sel ? 3 : 2;
    if (sel) { ctx.setLineDash([6,3]); } else { ctx.setLineDash([]); }
    ctx.strokeRect(bx1, by1, bx2-bx1, by2-by1);
    ctx.setLineDash([]);

  });

  // 拉框橡皮筋预览
  if (drawing && drawStart && drawCurrent) {
    const bx1 = drawStart.ix  * cvScale + cvOffX;
    const by1 = drawStart.iy  * cvScale + cvOffY;
    const bx2 = drawCurrent.ix * cvScale + cvOffX;
    const by2 = drawCurrent.iy * cvScale + cvOffY;
    ctx.strokeStyle = '#ffff00';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1);
    ctx.setLineDash([]);
  }
}

imgEl.onload = drawCanvas;

// ── 坐标转换 & 查找框 ────────────────────────────────────────
function evToImg(e) {
  const rect = canvas.getBoundingClientRect();
  const mx = (e.clientX - rect.left) * (canvas.width  / rect.width);
  const my = (e.clientY - rect.top)  * (canvas.height / rect.height);
  return { ix: (mx - cvOffX) / cvScale, iy: (my - cvOffY) / cvScale };
}
function findBoxAt(ix, iy) {
  for (let i = canvasBoxes.length - 1; i >= 0; i--) {
    const b = canvasBoxes[i];
    if (ix >= b.x1 && ix <= b.x2 && iy >= b.y1 && iy <= b.y2) return i;
  }
  return -1;
}

// ── Canvas 拉框 & 选框 ───────────────────────────────────────
canvas.addEventListener('mousemove', e => {
  const { ix, iy } = evToImg(e);
  if (drawing) {
    drawCurrent = { ix, iy };
    drawCanvas();
  } else {
    canvas.style.cursor = findBoxAt(ix, iy) >= 0 ? 'pointer' : 'crosshair';
  }
});

canvas.addEventListener('mousedown', e => {
  if (e.button !== 0) return;
  const { ix, iy } = evToImg(e);
  const found = findBoxAt(ix, iy);
  if (found >= 0) {
    selectedBox = (found === selectedBox) ? -1 : found;
    drawCanvas(); renderBoxPanel();
    return;
  }
  // 空白区域 → 开始拉框
  drawing = true;
  drawStart = { ix, iy };
  drawCurrent = { ix, iy };
  selectedBox = -1;
  canvas.style.cursor = 'crosshair';
});

canvas.addEventListener('mouseup', e => {
  if (!drawing) return;
  drawing = false;
  const { ix, iy } = evToImg(e);
  const x1 = Math.round(Math.min(drawStart.ix, ix));
  const y1 = Math.round(Math.min(drawStart.iy, iy));
  const x2 = Math.round(Math.max(drawStart.ix, ix));
  const y2 = Math.round(Math.max(drawStart.iy, iy));
  drawStart = null; drawCurrent = null;
  if (x2 - x1 > 4 && y2 - y1 > 4) {
    const cx = (x1 + x2) / 2 / origW;
    const cy = (y1 + y2) / 2 / origH;
    const bw = (x2 - x1) / origW;
    const bh = (y2 - y1) / origH;
    const cls = defaultNewCls;
    canvasBoxes.push({
      cls_id: cls, name: CLASS_NAMES_JS[cls], color: CLASS_COLORS_JS[cls],
      cx, cy, bw, bh, x1, y1, x2, y2
    });
    selectedBox = canvasBoxes.length - 1;
    setModified(true);
    renderBoxPanel();
  }
  drawCanvas();
});

// 鼠标移出窗口时取消拉框
window.addEventListener('mouseup', () => {
  if (!drawing) return;
  drawing = false; drawStart = null; drawCurrent = null; drawCanvas();
});

// ── 框列表侧栏 ───────────────────────────────────────────────
function renderBoxPanel() {
  document.getElementById('box-count').textContent = `标注框 (${canvasBoxes.length})`;
  const bl = document.getElementById('box-list');
  bl.innerHTML = '';

  canvasBoxes.forEach((b, i) => {
    const div = document.createElement('div');
    div.className = 'box-item' + (i === selectedBox ? ' selected' : '');
    div.style.borderLeftColor = b.color;
    div.onclick = () => {
      selectedBox = (selectedBox === i) ? -1 : i;
      drawCanvas(); renderBoxPanel();
    };

    let editHtml = '';
    if (i === selectedBox) {
      const opts = Object.entries(CLASS_NAMES_JS).map(([id,name]) =>
        `<option value="${id}" ${parseInt(id)===b.cls_id?'selected':''}>${id}: ${name}</option>`
      ).join('');
      editHtml = `
        <div class="box-edit">
          <select class="cls-select" onchange="changeClass(${i}, this.value)">
            ${opts}
          </select>
          <button class="btn-del" onclick="deleteBox(event,${i})">🗑 删除此框</button>
        </div>`;
    }

    div.innerHTML = `
      <b style="color:${b.color}">[${b.cls_id}] ${b.name}</b>
      <span style="color:#666;float:right">${b.x2-b.x1}×${b.y2-b.y1}</span><br>
      <span style="color:#555">(${b.x1},${b.y1}) → (${b.x2},${b.y2})</span>
      ${editHtml}`;
    bl.appendChild(div);
  });

  // 保存按钮样式
  const btn = document.getElementById('btn-save-label');
  btn.className = 'btn-save-label' + (labelModified ? ' modified' : '');
  btn.textContent = labelModified ? '💾 保存标注 (有修改)' : '💾 保存当前帧标注';
}

function changeClass(boxIdx, newId) {
  newId = parseInt(newId);
  const b = canvasBoxes[boxIdx];
  b.cls_id = newId;
  b.name   = CLASS_NAMES_JS[newId]  || `cls${newId}`;
  b.color  = CLASS_COLORS_JS[newId] || '#ffffff';
  setModified(true);
  drawCanvas(); renderBoxPanel();
}

function deleteBox(e, boxIdx) {
  e.stopPropagation();
  canvasBoxes.splice(boxIdx, 1);
  if (selectedBox >= canvasBoxes.length) selectedBox = canvasBoxes.length - 1;
  setModified(true);
  drawCanvas(); renderBoxPanel();
}

function setModified(v) {
  labelModified = v;
  document.getElementById('modified-badge').style.display = v ? 'block' : 'none';
}

async function saveLabelFile() {
  const boxes = canvasBoxes.map(b => ({
    cls_id: b.cls_id, cx: b.cx, cy: b.cy, bw: b.bw, bh: b.bh
  }));
  const r = await fetch('/save_label', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({idx:curIdx, boxes})
  });
  const d = await r.json();
  if (d.ok) {
    setModified(false);
    renderBoxPanel();
    toast(`✅ 标注已写盘 (${d.saved_boxes} 框)`);
  } else {
    toast('❌ 保存失败: ' + d.msg);
  }
}

// ── 登录 ─────────────────────────────────────────────────────
async function doLogin() {
  const name = document.getElementById('login-name').value.trim();
  if (!name) { alert("请输入用户名"); return; }
  const r = await fetch('/join',{
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({name})
  });
  const d = await r.json();
  if (!d.ok) { alert(d.msg); return; }
  ME = d.name; ME_COLOR = d.color;
  document.getElementById('login-mask').style.display = 'none';
  document.getElementById('user-badge').textContent   = '👤 '+ME;
  document.getElementById('user-badge').style.color   = ME_COLOR;
  document.getElementById('user-badge').style.borderColor = ME_COLOR;
  curTotal = d.total;
  resizeCanvas();
  await fetchFrame(d.idx);
  setInterval(doSync, 2000);
}

// ── 获取帧 ───────────────────────────────────────────────────
async function fetchFrame(idx) {
  if (labelModified) {
    if (!confirm('当前帧标注有未保存修改，确认离开？')) return;
    setModified(false);
  }
  idx = Math.max(0, Math.min(curTotal-1, idx));

  // 检查预加载缓存
  const cached = frameCache.get(idx);
  frameCache.delete(idx);

  let d;
  if (cached) {
    d = await cached;
    if (!d) {
      // 预加载失败，正常请求
      const r = await fetch(`/frame?idx=${idx}&user=${encodeURIComponent(ME)}`);
      d = await r.json();
    } else {
      // 命中缓存：后台轻量更新位置 & 刷新状态
      fetch('/touch', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({user: ME, idx})
      }).then(r => r.json()).then(fresh => {
        renderOverlay(fresh.status, fresh.frame_marks || {});
        renderBtnState(fresh.status);
        updateStat(fresh.accepted, fresh.rejected, fresh.reviewed,
                   fresh.g_accepted, fresh.g_rejected, fresh.g_reviewed, curTotal);
        renderUsers(fresh.online_users);
        curFrameMarks = fresh.frame_marks || {};
      }).catch(() => {});
    }
  } else {
    const r = await fetch(`/frame?idx=${idx}&user=${encodeURIComponent(ME)}`);
    d = await r.json();
  }
  curIdx = d.idx; curTotal = d.total;

  origW = d.orig_w; origH = d.orig_h;
  dispW = d.disp_w; dispH = d.disp_h;
  canvasBoxes  = d.boxes.map(b => Object.assign({}, b));
  selectedBox  = -1;
  setModified(false);
  curFrameMarks = d.frame_marks || {};

  imgEl.src = 'data:image/jpeg;base64,' + d.image_b64;

  // 顶栏
  document.getElementById('src-badge').textContent  = '['+d.source.toUpperCase()+']';
  document.getElementById('path-info').textContent  = `[${d.idx+1}/${d.total}]  ${d.img_path}`;
  document.getElementById('jump-input').value = d.idx + 1;
  document.getElementById('progress-bar').style.width = ((d.idx+1)/d.total*100)+'%';
  updateStat(d.accepted, d.rejected, d.reviewed, d.g_accepted, d.g_rejected, d.g_reviewed, d.total);

  // 覆盖层（只显示当前用户自己的标注状态，侧边显示他人）
  renderOverlay(d.status, curFrameMarks);
  renderBtnState(d.status);

  // label 缺失
  document.getElementById('no-label').style.display = d.label_exists ? 'none' : 'block';

  // 路径
  document.getElementById('img-path').textContent   = '📷 '+d.img_path;
  document.getElementById('label-path').textContent = '🏷 '+d.label_path;

  renderBoxPanel();
  renderUsers(d.online_users);
  // 预加载后续 2 帧
  setTimeout(() => prefetchFrame(idx + 1), 80);
  setTimeout(() => prefetchFrame(idx + 2), 350);
}

// ── 覆盖层 / 按钮状态 ────────────────────────────────────────
function renderOverlay(mySt, frameMarks) {
  const ov = document.getElementById('overlay');
  ov.style.display='none'; ov.className='';
  if (mySt==='accept') {
    ov.textContent='✔ ACCEPTED (你)';
    ov.className='ov-accept'; ov.style.display='block';
  } else if (mySt==='reject') {
    ov.textContent='✘ REJECTED (你)';
    ov.className='ov-reject'; ov.style.display='block';
  }
  // 显示其他人对该帧的标注（在 users-bar 下方追加小标签）
  renderFrameMarks(frameMarks);
}
function renderFrameMarks(frameMarks) {
  // 在 no-label 旁边显示他人标注情况
  let el = document.getElementById('other-marks');
  if (!el) {
    el = document.createElement('div');
    el.id = 'other-marks';
    el.style.cssText = 'position:absolute;top:10px;left:10px;display:flex;flex-direction:column;gap:4px;pointer-events:none;';
    document.getElementById('canvas-wrap').appendChild(el);
  }
  el.innerHTML = '';
  Object.entries(frameMarks || {}).forEach(([uname, v]) => {
    if (uname === ME) return;  // 自己的不在这里显示
    const chip = document.createElement('div');
    const isAccept = v.status === 'accept';
    chip.style.cssText = `background:${isAccept?'rgba(0,100,0,.75)':'rgba(100,0,0,.75)'};
      color:#fff;padding:3px 10px;border-radius:5px;font-size:12px;`;
    chip.textContent = `${isAccept?'✔':'✘'} ${uname}`;
    el.appendChild(chip);
  });
}
function renderBtnState(st) {
  document.getElementById('btn-accept').className =
    'btn-accept'+(st==='accept'?' btn-active-accept':'');
  document.getElementById('btn-reject').className =
    'btn-reject'+(st==='reject'?' btn-active-reject':'');
  // 有标注时才显示取消按钮
  document.getElementById('btn-clear').style.display = st ? 'inline-block' : 'none';
}
function updateStat(a,r,rev,ga,gr,grev,tot) {
  document.getElementById('stat-info').textContent =
    `我: ✔${a} ✘${r} (${rev})  |  全局: ✔${ga} ✘${gr} (${grev}/${tot})`;
}
function renderUsers(online) {
  const bar = document.getElementById('users-bar');
  bar.innerHTML='<span style="color:#555;font-size:11px">在线用户:</span>';
  Object.entries(online).forEach(([n,u]) => {
    const c=document.createElement('span');
    c.className='u-chip'; c.style.borderColor=u.color; c.style.color=u.color;
    c.innerHTML=`<span class="u-dot" style="background:${u.color}"></span>${n} <span style="color:#555">@${u.idx+1}</span>`;
    bar.appendChild(c);
  });
}

// ── 标注 ─────────────────────────────────────────────────────
async function doMark(st) {
  await fetch('/mark',{
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({idx:curIdx, status:st, user:ME})
  });
  if (st==='accept') {
    const r = await fetch(`/next_unreviewed?from=${curIdx+1}&user=${encodeURIComponent(ME)}`);
    const d = await r.json();
    await fetchFrame(d.idx);
    toast('✔ Accept → 第'+(d.idx+1)+'帧');
  } else {
    await fetchFrame(curIdx);
    toast('✘ Reject，停留当前帧');
  }
}

async function doClear() {
  await fetch('/mark',{
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({idx:curIdx, status:null, user:ME})
  });
  await fetchFrame(curIdx);
  toast('已取消标注');
}

// ── 导航 ─────────────────────────────────────────────────────
function go(d) { fetchFrame(curIdx+d); }
function jumpTo() {
  const v=parseInt(document.getElementById('jump-input').value)-1;
  if(!isNaN(v)) fetchFrame(v);
}
async function jumpNextNew() {
  const r=await fetch(`/next_unreviewed?from=${curIdx+1}&user=${encodeURIComponent(ME)}`);
  const d=await r.json();
  await fetchFrame(d.idx);
  toast('跳到第'+(d.idx+1)+'帧（未审）');
}

// ── 同步 ─────────────────────────────────────────────────────
async function doSync() {
  try {
    const r=await fetch(`/sync?since=${syncTs}&user=${encodeURIComponent(ME)}`);
    const d=await r.json();
    syncTs=d.ts;
    renderUsers(d.online_users);
    updateStat(d.accepted,d.rejected,d.reviewed,d.g_accepted,d.g_rejected,d.g_reviewed,curTotal);
    // 更新当前帧状态
    const myChg = d.changes[String(curIdx)];
    const frameChg = (d.frame_changes||{})[String(curIdx)];
    if (myChg) { renderOverlay(myChg.status, frameChg||curFrameMarks); renderBtnState(myChg.status); }
    else if (frameChg) { renderFrameMarks(frameChg); }
    if (frameChg) curFrameMarks = Object.assign(curFrameMarks, frameChg);
  } catch(e){}
}

// ── 清除记录 ─────────────────────────────────────────────────
async function resetRecords() {
  const choice = confirm(
    '⚠️ 清除范围选择：\n\n' +
    '• 点「确定」→ 清除【所有人】的标注记录\n' +
    '• 点「取消」→ 只清除【我自己】的记录'
  );
  const user = choice ? '' : ME;
  const tip = choice ? '所有人的记录' : `${ME} 的记录`;
  if (!confirm(`确认清除 ${tip}？此操作不可恢复！`)) return;
  const r = await fetch('/reset', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({user})
  });
  const d = await r.json();
  if (d.ok) {
    toast('🗑 ' + d.msg);
    await fetchFrame(0);
  } else {
    alert('清除失败');
  }
}

// ── 保存结果 ─────────────────────────────────────────────────
async function saveResults() {
  const outPath=document.getElementById('out-path-input').value.trim();
  const r=await fetch('/save',{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({out_path:outPath})
  });
  const d=await r.json();
  if(d.ok) alert(d.count?'保存成功:\n\n'+d.saved.join('\n'):'没有 Accept/Reject 帧，未保存');
  else alert('保存失败');
}

// ── 快捷键 ───────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if(e.target.tagName==='INPUT'||!ME) return;
  if(e.key==='ArrowLeft' ||e.key==='a') go(-1);
  if(e.key==='ArrowRight'||e.key==='d') go(+1);
  if(e.key==='q') doMark('accept');
  if(e.key==='e') doMark('reject');
  if(e.key==='s') saveLabelFile();
  if(e.key==='c') doClear();
  if(e.key==='n') jumpNextNew();
  if((e.key==='Delete'||e.key==='Backspace') && selectedBox>=0) {
    deleteBox({stopPropagation:()=>{}}, selectedBox);
  }
});

// ── Toast ────────────────────────────────────────────────────
function toast(msg,dur=2200){
  const t=document.getElementById('toast');
  t.textContent=msg; t.style.display='block';
  setTimeout(()=>t.style.display='none',dur);
}
</script>
</body>
</html>
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--train-txt", type=str, default=DEFAULT_TRAIN_TXT, help="图像路径 txt；未设置 --data-root 时使用"
    )
    parser.add_argument(
        "--val-txt", type=str, default=DEFAULT_VAL_TXT, help="验证图像路径 txt；未设置 --data-root 时使用"
    )
    parser.add_argument("--data-root", type=str, default="", help="case 数据根目录；会递归扫描 */images/* 图像")
    parser.add_argument("--progress-file", type=str, default="", help="审核进度 JSON 文件")
    parser.add_argument(
        "--weights", type=str, default=str(DEFAULT_WEIGHTS), help="用于生成初始 labels_10cls 的模型权重"
    )
    parser.add_argument(
        "--model-cfg",
        type=str,
        default=str(DEFAULT_MODEL_CFG),
        help="模型 yaml 配置；参考 infer.py 先建模型再 load 权重，留空则直接用权重",
    )
    parser.add_argument("--infer-on-start", action="store_true", help="启动 Web 服务前先对缺失 label 的图像做模型推理")
    parser.add_argument("--overwrite-labels", action="store_true", help="推理时覆盖已有 label 文件")
    parser.add_argument("--imgsz", type=int, default=1600)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    progress_file = args.progress_file
    if args.data_root and not progress_file:
        progress_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f"gt_viewer_{Path(args.data_root).name}_progress.json",
        )

    configure_dataset(
        train_txt=args.train_txt,
        val_txt=args.val_txt,
        data_root=args.data_root,
        progress_file=progress_file,
    )
    if args.infer_on_start:
        run_model_infer(
            weights=args.weights,
            image_paths=[p for p, _ in records],
            model_cfg=args.model_cfg,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            batch=args.batch,
            overwrite=args.overwrite_labels,
        )
        render_image.cache_clear()

    print(f"\n  GT Viewer 已启动 → http://{args.host}:{args.port}\n")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
