#!/usr/bin/env python3
r"""
AI Annotation Validator — 利用 Claude 视觉模型批量验收交通灯标注数据.
=======================================================================
规则：一张图片里只要有一个标注框错误，就排除该图片；全部正确才保留。
输出格式与训练 txt 完全一致（每行一个图像绝对路径）。

用法:
  # 并发模式（默认，8 线程）
  python tools/ai_annotation_validator.py

  # 指定输入/输出 & 线程数
  python tools/ai_annotation_validator.py \\
      --input  /pcpt/pcpt/data/2D-Det/COS/data_clean.txt \\
      --output /pcpt/pcpt/data/2D-Det/COS/ai_validated.txt \\
      --workers 16

  # 只验证前 1000 张（测试用）
  python tools/ai_annotation_validator.py --limit 1000

  # 断点续验（自动，脚本会跳过已处理的图像）
  python tools/ai_annotation_validator.py

  # 使用 Batch API（50% 折扣，异步，适合离线大批量）
  python tools/ai_annotation_validator.py --batch

  # 从进度文件重新生成 output.txt（无需再次调用 API）
  python tools/ai_annotation_validator.py --regen-output
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests as _requests
from PIL import Image, ImageDraw


# ── 轻量 DashScope 客户端（避免 openai SDK 版本冲突）──────────
class _QwenClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.session = _requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def chat(self, model: str, messages: list, max_tokens: int = 1024) -> str:
        """返回模型回复文本，失败抛出异常。."""
        resp = self.session.post(
            f"{self.base_url}/chat/completions",
            json={"model": model, "max_tokens": max_tokens, "messages": messages},
            timeout=120,
        )
        if resp.status_code == 429:
            raise _RateLimitError()
        if resp.status_code >= 500:
            raise _ServerError(resp.status_code)
        if not resp.ok:
            raise _APIError(resp.status_code, resp.text[:200])
        return resp.json()["choices"][0]["message"]["content"] or ""


class _RateLimitError(Exception):
    pass


class _ServerError(Exception):
    def __init__(self, code):
        self.status_code = code


class _APIError(Exception):
    def __init__(self, code, msg):
        self.status_code = code
        self.message = msg


# ─────────────────────────────────────────────────────────────────────────────
# 数据集配置（与 gt_viewer.py 保持完全一致）
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_TXT = "/pcpt/pcpt/data/2D-Det/COS/data_clean.txt"

CLASS_NAMES = {
    0: "RedLight",
    1: "GreenLight",
    2: "YellowLight",
    3: "BlackLight",
    4: "MovingTrafficLight",
    5: "MultiSemanticTrafficLights",
}
CLASS_COLORS = {
    0: (220, 50, 50),  # 红
    1: (50, 200, 50),  # 绿
    2: (220, 180, 0),  # 黄
    3: (130, 130, 130),  # 灰
    4: (50, 150, 255),  # 蓝
    5: (200, 50, 200),  # 紫
}

SCRIPT_DIR = Path(__file__).parent
DEFAULT_OUTPUT = "/pcpt/pcpt/data/2D-Det/COS/ai_validated.txt"
DEFAULT_PROGRESS = str(SCRIPT_DIR / "ai_validator_progress.json")
DEFAULT_LOG = str(SCRIPT_DIR / "ai_validated.jsonl")
DEFAULT_AMBIGUOUS = str(SCRIPT_DIR / "ai_ambiguous.txt")
DEFAULT_REJECTED = str(SCRIPT_DIR / "ai_rejected.txt")
DEFAULT_MODEL = "qwen3.5-plus"

# ─────────────────────────────────────────────────────────────────────────────
# API 配置（读取环境变量）
# ─────────────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# ─────────────────────────────────────────────────────────────────────────────
# 图像与标注工具
# ─────────────────────────────────────────────────────────────────────────────


def get_label_path(img_path: str) -> str:
    """img_path: .../images/xxx.jpg → .../labels_6cls/xxx.txt."""
    p = img_path.replace("/images/", "/labels_6cls/")
    return os.path.splitext(p)[0] + ".txt"


def load_boxes(label_path: str, img_w: int, img_h: int) -> list[dict]:
    """读取 YOLO 标注行，返回像素坐标 dict 列表。."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:5])
            x1 = int((cx - bw / 2) * img_w)
            y1 = int((cy - bh / 2) * img_h)
            x2 = int((cx + bw / 2) * img_w)
            y2 = int((cy + bh / 2) * img_h)
            boxes.append(
                {
                    "cls_id": cls_id,
                    "name": CLASS_NAMES.get(cls_id, f"cls{cls_id}"),
                    "color": CLASS_COLORS.get(cls_id, (255, 255, 255)),
                    "x1": max(0, x1),
                    "y1": max(0, y1),
                    "x2": min(img_w, x2),
                    "y2": min(img_h, y2),
                }
            )
    return boxes


def render_annotated(
    img_path: str, boxes: list[dict], max_w: int = 1280, max_h: int = 720
) -> tuple[Image.Image, float]:
    """加载图像，缩略到指定尺寸，绘制带序号的标注框。 返回 (绘制后的 PIL Image, 缩放比例 scale)。.
    """
    img = Image.open(img_path).convert("RGB")
    orig_w, _ = img.size
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    disp_w, _ = img.size
    scale = disp_w / orig_w

    draw = ImageDraw.Draw(img)
    for i, b in enumerate(boxes):
        sx1 = int(b["x1"] * scale)
        sy1 = int(b["y1"] * scale)
        sx2 = int(b["x2"] * scale)
        sy2 = int(b["y2"] * scale)
        color = b["color"]
        # 框线
        draw.rectangle([sx1, sy1, sx2, sy2], outline=color, width=2)
        # 标签背景 + 文字
        label = f"{i + 1}.{b['name']}"
        lw = len(label) * 7 + 4
        lh = 16
        draw.rectangle([sx1, max(0, sy1 - lh), sx1 + lw, sy1], fill=color)
        draw.text((sx1 + 2, max(0, sy1 - lh + 1)), label, fill=(255, 255, 255))
    return img, scale


def crop_box_zoom(
    base_img: Image.Image, box: dict, scale: float, target_size: int = 96, pad: int = 8
) -> Image.Image | None:
    """裁剪单个标注框区域并放大到 target_size，便于 Claude 辨别小目标。 仅对宽或高小于 60px 的框生效，返回 None 表示不需要裁剪。.
    """
    box_w = box["x2"] - box["x1"]
    box_h = box["y2"] - box["y1"]
    if max(box_w, box_h) * scale >= 60:
        return None  # 足够大，不需要放大图

    sx1 = max(0, int(box["x1"] * scale) - pad)
    sy1 = max(0, int(box["y1"] * scale) - pad)
    sx2 = min(base_img.width, int(box["x2"] * scale) + pad)
    sy2 = min(base_img.height, int(box["y2"] * scale) + pad)
    crop = base_img.crop((sx1, sy1, sx2, sy2))
    cw, ch = crop.size
    if cw < 2 or ch < 2:
        return None
    # 放大到 target_size（保持比例）
    factor = target_size / max(cw, ch)
    new_w, new_h = max(1, int(cw * factor)), max(1, int(ch * factor))
    return crop.resize((new_w, new_h), Image.NEAREST)


def pil_to_b64(img: Image.Image, quality: int = 82) -> str:
    """PIL Image → JPEG base64 字符串。."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Claude 提示词 & API 调用
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    你是自动驾驶数据集标注质量审核专家，负责验证交通灯的 YOLO 标注是否正确。

    === 类别定义 ===
    0  RedLight              : 红灯（灯体主色为红色，明确亮起，亮度清晰可辨）
    1  GreenLight             : 绿灯（灯体主色为绿色，明确亮起，亮度清晰可辨）
    2  YellowLight            : 黄灯（灯体主色为黄色，明确亮起，亮度清晰可辨）
    3  BlackLight             : 黑灯/熄灭灯（灯体不亮、极暗、微亮但颜色不可辨、或被遮挡）
    4  MovingTrafficLight     : 移动交通灯（安装在施工车辆上的临时信号灯）
    5  MultiSemanticTrafficLights : 多义交通灯（同一灯组含多种语义，如箭头+圆形复合）

    === ⚡ 本次任务的审核范围 ===
    **只需严格审核 RedLight（类别0）和 BlackLight（类别3）这两个类别。**
    其他类别（GreenLight / YellowLight / MovingTrafficLight / MultiSemanticTrafficLights）
    的标注结果请在 JSON 中正常输出，但 is_correct 一律填 true、is_ambiguous 填 false，
    reason 填"非审核目标类别，跳过"即可，不必实际检查。

    === 🎨 唯一判断维度：颜色语义（亮/灭 + 颜色） ===
    ⚠️ 重要前提：**完全忽略灯的形状、图案、箭头方向、数字倒计时**。
    无论灯体内部显示的是圆形、左箭头、右箭头、直行箭头还是数字，
    只要整体发出红色光 → RedLight；只要灯体熄灭/极暗 → BlackLight。
    形状和图案与本次审核无关，不得作为判断依据。

    === 红灯 / 黑灯 判断规则（只区分亮/灭，不挑颜色深浅）===

    ▶ 核心原则：**只判断灯有没有亮，不管颜色是正红还是橙红。**
      红色、橙红色、偏橙的红 → 统一算 RedLight（亮着就行）
      没亮、极暗、熄灭 → 统一算 BlackLight

    ──────────────────────────────────────────
    【RedLight】灯亮了 = 正确
    ──────────────────────────────────────────
    ▶ is_correct=true：灯体可见发光，颜色偏红或偏橙均可，肉眼能看出"这灯是亮的"

    ▶ is_correct=false（仅以下情况才拒）：
      · 灯体完全没有发光，是熄灭状态（应标 BlackLight）
      · 图像极度模糊/目标极小，完全看不出灯有没有亮

    ──────────────────────────────────────────
    【BlackLight】灯没亮 = 正确
    ──────────────────────────────────────────
    ▶ is_correct=true：灯体暗的、不亮的、模糊看不出亮灭的（默认算没亮）

    ▶ is_correct=false（仅以下情况才拒）：
      · 灯体明显亮起，能看到发光（无论红/橙/绿/黄，亮了就不是黑灯）

    ──────────────────────────────────────────
    ▶ 关于 is_ambiguous：统一填 false
    ▶ 判断口诀：**亮了算红灯，没亮算黑灯，橙色也是红，只看亮没亮**

    === 输出格式（严格 JSON，不要任何额外文字）===
    {
      "results": [
        {"id": 1, "label": "RedLight",   "is_correct": true,  "is_ambiguous": false, "reason": "灯体发出清晰明亮的红色光，亮度明确"},
        {"id": 2, "label": "RedLight",   "is_correct": false, "is_ambiguous": false, "reason": "灯体偏暗，红色不饱和，实为黑灯未亮起"},
        {"id": 3, "label": "BlackLight", "is_correct": false, "is_ambiguous": false, "reason": "图像过暗，无法确认灯体熄灭，看不清一律拒"},
        {"id": 4, "label": "GreenLight", "is_correct": true,  "is_ambiguous": false, "reason": "非审核目标类别，跳过"}
      ],
      "all_correct": false
    }
""")


def build_content(img_main: Image.Image, boxes: list[dict], crops: list[Image.Image | None]) -> list[dict]:
    """构建发送给 Qwen VL 的 message content（OpenAI 兼容格式）。."""
    content: list[dict] = []

    # 1. 全图（带标注框）
    content.append(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{pil_to_b64(img_main)}"},
        }
    )

    # 2. 标注框文字列表
    lines = []
    for i, b in enumerate(boxes):
        lines.append(
            f"  框{i + 1}: {b['name']} (cls={b['cls_id']})  像素位置 [{b['x1']},{b['y1']} → {b['x2']},{b['y2']}]"
        )
    content.append(
        {
            "type": "text",
            "text": f"上图共 {len(boxes)} 个交通灯标注（编号已标在图中）：\n" + "\n".join(lines),
        }
    )

    # 3. 对小框附加放大裁剪图（最多 6 个，避免 content 过大）
    small_items = [(i + 1, b["name"], c) for i, (b, c) in enumerate(zip(boxes, crops)) if c is not None][:6]
    if small_items:
        content.append(
            {
                "type": "text",
                "text": f"以下是 {len(small_items)} 个较小框的局部放大图（辅助判断）：",
            }
        )
        for box_id, name, crop_img in small_items:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{pil_to_b64(crop_img, quality=92)}"},
                }
            )
            content.append({"type": "text", "text": f"↑ 框{box_id} ({name})"})

    content.append(
        {
            "type": "text",
            "text": "请验证每个框的标注是否正确，严格按 JSON 格式输出结果，不要有任何其他文字。",
        }
    )
    return content


def extract_json(text: str) -> dict:
    """从 Claude 返回的文本中提取 JSON（兼容 markdown 代码块）。."""
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        # 取第一个 ``` 块内容
        for part in parts[1:]:
            stripped = part.lstrip("json").strip()
            if stripped.startswith("{"):
                text = stripped
                break
    # 找最外层 {} 块
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def call_qwen_api(
    client: _QwenClient,
    model: str,
    img_main: Image.Image,
    boxes: list[dict],
    crops: list[Image.Image | None],
    max_retries: int = 4,
) -> dict | None:
    """调用 Qwen VL API 进行标注验证，返回解析后的 dict，失败返回 None。."""
    content = build_content(img_main, boxes, crops)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]

    for attempt in range(max_retries):
        try:
            raw = client.chat(model=model, messages=messages, max_tokens=1024)
            return extract_json(raw)

        except json.JSONDecodeError:
            if attempt < max_retries - 1:
                time.sleep(2)
            continue

        except _RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"    [RATE_LIMIT] 等待 {wait}s...")
            time.sleep(wait)

        except _ServerError:
            time.sleep(10 * (attempt + 1))

        except _APIError as e:
            print(f"    [API_ERR {e.status_code}] {e.message[:80]}")
            return None

        except Exception as e:
            print(f"    [ERR] {type(e).__name__}: {e}")
            time.sleep(3)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 单张图像验证
# ─────────────────────────────────────────────────────────────────────────────


def validate_one(client: _QwenClient, model: str, img_path: str, skip_no_label: bool) -> tuple[bool, str, dict]:
    """验证单张图像的标注质量。 返回 (is_valid, summary, detail_dict)： is_valid = True 表示全部正确，False 表示排除 summary = 简短原因字符串（用于终端输出）
    detail_dict = 完整分析记录（写入 JSONL 日志）.
    """
    label_path = get_label_path(img_path)
    base_detail: dict = {
        "img_path": img_path,
        "label_path": label_path,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── 图像不存在 ──
    if not os.path.exists(img_path):
        return (
            False,
            "图像文件不存在",
            {**base_detail, "verdict": "reject", "reason": "图像文件不存在", "boxes": [], "results": []},
        )

    # ── 标注文件不存在 ──
    if not os.path.exists(label_path):
        if skip_no_label:
            return (
                False,
                "无标注文件（已排除）",
                {**base_detail, "verdict": "reject", "reason": "无标注文件", "boxes": [], "results": []},
            )
        return (
            True,
            "无标注文件（视为通过）",
            {**base_detail, "verdict": "accept", "reason": "无标注文件（视为通过）", "boxes": [], "results": []},
        )

    # ── 读取原图尺寸 & 标注框 ──
    try:
        with Image.open(img_path) as _img:
            orig_w, orig_h = _img.size
    except Exception as e:
        return (
            False,
            f"图像读取失败: {e}",
            {**base_detail, "verdict": "reject", "reason": str(e), "boxes": [], "results": []},
        )

    boxes = load_boxes(label_path, orig_w, orig_h)
    boxes_info = [
        {
            "id": i + 1,
            "cls_id": b["cls_id"],
            "name": b["name"],
            "x1": b["x1"],
            "y1": b["y1"],
            "x2": b["x2"],
            "y2": b["y2"],
            "w": b["x2"] - b["x1"],
            "h": b["y2"] - b["y1"],
        }
        for i, b in enumerate(boxes)
    ]
    base_detail["img_size"] = [orig_w, orig_h]
    base_detail["boxes"] = boxes_info

    if not boxes:
        return True, "标注为空（通过）", {**base_detail, "verdict": "accept", "reason": "标注为空", "results": []}

    # ── 渲染带框图像 ──
    try:
        img_annotated, scale = render_annotated(img_path, boxes)
    except Exception as e:
        return False, f"渲染失败: {e}", {**base_detail, "verdict": "reject", "reason": str(e), "results": []}

    crops = [crop_box_zoom(img_annotated, b, scale) for b in boxes]

    # ── 调用 Claude ──
    result = call_qwen_api(client, model, img_annotated, boxes, crops)
    if result is None:
        return (
            False,
            "API 调用失败（已排除）",
            {**base_detail, "verdict": "reject", "reason": "API 调用失败", "results": []},
        )

    results_list = result.get("results", [])

    # ── 只针对 RedLight / BlackLight 做严格过滤 ──────────────────────────────
    # 其他类别（GreenLight / YellowLight / MovingTrafficLight / MultiSemanticTrafficLights）
    # 的判断结果完全忽略，不影响 accept/reject 决策。
    TARGET_LABELS = {"RedLight", "BlackLight"}

    # 将 AI 结果按序号 id 与 boxes 对应，取出目标类别的框
    # boxes 列表下标 0-based，results_list id 从 1 开始
    id2box = {i + 1: b for i, b in enumerate(boxes)}

    def _is_target(r: dict) -> bool:
        """判断该结果是否属于需要过滤的目标类别（RedLight / BlackLight）。."""
        # 优先用 AI 返回的 label 字段；若缺失则回退到 boxes 原始标注
        label = r.get("label", "")
        if label in TARGET_LABELS:
            return True
        box = id2box.get(r.get("id", -1), {})
        return box.get("name", "") in TARGET_LABELS

    # 仅考虑目标类别的结果
    target_results = [r for r in results_list if _is_target(r)]

    # 在目标类别中区分"明确错误"与"歧义/看不清"
    wrong_clear = [r for r in target_results if not r.get("is_correct", True) and not r.get("is_ambiguous", False)]
    wrong_ambig = [r for r in target_results if not r.get("is_correct", True) and r.get("is_ambiguous", False)]
    has_ambiguous = len(wrong_ambig) > 0

    # 只要目标类别中存在任何错误（包括歧义），整张图即拒绝
    target_all_correct = len(wrong_clear) == 0 and len(wrong_ambig) == 0
    verdict = "accept" if target_all_correct else "reject"

    # ── 生成 summary ─────────────────────────────────────────────────────────
    non_target_count = len(boxes) - len(target_results)
    skip_note = f"（其余 {non_target_count} 个非红/黑灯框已跳过）" if non_target_count > 0 else ""

    if target_all_correct:
        if not target_results:
            summary = f"无 RedLight/BlackLight 框，直接通过{skip_note}"
        else:
            summary = f"RedLight/BlackLight 共 {len(target_results)} 个框全部正确{skip_note}"
    elif wrong_ambig and not wrong_clear:
        summary = "红/黑灯歧义框({}/{}): {}{}".format(
            len(wrong_ambig),
            len(target_results),
            "；".join(f"框{r['id']}[{r.get('label', '?')}]: {r.get('reason', '')[:40]}" for r in wrong_ambig),
            skip_note,
        )
    else:
        all_wrong = wrong_clear + wrong_ambig
        summary = "红/黑灯错误框({}/{}): {}{}".format(
            len(all_wrong),
            len(target_results),
            "；".join(
                f"框{r['id']}[{r.get('label', '?')}]{'⚠歧义' if r.get('is_ambiguous') else ''}: "
                f"{r.get('reason', '')[:35]}"
                for r in all_wrong
            ),
            skip_note,
        )

    detail = {
        **base_detail,
        "verdict": verdict,
        "has_ambiguous": has_ambiguous,
        "reason": summary,
        "results": results_list,
    }
    return target_all_correct, summary, detail


# ─────────────────────────────────────────────────────────────────────────────
# 进度管理（线程安全）
# ─────────────────────────────────────────────────────────────────────────────


class ProgressManager:
    """持久化进度： processed = { img_path: True/False } True=accept, False=reject 同时写入 JSONL 详细日志（每张图一行，含 Claude 逐框分析原因）。 每 N
    张自动写盘，Ctrl+C 也会保存。.
    """

    SAVE_INTERVAL = 100  # 每处理 N 张保存一次进度

    def __init__(
        self, progress_file: Path, output_file: Path, log_file: Path, ambiguous_file: Path, rejected_file: Path
    ):
        self.progress_file = progress_file
        self.output_file = output_file
        self.log_file = log_file
        self.ambiguous_file = ambiguous_file
        self.rejected_file = rejected_file
        self.lock = threading.Lock()
        self.processed: dict[str, bool] = {}
        self._dirty = 0
        self._load()

    def _load(self):
        if not self.progress_file.exists():
            return
        try:
            with open(self.progress_file) as f:
                data = json.load(f)
            self.processed = data.get("processed", {})
            print(f"[INFO] 恢复进度: 已处理 {len(self.processed)} 张 (接受 {self.accepted} / 拒绝 {self.rejected})")
            print(f"[INFO] 详细日志: {self.log_file}")
        except Exception as e:
            print(f"[WARN] 读取进度文件失败: {e}")

    def save(self):
        try:
            with open(self.progress_file, "w") as f:
                json.dump(
                    {"processed": self.processed, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            print(f"[WARN] 保存进度失败: {e}")

    def update(self, img_path: str, is_valid: bool, detail: dict | None = None):
        with self.lock:
            self.processed[img_path] = is_valid
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            # 追加写入 output.txt（仅 accept）
            if is_valid:
                with open(self.output_file, "a") as f:
                    f.write(img_path + "\n")
            # 追加写入 rejected.txt（所有拒绝）
            if not is_valid:
                with open(self.rejected_file, "a", encoding="utf-8") as f:
                    f.write(img_path + "\n")
            # 追加写入歧义列表（仅 reject 且含歧义框）
            if not is_valid and detail and detail.get("has_ambiguous"):
                with open(self.ambiguous_file, "a", encoding="utf-8") as f:
                    f.write(img_path + "\n")
            # 追加写入 JSONL 详细日志（每张图一行）
            if detail is not None:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(detail, ensure_ascii=False) + "\n")
            self._dirty += 1
            if self._dirty >= self.SAVE_INTERVAL:
                self.save()
                self._dirty = 0

    def is_done(self, img_path: str) -> bool:
        return img_path in self.processed

    def rebuild_output(self):
        """从进度字典重新生成 output.txt（断点续验后重建用）。."""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, "w") as f:
            for path, valid in self.processed.items():
                if valid:
                    f.write(path + "\n")
        print(f"[INFO] 重建完成: {self.accepted} 条 → {self.output_file}")

    @property
    def accepted(self) -> int:
        return sum(1 for v in self.processed.values() if v)

    @property
    def rejected(self) -> int:
        return sum(1 for v in self.processed.values() if not v)

    @property
    def ambiguous(self) -> int:
        """从 JSONL 日志统计歧义拒绝数（近似：读文件代价高，仅在 summary 时用）。."""
        try:
            count = 0
            with open(self.ambiguous_file, encoding="utf-8") as f:
                count = sum(1 for l in f if l.strip())
            return count
        except FileNotFoundError:
            return 0

    @property
    def total(self) -> int:
        return len(self.processed)


# ─────────────────────────────────────────────────────────────────────────────
# Batch API 模式
# ─────────────────────────────────────────────────────────────────────────────


def run_batch_mode(
    client: _QwenClient,
    model: str,
    todo: list[str],
    progress: ProgressManager,
    skip_no_label: bool,
    batch_size: int = 50,
):
    """Batch API 模式（DashScope/Qwen 暂不支持，请使用并发模式）。."""
    print("[WARN] Qwen/DashScope 暂不支持 Batch API，请改用并发模式（去掉 --batch 参数）。")


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="AI 标注验收：利用 Claude 视觉模型验证交通灯标注质量",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例:
              python tools/ai_annotation_validator.py --limit 200 --workers 8
              python tools/ai_annotation_validator.py --batch --batch-size 30
              python tools/ai_annotation_validator.py --regen-output
        """),
    )
    parser.add_argument("--input", default=TRAIN_TXT, help="输入 txt 路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出 txt 路径（全部正确）")
    parser.add_argument("--progress", default=DEFAULT_PROGRESS, help="进度文件路径")
    parser.add_argument("--log", default=DEFAULT_LOG, help="JSONL 详细日志路径")
    parser.add_argument("--ambiguous", default=DEFAULT_AMBIGUOUS, help="歧义图像列表 txt 路径")
    parser.add_argument("--rejected", default=DEFAULT_REJECTED, help="所有拒绝图像列表 txt 路径")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude 模型 ID")
    parser.add_argument("--workers", type=int, default=8, help="并发线程数（并发模式）")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少张（0=全部）")
    parser.add_argument("--skip-no-label", action="store_true", help="无标注文件的图像直接排除")
    parser.add_argument("--batch", action="store_true", help="使用 Batch API（半价，异步）")
    parser.add_argument("--batch-size", type=int, default=40, help="Batch API 每批图像数")
    parser.add_argument("--regen-output", action="store_true", help="不调用 API，仅从进度文件重建 output.txt")
    args = parser.parse_args()

    # ── 检查 API Key ──────────────────────────────────────────
    if not API_KEY:
        print("[ERROR] 未检测到 DASHSCOPE_API_KEY 或 OPENAI_API_KEY 环境变量！")
        sys.exit(1)

    # ── 初始化 ────────────────────────────────────────────────
    progress = ProgressManager(
        Path(args.progress), Path(args.output), Path(args.log), Path(args.ambiguous), Path(args.rejected)
    )

    # 仅重建输出文件
    if args.regen_output:
        progress.rebuild_output()
        return

    # ── 读取图像列表 ──────────────────────────────────────────
    if not os.path.exists(args.input):
        print(f"[ERROR] 输入文件不存在: {args.input}")
        sys.exit(1)

    with open(args.input) as f:
        all_paths = [l.strip() for l in f if l.strip()]
    print(f"[INFO] 输入图像总数: {len(all_paths)}")

    # 过滤已处理
    todo = [p for p in all_paths if not progress.is_done(p)]
    if args.limit > 0:
        todo = todo[: args.limit]

    print(f"[INFO] 待处理: {len(todo)} 张  |  已跳过(进度文件): {len(all_paths) - len(todo)} 张")
    print(f"[INFO] 模型: {args.model}  |  输出: {args.output}")
    print(f"[INFO] 详细日志: {args.log}")
    print(f"[INFO] 歧义记录: {args.ambiguous}")
    print(f"[INFO] 错误记录: {args.rejected}")

    if not todo:
        print("[INFO] 无待处理图像，所有任务已完成。")
        progress.save()
        _print_summary(progress)
        return

    # ── 初始化 output.txt（写入已有 accepted 结果）──────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for img_path, valid in progress.processed.items():
            if valid:
                f.write(img_path + "\n")

    # ── 创建 Qwen 客户端 ──────────────────────────────────────
    client = _QwenClient(api_key=API_KEY, base_url=BASE_URL)

    # ── 执行验收 ──────────────────────────────────────────────
    start_time = time.time()

    if args.batch:
        # ─── Batch API 模式 ────────────────────────────────────
        print(f"[INFO] 使用 Batch API 模式，batch_size={args.batch_size}")
        try:
            run_batch_mode(client, args.model, todo, progress, args.skip_no_label, args.batch_size)
        except KeyboardInterrupt:
            print("\n[INFO] 用户中断，保存进度...")
    else:
        # ─── 并发模式 ─────────────────────────────────────────
        print(f"[INFO] 使用并发模式，workers={args.workers}")
        done_lock = threading.Lock()
        done_n = [0]

        def process_one(img_path: str):
            is_valid, reason, detail = validate_one(client, args.model, img_path, args.skip_no_label)
            progress.update(img_path, is_valid, detail)

            with done_lock:
                done_n[0] += 1
                n = done_n[0]

            elapsed = time.time() - start_time
            speed = n / elapsed * 60 if elapsed > 0 else 0  # 张/min
            eta_h = (len(todo) - n) / (speed / 60) / 3600 if speed > 0 else 0

            mark = "✔" if is_valid else "✘"
            reason_short = reason[:60]
            print(
                f"[{n:6d}/{len(todo)}] {mark}  {Path(img_path).name[:38]:<38}  "
                f"A:{progress.accepted} R:{progress.rejected}  "
                f"{speed:5.1f}张/min  ETA:{eta_h:.1f}h  {reason_short}"
            )

        try:
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                futures = {ex.submit(process_one, p): p for p in todo}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"[ERR] {futures[future]}: {e}")
        except KeyboardInterrupt:
            print("\n[INFO] 用户中断，保存进度...")

    progress.save()
    _print_summary(progress, start_time)


def _print_summary(progress: ProgressManager, start_time: float | None = None):
    elapsed = time.time() - start_time if start_time else 0
    ambiguous = progress.ambiguous
    print("\n" + "═" * 62)
    print("  验收完成！")
    print(f"  总处理: {progress.total} 张")
    print(f"  接受 (全对):          {progress.accepted} 张  → {progress.output_file}")
    print(f"  拒绝 (明确错误):      {progress.rejected - ambiguous} 张  → {progress.rejected_file}")
    print(f"  拒绝 (歧义/看不清):   {ambiguous} 张  → {progress.ambiguous_file}")
    if progress.total > 0:
        print(f"  接受率: {progress.accepted / progress.total * 100:.1f}%")
    if elapsed > 60:
        print(f"  耗时: {elapsed / 3600:.2f} h")
    print("═" * 62)


if __name__ == "__main__":
    main()
