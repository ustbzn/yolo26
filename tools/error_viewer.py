#!/usr/bin/env python3
"""
AI 标注错误查看器
=================
读取 ai_annotation_validator.py 生成的 JSONL 日志，
在浏览器中逐帧展示被拒绝的图像、标注框及 AI 的逐框分析原因。

⚠️  当前审核范围：仅 RedLight（红灯）和 BlackLight（黑灯）两个类别
    · RedLight  ——严格模式：必须清晰亮起、红色饱和，任何不确定 → 拒绝
    · BlackLight——宽松模式：框内无明显发光即通过，模糊/过小默认算黑灯
    · 其他类别（绿/黄/移动/多义）：跳过，不影响 accept/reject 结果

用法:
  python tools/error_viewer.py                    # 默认加载 sample_300_validated.jsonl，只看拒绝帧
  python tools/error_viewer.py --log xxx.jsonl    # 指定日志文件
  python tools/error_viewer.py --filter reject    # 只看被拒绝的帧（默认）
  python tools/error_viewer.py --filter accept    # 只看被接受的帧
  python tools/error_viewer.py --filter wrong     # 只看明确标注错误的帧
  python tools/error_viewer.py --filter all       # 全部帧（含已接受）
  python tools/error_viewer.py --port 8890        # 指定端口（默认 8891）

框颜色说明（图中标注框）:
  绿色  ✔ 正确  — AI 判定该框标注无误
  红色  ✘ 错误  — AI 判定该框标注有误（RedLight/BlackLight 混淆或看不清红灯）
  灰色        — 非审核目标类别，已跳过（绿/黄/移动/多义灯）

快捷键:
  ← / → 或 A / D   上一帧 / 下一帧
  数字 + 回车       跳转到指定帧
"""

import os, io, base64, json, argparse
from pathlib import Path
from flask import Flask, request, jsonify
from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
DEFAULT_LOG  = str(SCRIPT_DIR / "sample_300_validated.jsonl")

CLASS_COLORS = {
    0: (220, 50,  50),
    1: (50,  200, 50),
    2: (220, 180,  0),
    3: (130, 130, 130),
    4: (50,  150, 255),
    5: (200,  50, 200),
}

# 框状态颜色
COLOR_CORRECT   = (60,  200,  60)   # 绿 — 正确
COLOR_WRONG     = (220,  50,  50)   # 红 — 错误
COLOR_AMBIGUOUS = (240, 160,  30)   # 橙 — 歧义


# ──────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────

def load_records(log_path: str, filter_mode: str):
    """从 JSONL 加载记录，按 filter_mode 过滤。"""
    records = []
    if not os.path.exists(log_path):
        print(f"[WARN] 日志文件不存在: {log_path}")
        return records

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            verdict   = rec.get("verdict", "accept")
            has_ambig = rec.get("has_ambiguous", False)

            if filter_mode == "reject":
                if verdict != "reject":
                    continue
            elif filter_mode == "accept":
                if verdict != "accept":
                    continue
            elif filter_mode == "ambiguous":
                if not (verdict == "reject" and has_ambig):
                    continue
            elif filter_mode == "wrong":
                if not (verdict == "reject" and not has_ambig):
                    continue
            # filter_mode == "all" → 全收

            records.append(rec)

    print(f"[INFO] 加载 {len(records)} 条记录 (filter={filter_mode})")
    return records


# ──────────────────────────────────────────────────────────────
# 图像渲染
# ──────────────────────────────────────────────────────────────

def render_frame(rec: dict, max_w=1440, max_h=900):
    """
    渲染带标注框的图像：
      - 正确框 → 绿色
      - 错误框 → 红色
      - 歧义框 → 橙色
    返回 base64 JPEG 字符串。
    """
    img_path = rec.get("img_path", "")
    results  = {r["id"]: r for r in rec.get("results", [])}
    boxes    = rec.get("boxes", [])

    # 加载图像（不存在则黑底提示）
    if os.path.exists(img_path):
        img = Image.open(img_path).convert("RGB")
    else:
        img = Image.new("RGB", (960, 540), (20, 20, 20))
        ImageDraw.Draw(img).text(
            (20, 250), f"图像不存在:\n{img_path}", fill=(255, 80, 80)
        )
    orig_w, orig_h = img.size
    img.thumbnail((max_w, max_h), Image.LANCZOS)
    disp_w, disp_h = img.size
    scale = disp_w / orig_w

    draw = ImageDraw.Draw(img)

    for i, box in enumerate(boxes):
        box_id  = i + 1
        res     = results.get(box_id, {})
        correct = res.get("is_correct", True)
        ambig   = res.get("is_ambiguous", False)

        if correct:
            color = COLOR_CORRECT
        elif ambig:
            color = COLOR_AMBIGUOUS
        else:
            color = COLOR_WRONG

        sx1 = int(box["x1"] * scale)
        sy1 = int(box["y1"] * scale)
        sx2 = int(box["x2"] * scale)
        sy2 = int(box["y2"] * scale)
        w   = box.get("w", sx2 - sx1)
        h   = box.get("h", sy2 - sy1)

        # 框线（错误加粗）
        lw = 3 if not correct else 2
        draw.rectangle([sx1, sy1, sx2, sy2], outline=color, width=lw)

        # 标签背景
        label = f"{box_id}.{box.get('name', '?')} {w}x{h}"
        lw_px = len(label) * 7 + 4
        lh_px = 16
        ty = max(0, sy1 - lh_px)
        draw.rectangle([sx1, ty, sx1 + lw_px, ty + lh_px], fill=color)
        draw.text((sx1 + 2, ty + 1), label, fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode(), disp_w, disp_h


# ──────────────────────────────────────────────────────────────
# Flask App
# ──────────────────────────────────────────────────────────────

app = Flask(__name__)
RECORDS: list = []
_log_path:    str = ""
_filter_mode: str = "reject"


@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/reload")
def api_reload():
    """重新读取 JSONL 文件，返回最新记录数。"""
    global RECORDS
    RECORDS = load_records(_log_path, _filter_mode)
    return jsonify({"total": len(RECORDS), "reloaded": True})


@app.route("/api/info")
def api_info():
    return jsonify({"total": len(RECORDS)})


@app.route("/api/frame/<int:idx>")
def api_frame(idx: int):
    if not RECORDS or idx < 0 or idx >= len(RECORDS):
        return jsonify({"error": "out of range"}), 404

    rec = RECORDS[idx]
    img_b64, disp_w, disp_h = render_frame(rec)

    # 构造逐框分析列表
    results_map = {r["id"]: r for r in rec.get("results", [])}
    boxes_info  = []
    for i, box in enumerate(rec.get("boxes", [])):
        bid = i + 1
        res = results_map.get(bid, {})
        correct = res.get("is_correct", True)
        ambig   = res.get("is_ambiguous", False)
        if correct:
            status = "correct"
        elif ambig:
            status = "ambiguous"
        else:
            status = "wrong"
        boxes_info.append({
            "id":     bid,
            "name":   box.get("name", "?"),
            "w":      box.get("w", 0),
            "h":      box.get("h", 0),
            "status": status,
            "reason": res.get("reason", ""),
        })

    return jsonify({
        "idx":          idx,
        "total":        len(RECORDS),
        "img_path":     rec.get("img_path", ""),
        "verdict":      rec.get("verdict", ""),
        "has_ambiguous":rec.get("has_ambiguous", False),
        "reason":       rec.get("reason", ""),
        "timestamp":    rec.get("timestamp", ""),
        "img_size":     rec.get("img_size", []),
        "img_b64":      img_b64,
        "disp_w":       disp_w,
        "disp_h":       disp_h,
        "boxes":        boxes_info,
    })


# ──────────────────────────────────────────────────────────────
# 前端页面
# ──────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>AI 标注错误查看器</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #1a1a1a; color: #eee; font-family: 'Segoe UI', sans-serif;
         display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

  /* ── 顶部工具栏 ── */
  #topbar {
    display: flex; align-items: center; gap: 12px;
    padding: 8px 16px; background: #111; border-bottom: 1px solid #333;
    flex-shrink: 0;
  }
  #topbar h1 { font-size: 15px; color: #aaa; font-weight: 400; }
  #counter   { font-size: 14px; color: #ccc; margin-left: auto; }
  #verdict-badge {
    padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600;
  }
  .badge-reject   { background:#c0392b; color:#fff; }
  .badge-ambiguous{ background:#e67e22; color:#fff; }
  .badge-accept   { background:#27ae60; color:#fff; }

  /* ── 主体布局 ── */
  #main { display: flex; flex: 1; overflow: hidden; }

  /* ── 图像区 ── */
  #img-area {
    flex: 1; display: flex; align-items: center; justify-content: center;
    overflow: hidden; background: #111; position: relative;
  }
  #frame-img { max-width: 100%; max-height: 100%; object-fit: contain; }
  #nav-left, #nav-right {
    position: absolute; top: 50%; transform: translateY(-50%);
    background: rgba(0,0,0,.5); border: none; color: #fff;
    font-size: 28px; width: 44px; height: 64px; cursor: pointer;
    border-radius: 4px; transition: background .15s;
  }
  #nav-left:hover,#nav-right:hover { background: rgba(255,255,255,.15); }
  #nav-left  { left:  8px; }
  #nav-right { right: 8px; }

  /* ── 右侧面板 ── */
  #panel {
    width: 340px; display: flex; flex-direction: column;
    background: #1e1e1e; border-left: 1px solid #333; overflow: hidden;
  }
  #path-info {
    padding: 10px 12px; font-size: 11px; color: #888; word-break: break-all;
    border-bottom: 1px solid #2a2a2a; flex-shrink: 0;
  }
  #path-info span { color: #aaa; }
  #overall-reason {
    padding: 10px 12px; font-size: 12px; color: #f5a623;
    border-bottom: 1px solid #2a2a2a; flex-shrink: 0; min-height: 36px;
  }
  #boxes-list { flex: 1; overflow-y: auto; padding: 8px 0; }

  .box-item {
    padding: 9px 14px; border-bottom: 1px solid #2a2a2a; cursor: default;
  }
  .box-item:hover { background: #252525; }
  .box-header {
    display: flex; align-items: center; gap: 8px; margin-bottom: 4px;
  }
  .box-dot {
    width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
  }
  .dot-correct   { background: #2ecc71; }
  .dot-wrong     { background: #e74c3c; }
  .dot-ambiguous { background: #e67e22; }
  .box-name  { font-size: 13px; font-weight: 600; }
  .box-size  { font-size: 11px; color: #666; margin-left: auto; }
  .box-reason {
    font-size: 11.5px; color: #aaa; line-height: 1.5; padding-left: 18px;
  }
  .tag-wrong     { color: #e74c3c; font-weight: 600; }
  .tag-ambiguous { color: #e67e22; font-weight: 600; }
  .tag-correct   { color: #2ecc71; }

  /* ── 底部导航 ── */
  #bottombar {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 16px; background: #111; border-top: 1px solid #333;
    flex-shrink: 0;
  }
  #progress-bar-wrap {
    flex: 1; height: 6px; background: #333; border-radius: 3px; overflow: hidden;
  }
  #progress-bar { height: 100%; background: #4fc3f7; border-radius: 3px; width: 0%; }
  #jump-input {
    width: 60px; padding: 4px 6px; background: #2a2a2a; border: 1px solid #444;
    color: #eee; border-radius: 4px; font-size: 13px; text-align: center;
  }
  .btn {
    padding: 5px 14px; background: #2a2a2a; border: 1px solid #555; color: #ccc;
    border-radius: 4px; cursor: pointer; font-size: 13px; transition: background .15s;
  }
  .btn:hover { background: #3a3a3a; }

  #reload-btn {
    padding: 5px 14px; background: #1a3a2a; border: 1px solid #2ecc71; color: #2ecc71;
    border-radius: 4px; cursor: pointer; font-size: 13px; transition: background .15s;
  }
  #reload-btn:hover { background: #2ecc71; color: #000; }
  #reload-btn.loading { opacity: .5; cursor: wait; }
</style>
</head>
<body>

<!-- 顶部 -->
<div id="topbar">
  <h1>🔍 AI 标注查看器</h1>
  <span id="verdict-badge" class="verdict-badge"></span>
  <span id="ts-info" style="font-size:11px;color:#666;"></span>
  <button id="reload-btn" onclick="reloadData()">⟳ 刷新</button>
  <span id="counter">0 / 0</span>
</div>

<!-- 主体 -->
<div id="main">

  <!-- 图像 -->
  <div id="img-area">
    <div id="loading">加载中...</div>
    <img id="frame-img" style="display:none;" alt="frame">
    <button id="nav-left"  onclick="go(-1)">&#8249;</button>
    <button id="nav-right" onclick="go(+1)">&#8250;</button>
  </div>

  <!-- 右侧面板 -->
  <div id="panel">
    <div id="path-info">路径：<span id="img-path">-</span></div>
    <div id="overall-reason"></div>
    <div id="boxes-list"></div>
  </div>

</div>

<!-- 底部 -->
<div id="bottombar">
  <button class="btn" onclick="go(-1)">◀ 上一帧</button>
  <div id="progress-bar-wrap"><div id="progress-bar"></div></div>
  <button class="btn" onclick="go(+1)">下一帧 ▶</button>
  <input id="jump-input" type="number" min="1" placeholder="跳转" title="输入帧号回车跳转">
  <button class="btn" onclick="jumpTo()">跳转</button>
</div>

<script>
let cur = 0, total = 0;

async function init() {
  const r = await fetch('/api/info').then(r => r.json());
  total = r.total;
  if (total === 0) {
    document.getElementById('loading').textContent = '无数据，请检查日志文件和过滤条件';
    return;
  }
  loadFrame(0);
}

async function loadFrame(idx) {
  if (idx < 0) idx = 0;
  if (idx >= total) idx = total - 1;
  cur = idx;

  document.getElementById('loading').style.display = 'block';
  document.getElementById('frame-img').style.display = 'none';

  const d = await fetch(`/api/frame/${idx}`).then(r => r.json());
  if (d.error) return;

  // 图像
  const img = document.getElementById('frame-img');
  img.src = 'data:image/jpeg;base64,' + d.img_b64;
  img.onload = () => {
    document.getElementById('loading').style.display = 'none';
    img.style.display = 'block';
  };

  // 顶部
  document.getElementById('counter').textContent = `${idx+1} / ${total}`;
  const badge = document.getElementById('verdict-badge');
  const tsInfo = document.getElementById('ts-info');
  if (d.has_ambiguous) {
    badge.textContent = '⚠ 歧义';
    badge.className = 'verdict-badge badge-ambiguous';
  } else if (d.verdict === 'reject') {
    badge.textContent = '✘ 错误';
    badge.className = 'verdict-badge badge-reject';
  } else {
    badge.textContent = '✔ 接受';
    badge.className = 'verdict-badge badge-accept';
  }
  const sz = d.img_size.length ? `${d.img_size[0]}×${d.img_size[1]}` : '';
  tsInfo.textContent = `${d.timestamp}  ${sz}`;

  // 路径
  const parts = d.img_path.split('/');
  document.getElementById('img-path').textContent = parts.slice(-3).join('/');
  document.getElementById('img-path').title = d.img_path;

  // 总体原因
  document.getElementById('overall-reason').textContent = d.reason || '';

  // 逐框列表
  const list = document.getElementById('boxes-list');
  list.innerHTML = '';
  d.boxes.forEach(b => {
    const item = document.createElement('div');
    item.className = 'box-item';

    const statusTag = b.status === 'correct'
      ? '<span class="tag-correct">✔ 正确</span>'
      : b.status === 'ambiguous'
        ? '<span class="tag-ambiguous">⚠ 歧义</span>'
        : '<span class="tag-wrong">✘ 错误</span>';
    const dotCls = 'dot-' + b.status;

    item.innerHTML = `
      <div class="box-header">
        <div class="box-dot ${dotCls}"></div>
        <span class="box-name">框${b.id} · ${b.name}</span>
        <span class="box-size">${b.w}×${b.h}px</span>
        ${statusTag}
      </div>
      ${b.reason ? `<div class="box-reason">${b.reason}</div>` : ''}
    `;
    list.appendChild(item);
  });
  if (d.boxes.length === 0) {
    list.innerHTML = '<div style="padding:16px;color:#555;font-size:12px;">无标注框记录</div>';
  }

  // 进度条
  document.getElementById('progress-bar').style.width = `${(idx+1)/total*100}%`;
  document.getElementById('jump-input').value = idx + 1;
}

function go(delta) { loadFrame(cur + delta); }

async function reloadData() {
  const btn = document.getElementById('reload-btn');
  btn.classList.add('loading');
  btn.textContent = '⟳ 加载中...';
  const r = await fetch('/api/reload').then(r => r.json());
  total = r.total;
  btn.classList.remove('loading');
  btn.textContent = '⟳ 刷新';
  // 停在当前帧，若当前帧超出范围则跳到最后一帧
  loadFrame(Math.min(cur, total - 1));
}

function jumpTo() {
  const v = parseInt(document.getElementById('jump-input').value, 10);
  if (!isNaN(v)) loadFrame(v - 1);
}

document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft'  || e.key === 'a' || e.key === 'A') go(-1);
  if (e.key === 'ArrowRight' || e.key === 'd' || e.key === 'D') go(+1);
});
document.getElementById('jump-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') jumpTo();
});

init();
</script>
</body>
</html>
"""


# ──────────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI 标注错误查看器 — 仅审核 RedLight / BlackLight 两类",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log",    default=DEFAULT_LOG,
                        help="JSONL 日志文件路径（ai_annotation_validator.py 生成）")
    parser.add_argument("--filter", default="reject",
                        choices=["reject", "accept", "wrong", "all"],
                        help="过滤模式: reject=全部拒绝(默认), accept=全部接受, wrong=仅明确错误, all=全部")
    parser.add_argument("--port",   type=int, default=8888, help="Web 端口（默认 8888）")
    args = parser.parse_args()

    global RECORDS, _log_path, _filter_mode
    _log_path    = args.log
    _filter_mode = args.filter
    RECORDS = load_records(args.log, args.filter)

    if not RECORDS:
        print("[WARN] 没有符合条件的记录，请检查日志路径或过滤条件。")

    host = "0.0.0.0"
    print(f"\n[INFO] 共 {len(RECORDS)} 条记录  (filter={args.filter})")
    print(f"[INFO] 浏览器访问: http://<server-ip>:{args.port}")
    print(f"[INFO] 快捷键: ← / → 或 A / D 切换帧\n")
    app.run(host=host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
