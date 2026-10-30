# AI 标注验收工具 使用说明

> 利用 Claude 视觉模型对交通灯标注数据进行批量质量验收，并通过 Web 界面可视化查看错误结果。

---

## 工具概览

| 脚本                         | 功能                                                     |
| ---------------------------- | -------------------------------------------------------- |
| `ai_annotation_validator.py` | 调用 Claude API 逐图验收标注质量，输出合格/拒绝/歧义列表 |
| `error_viewer.py`            | Web 可视化查看器，浏览被拒绝图像及 Claude 的逐框分析原因 |

---

## 一、ai_annotation_validator.py

### 验收规则

- 一张图片中**只要有一个标注框错误**，整张图排除
- **全部正确**才保留进输出列表
- **看不清楚 / 图像模糊 / 严重遮挡** → 一律视为错误（记入歧义列表）

### 标注类别

| ID  | 类别名                     | 含义                     |
| --- | -------------------------- | ------------------------ |
| 0   | RedLight                   | 红灯（亮起）             |
| 1   | GreenLight                 | 绿灯（亮起）             |
| 2   | YellowLight                | 黄灯（亮起）             |
| 3   | BlackLight                 | 黑灯 / 熄灭灯            |
| 4   | MovingTrafficLight         | 移动交通灯（施工车辆上） |
| 5   | MultiSemanticTrafficLights | 多义交通灯               |

### 环境要求

```bash
pip install anthropic pillow flask

# 设置 API Key（二选一）
export ANTHROPIC_AUTH_TOKEN=your_key
export ANTHROPIC_API_KEY=your_key

# 可选：自定义代理地址
export ANTHROPIC_BASE_URL=https://api.aipaibox.com
```

### 输出文件

运行后自动生成以下文件（默认均在 `tools/` 目录下）：

| 文件                         | 内容                                          |
| ---------------------------- | --------------------------------------------- |
| `ai_validated.txt`           | ✅ 全部正确的图像路径（与训练 txt 格式相同）  |
| `ai_rejected.txt`            | ❌ 所有被拒绝的图像路径                       |
| `ai_ambiguous.txt`           | ⚠️ 因歧义/看不清被拒的图像路径                |
| `ai_validated.jsonl`         | 📋 逐帧完整分析记录（JSONL 格式，每行一张图） |
| `ai_validator_progress.json` | 🔖 断点续跑进度文件（自动维护）               |

### 常用命令

**① 快速抽样测试（推荐先跑）**

```bash
# 先随机抽取样本
python3 -c "
import random, pathlib
random.seed(42)
src = pathlib.Path('/pcpt/pcpt/data/2D-Det/COS/data_clean.txt')
dst = pathlib.Path('/pcpt/pcpt/data/2D-Det/COS/sample_300.txt')
lines = [l.strip() for l in src.read_text().splitlines() if l.strip()]
dst.write_text('\n'.join(random.sample(lines, 300)) + '\n')
"

# 跑抽样验收
python tools/ai_annotation_validator.py \
  --input /pcpt/pcpt/data/2D-Det/COS/sample_300.txt \
  --output /pcpt/pcpt/data/2D-Det/COS/sample_300_validated.txt \
  --progress tools/sample_300_progress.json \
  --log tools/sample_300_validated.jsonl \
  --ambiguous tools/sample_300_ambiguous.txt \
  --rejected tools/sample_300_rejected.txt \
  --workers 8
```

**② 全量验收（后台运行）**

```bash
nohup python -u tools/ai_annotation_validator.py \
  --workers 8 \
  > /tmp/ai_validator.log 2>&1 &
echo $! # 记录 PID

# 实时查看进度
tail -f /tmp/ai_validator.log
```

**③ 断点续跑（中断后直接重跑同一命令即可）**

```bash
# 脚本自动读取进度文件，跳过已处理的图像
python tools/ai_annotation_validator.py --workers 8
```

**④ 仅重建输出文件（不调用 API）**

```bash
# 进度文件已有结果，重新生成 validated.txt
python tools/ai_annotation_validator.py --regen-output
```

**⑤ 使用 Batch API（省钱 50%，适合大批量离线任务）**

```bash
python tools/ai_annotation_validator.py --batch --batch-size 40
```

### 全部参数

| 参数              | 默认值                             | 说明                           |
| ----------------- | ---------------------------------- | ------------------------------ |
| `--input`         | `data_clean.txt`                   | 输入图像列表 txt               |
| `--output`        | `ai_validated.txt`                 | 验收通过的图像列表             |
| `--progress`      | `tools/ai_validator_progress.json` | 进度文件（断点续跑用）         |
| `--log`           | `tools/ai_validated.jsonl`         | 逐帧详细分析日志               |
| `--ambiguous`     | `tools/ai_ambiguous.txt`           | 歧义拒绝列表                   |
| `--rejected`      | `tools/ai_rejected.txt`            | 全部拒绝列表                   |
| `--model`         | `claude-sonnet-4-6`                | Claude 模型 ID                 |
| `--workers`       | `8`                                | 并发线程数                     |
| `--limit`         | `0`（全部）                        | 最多处理张数，0=不限           |
| `--skip-no-label` | 关闭                               | 无标注文件的图像直接排除       |
| `--batch`         | 关闭                               | 使用 Batch API（半价异步）     |
| `--batch-size`    | `40`                               | Batch API 每批图像数           |
| `--regen-output`  | 关闭                               | 从进度文件重建输出，不调用 API |

### JSONL 日志格式

每行一条 JSON 记录，结构如下：

```json
{
  "img_path": "/pcpt/.../images/xxx.jpg",
  "label_path": "/pcpt/.../labels_6cls/xxx.txt",
  "timestamp": "2026-04-16 10:00:00",
  "img_size": [1920, 1080],
  "boxes": [{ "id": 1, "cls_id": 0, "name": "RedLight", "x1": 100, "y1": 200, "x2": 120, "y2": 230, "w": 20, "h": 30 }],
  "verdict": "reject",
  "has_ambiguous": false,
  "reason": "错误框(1/3): 框2[GreenLight]: 灯体实际显示为红色",
  "results": [
    { "id": 1, "label": "RedLight", "is_correct": true, "is_ambiguous": false, "reason": "灯体清晰显示红色" },
    { "id": 2, "label": "GreenLight", "is_correct": false, "is_ambiguous": false, "reason": "灯体实际为红色" },
    { "id": 3, "label": "BlackLight", "is_correct": false, "is_ambiguous": true, "reason": "图像过暗无法判断" }
  ]
}
```

---

## 二、error_viewer.py

### 功能

- 浏览器可视化查看被拒绝的图像
- 标注框按判定结果着色：🟢 正确 / 🔴 错误 / 🟠 歧义
- 右侧面板显示 Claude 对每个框的分析原因
- 支持帧跳转、键盘快捷键

### 启动

```bash
# 后台启动（默认端口 8888，加载全部拒绝帧）
nohup python -u tools/error_viewer.py \
  --log tools/sample_300_validated.jsonl \
  --filter reject \
  --port 8888 \
  > /tmp/error_viewer.log 2>&1 &

# 浏览器访问
http://localhost:8888
```

### 过滤模式

| `--filter`       | 显示内容                   |
| ---------------- | -------------------------- |
| `reject`（默认） | 所有被拒绝的图像（含歧义） |
| `wrong`          | 仅明确标注错误的图像       |
| `ambiguous`      | 仅歧义 / 看不清的图像      |
| `all`            | 全部图像（含通过的）       |

**切换模式示例：**

```bash
pkill -f error_viewer.py

# 只看歧义
nohup python -u tools/error_viewer.py --log tools/sample_300_validated.jsonl --filter ambiguous --port 8888 > /tmp/error_viewer.log 2>&1 &

# 只看明确错误
nohup python -u tools/error_viewer.py --log tools/sample_300_validated.jsonl --filter wrong --port 8888 > /tmp/error_viewer.log 2>&1 &
```

### 界面说明

```
┌─────────────────────────────────────────────────────────┐
│ 🔍 AI标注错误查看器  [✘ 错误]  2026-04-16  1920×1080  5/132 │  ← 顶栏
├───────────────────────────────────────┬─────────────────┤
│                                       │ 路径: .../xxx.jpg│
│                                       │ 错误框(1/3):...  │  ← 总体原因
│         图像（带颜色标注框）            ├─────────────────┤
│                                       │ 🔴 框1·RedLight  │
│    ◀ 上一帧              下一帧 ▶     │    20×30px  ✘错误│
│                                       │    灯体实为绿色  │
│                                       │ 🟢 框2·BlackLight│
│                                       │    4×12px   ✔正确│
├───────────────────────────────────────┴─────────────────┤
│  ◀ 上一帧  ████████░░░░░░░░░░  下一帧 ▶   [跳转: 5]    │  ← 底栏
└─────────────────────────────────────────────────────────┘
```

### 快捷键

| 按键              | 功能            |
| ----------------- | --------------- |
| `←` / `→`         | 上一帧 / 下一帧 |
| `A` / `D`         | 上一帧 / 下一帧 |
| 底部输入框 + 回车 | 跳转到指定帧号  |

### 全部参数

| 参数       | 默认值                             | 说明                                       |
| ---------- | ---------------------------------- | ------------------------------------------ |
| `--log`    | `tools/sample_300_validated.jsonl` | JSONL 日志文件路径                         |
| `--filter` | `reject`                           | 过滤模式：reject / wrong / ambiguous / all |
| `--port`   | `8888`                             | Web 服务端口                               |

---

## 三、典型工作流程

```
1. 抽样验收（评估质量）
   └─ 随机抽 300 帧 → ai_annotation_validator.py --limit 300
           ↓
2. 查看错误（定位问题）
   └─ error_viewer.py --filter reject
           ↓
3. 分析结果
   ├─ 接受率高 → 全量运行验收
   └─ 接受率低 → 排查标注规范问题
           ↓
4. 全量验收（后台长跑）
   └─ ai_annotation_validator.py --workers 8
           ↓
5. 输出训练数据
   └─ ai_validated.txt（格式与原始训练 txt 完全一致）
```

---

## 四、费用参考

| 模式                   | 速度         | 单价（估算） | 20万张费用（估算） |
| ---------------------- | ------------ | ------------ | ------------------ |
| 并发模式（8线程）      | ~50张/min    | ~$0.008/张   | ~$1,600            |
| Batch API（`--batch`） | 异步，数小时 | ~$0.004/张   | ~$800              |

> 建议先用 `--limit 300` 抽样验证效果，再决定是否全量运行。
