# GT Viewer — 交通灯标注审核/修正工具

多人协同的 Web 可视化工具，用于查看、修正、绘制和保存 YOLO 格式的 10cls 交通灯标注。
支持两种数据来源：

- 通过 `--data-root` 扫描 case 数据目录下的 `*/images/*`
- 通过 `--train-txt` / `--val-txt` 读取图片路径列表

---

## 环境依赖

```bash
pip install flask pillow
```

---

## case_data_mau 快速启动

```bash
cd /pcpt/pcpt/project/atlb/vla/ultralytics

setsid python -u tools/gt_viewer.py \
  --data-root /pcpt/pcpt/project/atlb/vla/ultralytics/case_data_mau \
  --host 0.0.0.0 \
  --port 8888 \
  > /tmp/case_data_gt_viewer.log 2>&1 < /dev/null &
```

查看启动日志：

```bash
tail -20 /tmp/case_data_gt_viewer.log
```

---

## 先模型推理，再进入工具修正

如果 `labels_10cls` 为空，可先用模型生成初始标注。当前 case 数据使用的 10 类 timer 模型权重与配置：

```bash
# 权重
/pcpt/pcpt/project/atlb/vla/ultralytics/runs/detect/trafficlights_logsim_finetune_timer/3600frames_lr1e-45/weights/best.pt
# 模型配置（参考 infer.py：先用 yaml 建模型再 load 权重）
/pcpt/pcpt/project/atlb/vla/ultralytics/ultralytics/cfg/models/26/yolo26m-timer.yaml
```

启动时自动推理缺失 label：

```bash
cd /pcpt/pcpt/project/atlb/vla/ultralytics

python tools/gt_viewer.py \
  --data-root /pcpt/pcpt/project/atlb/vla/ultralytics/case_data_mau \
  --weights /pcpt/pcpt/project/baozhiping/ultralytics/vision_models/yolo26_sp_v18/yolo26_v18.pt \
  --model-cfg /pcpt/pcpt/project/atlb/vla/ultralytics/ultralytics/cfg/models/26/yolo26m-timer.yaml \
  --infer-on-start \
  --imgsz 1600 \
  --conf 0.3 \
  --iou 0.7 \
  --device 0 \
  --batch 1 \
  --host 0.0.0.0 \
  --port 8889
```

说明：

- 默认只为缺失的 label 生成 `labels_10cls/*.txt`
- 不会覆盖已有 label
- 如需覆盖已有 label，显式追加 `--overwrite-labels`
- `--weights` 和 `--model-cfg` 已默认指向上述 timer 模型，省略时即用默认值
- 留空 `--model-cfg`（传 `""`）则退回直接用权重加载，不走 yaml 建模
- 推理完成后会继续启动 Web 标注工具

---

## 重启/关闭

```bash
ps -ef | awk '/gt_viewer[.]py/ && /case_data_mau/ {print $2}' | xargs -r kill
```

重新启动使用「case_data_mau 快速启动」命令。

---

## 远程访问（SSH 端口转发）

在**本地电脑**终端执行：

```bash
ssh -p 46852 -L 8881:127.0.0.1:8889 root@115.190.76.196
ssh -p 31970 -L 8889:127.0.0.1:8889 root@115.190.76.196
```

连上后浏览器打开：

```
http://127.0.0.1:8889
```

如果本地 `8888` 端口被占用，改用：

```bash
ssh -p 31970 -L 18888:127.0.0.1:8888 root@115.190.76.196
```

浏览器打开：

```
http://127.0.0.1:18888
```

---

## 服务器直接访问

```
http://9.3.168.225:8888
```

> 默认端口 `8888`，可通过 `--port` 修改。

---

## 数据配置

case 数据目录结构：

```text
case_data_mau/
  shunyi_case/
    images/
    labels_10cls/
  zhuhai_case2/
    images/
    labels_10cls/
  zhuhai_case3/
    images/
    labels_10cls/
```

工具会将图片路径中的 `images/` 自动映射到 `labels_10cls/`：

```text
.../images/0706_000001.jpg
.../labels_10cls/0706_000001.txt
```

也可以读取 txt 数据集：

```bash
python tools/gt_viewer.py \
  --train-txt /path/to/train.txt \
  --val-txt /path/to/val.txt \
  --host 0.0.0.0 \
  --port 8888
```

**10 个类别：**

| ID  | 类名                       | 颜色    |
| --- | -------------------------- | ------- |
| 0   | RedLight                   | 🔴 红   |
| 1   | GreenLight                 | 🟢 绿   |
| 2   | YellowLight                | 🟡 黄   |
| 3   | BlackLight                 | ⚫ 灰   |
| 4   | MovingTrafficLight         | 🔵 蓝   |
| 5   | MultiSemanticTrafficLights | 🟣 紫   |
| 6   | BlackTimingCard            | ⚫ 深灰 |
| 7   | GreenTimingCard            | 🟢 深绿 |
| 8   | RedTimingCard              | 🔴 暗红 |
| 9   | YellowTimingCard           | 🟡 金黄 |

---

## 操作说明

### 1. 登录

打开页面后输入你的**用户名**（如 `Alice`），点击「进入」。  
多人同时使用时每人输入不同名字，系统自动分配颜色。

---

### 2. 浏览帧

| 操作         | 按钮          | 快捷键     |
| ------------ | ------------- | ---------- |
| 上一帧       | ◀ 上一帧      | `←` 或 `A` |
| 下一帧       | 下一帧 ▶      | `→` 或 `D` |
| 跳到指定帧   | 输入帧号 → Go | 输入框回车 |
| 下一个未审帧 | ⏭ 下一未审帧  | `N`        |

---

### 3. 标注

| 操作             | 含义           | 行为               |
| ---------------- | -------------- | ------------------ |
| ✔ **Accept**     | 标注正确，保留 | 自动跳到下一未审帧 |
| ✘ **Reject**     | 标注有误，剔除 | 停留当前帧         |
| 再次点击同一按钮 | 取消本次标注   | —                  |

快捷键：`Q` = Accept，`E` = Reject

---

### 4. 修改标注框

| 操作             | 方法                                              |
| ---------------- | ------------------------------------------------- |
| 新增框           | 在图像空白处按住鼠标左键拖拽                      |
| 选择框           | 点击已有框                                        |
| 修改类别         | 选中框后在右侧类别下拉框修改                      |
| 删除框           | 选中框后点「删除此框」或按 `Delete` / `Backspace` |
| 保存当前帧 label | 点击右侧「保存当前帧标注」                        |

新增框默认类别可在右下角「拉框默认类别」中选择。

---

### 5. 多人协同

- 顶部「在线用户栏」实时显示所有在线人员及其当前帧号
- 每 2 秒自动同步标注状态
- 他人标注的帧会显示 `✔ ACCEPTED (Bob)` 等提示
- **⏭ 下一未审帧** 自动跳过所有已被任何人标注的帧，高效分工

---

### 6. 保存结果

1. 右下角输入框确认输出路径（默认如下）
2. 点击「💾 保存结果」

```
/pcpt/pcpt/data/2D-Det/COS/TXT/10cls/reviewed_output/accepted.txt
```

输出格式与原始 txt **完全一致**，每行一条图像绝对路径，仅包含 Accept 的帧。

---

## 断点续标

每次 Accept / Reject 操作都会**实时写盘**：

```
tools/gt_viewer_case_data_mau_progress.json
```

如果没有使用 `--data-root`，默认进度文件为：

```
tools/gt_viewer_progress.json
```

服务重启后自动恢复上次进度，标注记录不丢失。

---

## 常见问题

**Q：图像显示「图像不存在」？**  
检查 txt 文件中的路径是否挂载正常。

**Q：label 文件不存在（橙色警告）？**  
说明该帧还没有 `labels_10cls/*.txt`。可先用 `--infer-on-start` 批量生成，也可以手动画框后保存当前帧标注。

**Q：想重置所有进度？**  
删除对应进度文件，重启服务即可。case 数据默认是 `tools/gt_viewer_case_data_mau_progress.json`。

**Q：多人同时 Accept 同一帧？**  
后操作的人会覆盖前一人的标注，覆盖层显示最新标注人。

**Q：浏览器打不开？**  
先在服务器检查服务和端口：

```bash
ps -ef | awk '/gt_viewer[.]py/ {print}'
netstat -ltnp 2> /dev/null | awk '$4 ~ /:8888$/ {print}'
curl -I http://127.0.0.1:8888/
```

如果服务器本机正常但本地打不开，使用 SSH 端口转发访问。
