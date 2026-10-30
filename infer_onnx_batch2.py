import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils.plotting import colors
import cv2
import numpy as np

ONNX = "/pcpt/pcpt/project/atlb/vla/ultralytics/runs/detect/trafficlights_logsim_finetune_timer-dla/3600frames_lr1e-3/weights/best.onnx"
SOURCE = "/pcpt/pcpt/project/atlb/vla/ultralytics/case_data_mau/RBS-167302-logsim-60/images"


def set_custom_class_colors():
    class_color_map = {
        0: (255, 0, 0), 1: (0, 255, 0), 2: (255, 255, 0),
        3: (0, 0, 0), 4: (0, 0, 255), 5: (128, 0, 128),
        6: (255, 165, 0), 7: (0, 255, 255), 8: (255, 192, 203), 9: (128, 128, 128),
    }
    for cls_id, rgb in class_color_map.items():
        if 0 <= cls_id < len(colors.palette):
            colors.palette[cls_id] = rgb


def label_frame(img, text):
    """在帧左上角加文件名标注（带半透明黑底）。"""
    img = img.copy()
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(img, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(img, text, (10, th + 10), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
    return img


def main():
    set_custom_class_colors()

    # 取前 4 帧，按每批 2 帧分两次推理
    imgs = sorted(Path(SOURCE).glob("*.jpg"))[:4]
    batches = [imgs[0:2], imgs[2:4]]

    model = YOLO(ONNX, task="detect")

    batch_rows = []
    for i, batch in enumerate(batches):
        print(f"==> 第 {i + 1} 次推理: {[p.name for p in batch]}")
        results = model.predict(
            source=[str(p) for p in batch],
            imgsz=[736, 1280],
            conf=0.3,
            iou=0.7,
            save=True,
            show=False,
            line_width=4,
            show_conf=True,
            show_labels=True,
        )
        # 该 batch 内 2 帧各自加文件名后水平拼接成一行
        frames = [
            label_frame(r.plot(line_width=4), f"batch{i + 1} | {p.name}")
            for r, p in zip(results, batch)
        ]
        batch_rows.append(np.hstack(frames))

    # 两个 batch 上下排列
    grid = np.vstack(batch_rows)
    out_path = ROOT / "runs/detect/batch2_compare.jpg"
    cv2.imwrite(str(out_path), grid)
    print(f"并排结果保存到: {out_path}")

    print("Inference done. 共 2 次推理, 4 帧。")


if __name__ == "__main__":
    main()
