import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils.plotting import colors


def set_custom_class_colors():
    class_color_map = {
        0: (255, 0, 0),  # 红灯 - 红色
        1: (0, 255, 0),  # 绿灯 - 绿色
        2: (255, 255, 0),  # 黄灯 - 黄色
        3: (0, 0, 0),  # 黑灯 - 黑色
        4: (0, 0, 255),  # 可移动交通灯 - 蓝色
        5: (128, 0, 128),  # 多语义交通灯 - 紫色
        6: (255, 128, 0),  # 交通锥 - 橙色
        7: (255, 0, 255),  # 警示三角牌 - 品红
        8: (0, 255, 255),  # 灭火器 - 青色
        9: (128, 255, 0),  # 绿植杂物 - 黄绿色
        10: (255, 192, 203),  # 纸类 - 粉色
        11: (139, 69, 19),  # 瓶子 - 棕色
        12: (128, 128, 128),  # 未知 - 灰色
    }
    for cls_id, rgb in class_color_map.items():
        if 0 <= cls_id < len(colors.palette):
            colors.palette[cls_id] = rgb


def main():
    set_custom_class_colors()

    # val_txt = ROOT / 'dataset/val.txt'
    # val_txt = '/pcpt/pcpt/project/atlb/vla/ultralytics/dataset/small_obj_val.txt'
    val_txt = "/pcpt/pcpt/project/atlb/vla/ultralytics/dataset/val_data_filter.txt"
    with open(val_txt) as f:
        sources = [line.strip() for line in f if line.strip()]

    model = YOLO(str(ROOT / "ultralytics/cfg/models/26/yolo26m-lt.yaml"))
    model.load("/pcpt/pcpt/project/atlb/vla/ultralytics/runs/detect/trafficlights_logsim/train3/weights/best.pt")

    for i, src in enumerate(sources):
        model.predict(
            source=src,
            imgsz=1280,
            conf=0.3,
            iou=0.7,
            device=0,
            save=True,
            show=False,
            line_width=4,
            show_conf=True,
            show_labels=True,
        )
        if (i + 1) % 100 == 0:
            print(f"[{i + 1}/{len(sources)}] done")

    print("Inference done.")


if __name__ == "__main__":
    main()
