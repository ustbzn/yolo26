import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from ultralytics.utils.plotting import colors


def set_custom_class_colors():
    # RGB colors for class IDs 0..5
    class_color_map = {
        0: (255, 0, 0),    # red
        1: (0, 255, 0),    # green
        2: (255, 255, 0),  # yellow
        3: (0, 0, 0),      # black
        4: (0, 0, 255),    # blue
        5: (128, 0, 128),  # purple
    }

    for cls_id, rgb in class_color_map.items():
        if 0 <= cls_id < len(colors.palette):
            colors.palette[cls_id] = rgb


def main():
    set_custom_class_colors()

    model = YOLO(str(ROOT / 'ultralytics/cfg/models/26/yolo26m-lt.yaml'))
    model.load('/pcpt/pcpt/project/atlb/vla/ultralytics/runs/detect/trafficlights_logsim/30wframes_imgsz_1280_bs5122/weights/best.pt')
    # model.load('/pcpt/pcpt/project/atlb/vla/ultralytics/runs/detect/train17/weights/epoch99.pt')
    results = model.predict(
        # source='/pcpt/pcpt/project/baozhiping/ultralytics-main_1/yolo26_1/',
        # source='/pcpt/pcpt/project/baozhiping/ultralytics-main_1/yolo26_1_30/',
        source='/pcpt/pcpt/project/baozhiping/ultralytics-main_1/yolo26_2_60/',
        # source='/pcpt/pcpt/project/baozhiping/xinjiapo',
        # source='/pcpt/pcpt/project/atlb/vla/ultralytics/case_data_mau/shunyi_case2/images',
        # source='/pcpt/pcpt/project/baozhiping/xinjiapo_2',
        # source='/pcpt/pcpt/project/baozhiping/xinjiapo_3',
        
        
        imgsz=1600,
        conf=0.6,
        iou=0.7,
        device=0,
        save=True,
        show=False,

        # 👇 关键参数
        line_width=4,      # 框变细
        show_conf=True,   # 不显示置信度
        show_labels=True,  # 只显示类别（如果你还想隐藏类别也可以关）
    )

    print("Inference done.")


if __name__ == '__main__':
    main()