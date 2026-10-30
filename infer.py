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
        0: (255, 0, 0),      # red
        1: (0, 255, 0),      # green
        2: (255, 255, 0),    # yellow
        3: (0, 0, 0),        # black
        4: (0, 0, 255),      # blue
        5: (128, 0, 128),    # purple
        6: (255, 165, 0),    # orange
        7: (0, 255, 255),    # cyan
        8: (255, 192, 203),  # pink
        9: (128, 128, 128),  # gray
    }

    for cls_id, rgb in class_color_map.items():
        if 0 <= cls_id < len(colors.palette):
            colors.palette[cls_id] = rgb


def main():
    set_custom_class_colors()

    # model = YOLO(str(ROOT / 'ultralytics/cfg/models/26/yolo26m-lt.yaml'))
    model = YOLO(str(ROOT / 'ultralytics/cfg/models/26/yolo26m-timer-dla.yaml'))
    
    model.load('/pcpt/pcpt/project/atlb/vla/ultralytics/runs/detect/trafficlights_logsim_finetune_timer-dla_m_dynamic/3600frames_lr1e-3/weights/best.pt')
    # model.load('/pcpt/pcpt/project/atlb/vla/ultralytics/runs/detect/train17/weights/epoch99.pt')
    results = model.predict(
        # source='/pcpt/pcpt/project/baozhiping/ultralytics-main_1/yolo26_1/',
        # source='/pcpt/pcpt/project/baozhiping/ultralytics-main_1/yolo26_1_30/',
        # source='/pcpt/pcpt/project/baozhiping/ultralytics-main_1/yolo26_2_60/',
        # source='/pcpt/pcpt/project/baozhiping/xinjiapo',
        # source='/pcpt/pcpt/project/atlb/vla/ultralytics/case_data_mau/shunyi_case2/images',
        # source='/pcpt/pcpt/project/baozhiping/xinjiapo_2',
        # source='/pcpt/pcpt/project/baozhiping/xinjiapo_3',
        source = '/pcpt/pcpt/project/atlb/vla/ultralytics/case_data_mau/RBS-259715',
        # source = '/pcpt/pcpt/project/atlb/vla/ultralytics/case_data_mau/RBS-167302-logsim-60/images',
        # source = "/pcpt/pcpt/project/atlb/vla/ultralytics/case_data_mau/RBS-204458-60/images",
        # source = '/pcpt/pcpt/data/2D-Det/COS/2026_Q1/0128/badcase_zhuhai_20251203_47708_image60/images/1762843666.889003038.jpg',

        imgsz=800,
        conf=0.3,
        iou=0.7,
        device='cpu',
        save=True,
        show=False,
        # batch=2,
        # stream=False,

        # 👇 关键参数
        line_width=4,      # 框变细
        show_conf=True,   # 不显示置信度
        show_labels=True,  # 只显示类别（如果你还想隐藏类别也可以关）
    )

    print("Inference done.")


if __name__ == '__main__':
    main()