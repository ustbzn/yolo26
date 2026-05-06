
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def main():
    model = YOLO(str(ROOT / 'ultralytics/cfg/models/26/yolo26m.yaml'))
    model.load(str(ROOT / 'yolo26m.pt'))
    # model = YOLO(str(ROOT / 'ultralytics/cfg/models/26/yolo26l.yaml'))
    # model.load(str(ROOT / 'yolo26l.pt'))
    # model.load('/pcpt/pcpt/project/atlb/vla/ultralytics/runs/detect/train4/weights/best.pt')
    # model.load('/pcpt/pcpt/project/atlb/vla/ultralytics/runs/detect/train7/weights/best.pt')

    model.train(
        # data=str(ROOT / 'ultralytics/cfg/datasets/mogo_small_obj.yaml'),
        data=str(ROOT / 'ultralytics/cfg/datasets/mogo_TrafficLights_small_obj.yaml'),
        cache=False,
        imgsz=1600,
        epochs=100,
        batch=96,
        save_period=1,
        patience=20,
        workers=4,
        device=[0, 1, 2, 3, 4, 5, 6, 7],
        # device=0,
        optimizer='MuSGD',
        multi_scale=0.2, #[1280, 1920]
        scale=0.3, # 0.5
        mixup=0.1, # 0.0
        # mosaic=0.0,
        # erasing=0.0,
        # copy_paste=0.0,
        # scale=0.25,
        # mixup=0.15,
        # cls=0.5,
        project='trafficlights_small_obj',
    )


if __name__ == '__main__':
    main()
    # /pcpt/pcpt/project/baozhiping/ultralytics-main_1/RBS-19_60/0706_000116.jpg
    
