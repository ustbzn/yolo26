import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def main():
    model = YOLO(str(ROOT / "ultralytics/cfg/models/26/yolo26m.yaml"))
    model.load(str(ROOT / "yolo26m.pt"))
    # model = YOLO(str(ROOT / 'ultralytics/cfg/models/26/yolo26l.yaml'))
    # model.load(str(ROOT / 'yolo26l.pt'))
    # model.load('/pcpt/pcpt/project/atlb/vla/ultralytics/runs/detect/train4/weights/best.pt')
    # model.load('/pcpt/pcpt/project/atlb/vla/ultralytics/runs/detect/train7/weights/best.pt')

    model.train(
        # data=str(ROOT / 'ultralytics/cfg/datasets/mogo.yaml'),
        data=str(ROOT / "ultralytics/cfg/datasets/mogo_small_obj.yaml"),
        cache=False,
        # imgsz=1280,
        imgsz=640,
        epochs=150,
        # batch=64,
        batch=128,
        # batch=1024,
        save_period=2,
        # close_mosaic=10,
        workers=4,
        device=[0, 1, 2, 3, 4, 5, 6, 7],
        # device=0,
        # optimizer='SGD',  # using SGD,auto
        optimizer="MuSGD",  # using SGD,auto
        # BlackLight tuning: class 3 is "light off" state
        # lr0=0.005,
        # lrf=0.01,
        # warmup_epochs=5.0,
        # cls=0.8,
        # hsv_s=0.45,
        # hsv_v=0.25,
        # mosaic=0.35,
        # mixup=0.0,
        # scale=0.3,
        project="small_obj",
        # name='exp',
    )


if __name__ == "__main__":
    main()
    # /pcpt/pcpt/project/baozhiping/ultralytics-main_1/RBS-19_60/0706_000116.jpg
