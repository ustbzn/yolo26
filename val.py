import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def main():
    model = YOLO(str(ROOT / "ultralytics/cfg/models/26/yolo26m-lt.yaml"))
    model.load(
        "/pcpt/pcpt/project/atlb/vla/ultralytics/runs/detect/trafficlights_logsim/30wframes_imgsz_1280_bs5122/weights/best.pt"
    )

    metrics = model.val(
        data=str(ROOT / "ultralytics/cfg/datasets/mogo.yaml"),
        split="val",
        imgsz=1280,
        conf=0.3,
        iou=0.7,
        device=0,
        save_json=False,
        plots=True,
    )

    print("Validation done.")
    print(metrics)


if __name__ == "__main__":
    main()
