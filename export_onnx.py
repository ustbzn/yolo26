import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default=str(ROOT / "runs/detect/train4/weights/best.pt"),
        help="Path to .pt model file",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size")
    parser.add_argument("--batch", type=int, default=2, help="Batch size for export")
    parser.add_argument("--device", type=str, default="0", help="Export device, e.g. '0' or 'cpu'")
    parser.add_argument("--opset", type=int, default=None, help="ONNX opset version")
    parser.add_argument("--dynamic", action="store_true", help="Enable dynamic input shape")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX graph")
    parser.add_argument("--half", action="store_true", help="Export with FP16")
    parser.add_argument("--int8", action="store_true", help="INT8 export (requires calibration data)")
    parser.add_argument("--nms", action="store_true", help="Fuse NMS into exported model if supported")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Dataset yaml path (recommended when using --int8)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))

    export_args = {
        "format": "onnx",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "dynamic": args.dynamic,
        "simplify": args.simplify,
        "half": args.half,
        "int8": args.int8,
        "nms": args.nms,
    }
    if args.opset is not None:
        export_args["opset"] = args.opset
    if args.data is not None:
        export_args["data"] = args.data

    exported_file = model.export(**export_args)
    print(f"ONNX export completed: {exported_file}")


if __name__ == "__main__":
    main()
