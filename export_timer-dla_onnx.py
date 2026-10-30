import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

# 训练得到的 DLA 友好权重（yolo26m-timer-dla.yaml）
WEIGHTS = "/pcpt/pcpt/project/atlb/vla/ultralytics/runs/detect/trafficlights_logsim_finetune_timer-dla_m_dynamic/3600frames_lr1e-3/weights/best.pt"


def main():
    model = YOLO(str(WEIGHTS))

    onnx_path = model.export(
        format='onnx',
        # imgsz=[736, 1280],   # 与训练一致
        # imgsz=[384, 640],   # 与训练一致
        imgsz=[480, 800],   # 与训练一致
        
        
        opset=17,            # TensorRT 8.6 / DLA 3.14 支持到 opset 17
        batch=2,             # DLA 部署常用静态 batch
        dynamic=False,       # DLA 需固定输入尺寸
        simplify=True,       # 简化计算图
        half=False,          # FP 由 TensorRT 构建 engine 时再定 (INT8/FP16)
        device='cpu',        # 导出在 CPU 上做即可
    )
    print(f'ONNX exported to: {onnx_path}')


if __name__ == '__main__':
    main()


#  trtexec --onnx=best.onnx --useDLACore=0 --allowGPUFallback --fp16 --saveEngine=best_dla.engine
