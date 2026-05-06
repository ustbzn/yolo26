# mkdir -p runs/detect/train

# python train.py
python train_trafficlights.py
# python train.py > runs/detect/train_$(date +"%Y%m%d_%H%M%S").log 2>&1
# ps -ef|grep python |grep -v grep|awk '{print $2}'|xargs kill -9

cd /pcpt/pcpt/project/atlb/model/multimodal_bevfusion

bash fake_gpu.sh