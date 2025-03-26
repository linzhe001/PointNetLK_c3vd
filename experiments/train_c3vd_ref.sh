#! /usr/bin/bash

# 设置工作目录为项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."
echo "当前工作目录: $(pwd)"

# 显示系统状态
echo "系统内存情况:"
free -h
echo "GPU内存情况:"
nvidia-smi

# 创建结果和日志目录
mkdir -p ./results
mkdir -p ./log

# Python命令
PY3="nice -n 10 python"

# 设置变量
DATASET_PATH="/mnt/c/Users/asus/Downloads/C3VD_datasets"
CATEGORY_FILE="experiments/sampledata/c3vd.txt"
NUM_POINTS=1024
DEVICE="cuda:0"
BATCH_SIZE=4
DATE_TAG=$(date +"%m%d")

# 添加在脚本开始处
echo "========== CUDA可用性检查 =========="
${PY3} -c "
import torch
print(f'CUDA是否可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA设备数量: {torch.cuda.device_count()}')
    print(f'当前CUDA设备: {torch.cuda.current_device()}')
    print(f'设备名称: {torch.cuda.get_device_name(0)}')
    # 测试简单操作速度
    import time
    cpu_tensor = torch.randn(1000, 1000)
    start = time.time()
    cpu_tensor @ cpu_tensor
    cpu_time = time.time() - start
    
    gpu_tensor = torch.randn(1000, 1000, device='cuda:0')
    start = time.time()
    gpu_tensor @ gpu_tensor
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    
    print(f'CPU矩阵乘法时间: {cpu_time:.4f}秒')
    print(f'GPU矩阵乘法时间: {gpu_time:.4f}秒')
    print(f'加速比: {cpu_time/gpu_time:.1f}x')
else:
    print('警告: CUDA不可用，将使用CPU训练（速度会非常慢）')
"

# 快速数据集检查 - 简化版
echo "========== 快速检查数据集 =========="
${PY3} -c "
import os, glob
dataset_path = '${DATASET_PATH}'
source_root = os.path.join(dataset_path, 'C3VD_ply_rot_scale_trans')
target_root = os.path.join(dataset_path, 'visible_point_cloud_ply')

# 统计场景和点云对
scenes = [os.path.basename(d) for d in glob.glob(os.path.join(source_root, '*')) if os.path.isdir(d)]
pairs_count = 0

for scene in scenes:
    source_files = glob.glob(os.path.join(source_root, scene, '????_adjusted.ply'))
    target_files = [os.path.join(target_root, scene, f'frame_{os.path.basename(f).split(\"_\")[0]}_visible.ply') 
                    for f in source_files]
    valid_pairs = sum(1 for f in target_files if os.path.exists(f))
    pairs_count += valid_pairs
    print(f'场景 {scene}: {valid_pairs}对点云')

train_size = int(pairs_count * 0.8)
test_size = pairs_count - train_size

print(f'总计: {pairs_count}对点云, 训练集约{train_size}个, 测试集约{test_size}个')
print(f'批次大小: {${BATCH_SIZE}}')

if train_size % ${BATCH_SIZE} != 0 or test_size % ${BATCH_SIZE} != 0:
    print(f'警告: 训练集或测试集大小不是批次大小的整数倍')
    print(f'最后一批次: 训练={train_size%${BATCH_SIZE}}, 测试={test_size%${BATCH_SIZE}}')
"

# 第一阶段：训练分类器
echo "========== 训练分类器 =========="
${PY3} experiments/train_classifier.py \
  -o ./results/c3vd_classifier_${DATE_TAG} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l ./log/c3vd_classifier_${DATE_TAG}.log \
  --dataset-type c3vd \
  --num-points ${NUM_POINTS} \
  --device ${DEVICE} \
  --epochs 10 \
  --batch-size ${BATCH_SIZE} \
  --workers 2 \
  --drop-last

# 检查上一个命令是否成功
if [ $? -ne 0 ]; then
    echo "分类器训练失败，退出脚本"
    exit 1
fi

# 第二阶段：训练PointLK
echo "========== 训练PointLK =========="
${PY3} experiments/train_pointlk.py \
  -o ./results/c3vd_pointlk_${DATE_TAG} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l ./log/c3vd_pointlk_${DATE_TAG}.log \
  --dataset-type c3vd \
  --transfer-from ./results/c3vd_classifier_${DATE_TAG}_feat_best.pth \
  --num-points ${NUM_POINTS} \
  --mag 0.8 \
  --pointnet tune \
  --epochs 10 \
  --batch-size ${BATCH_SIZE} \
  --workers 2 \
  --device ${DEVICE} \
  --drop-last

echo "训练完成！"

# 保存配置信息
echo "数据集路径: ${DATASET_PATH}" > ./results/c3vd_training_${DATE_TAG}_config.txt
echo "类别文件: ${CATEGORY_FILE}" >> ./results/c3vd_training_${DATE_TAG}_config.txt
echo "点云数量: ${NUM_POINTS}" >> ./results/c3vd_training_${DATE_TAG}_config.txt
echo "批量大小: ${BATCH_SIZE}" >> ./results/c3vd_training_${DATE_TAG}_config.txt
echo "设备: ${DEVICE}" >> ./results/c3vd_training_${DATE_TAG}_config.txt
echo "训练日期: $(date)" >> ./results/c3vd_training_${DATE_TAG}_config.txt

# 训练完成后禁用
# sudo swapoff /swapfile
# sudo rm /swapfile
