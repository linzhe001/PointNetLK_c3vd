#!/usr/bin/bash
#$ -l tmem=64G
#$ -l h_vmem=64G            
#$ -l h_rt=720000  
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -pe gpu 1
#$ -N ljiang_train_new_mamba_c3vd
#$ -o /SAN/medic/MRpcr/logs/new_mamba_c3vd_output.log
#$ -e /SAN/medic/MRpcr/logs/new_mamba_c3vd_error.log
#$ -wd /SAN/medic/MRpcr

#=============================================================================
# 新 Mamba3D C3VD 分类器训练脚本 (带领域适应)
#=============================================================================
#
# 此脚本训练一个带有领域判别器的Mamba3D分类器。
#
#=============================================================================

# Set working directory to project root
cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
echo "Current working directory: $(pwd)"

# Activate Conda environment
echo "Activating Conda environment..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Create result and log directories
mkdir -p /SAN/medic/MRpcr/results/new_mamba_c3vd

# Python command
PY3="nice -n 10 python"

# --- 基本配置 ---
DATASET_PATH="/SAN/medic/MRpcr/C3VD_datasets"
CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/c3vd.txt"
NUM_POINTS=1024
DEVICE="cuda:0"
BATCH_SIZE=16
DATE_TAG=$(date +"%m%d")

# --- Mamba3D 模型配置 ---
DIM_K=1024
NUM_MAMBA_BLOCKS=1
D_STATE=8
EXPAND=2
SYMFN="max"

# --- 损失函数权重配置 ---
ADVERSARIAL_LAMBDA=0.1

# --- 体素化配置 ---
# (此部分仅用于PointLK，分类器中不使用)

# --- 数据集处理配置 ---
# (此部分仅用于PointLK，分类器中不使用)

# 场景划分参数 (分类器随机划分)
CLASSIFIER_SCENE_SPLIT=""

# --- 打印配置信息 ---
echo "========== 新Mamba3D C3VD分类器训练配置 =========="
echo "模型: Mamba3D (带领域适应)"
echo "数据集路径: ${DATASET_PATH}"
echo "批次大小: ${BATCH_SIZE}"
echo "对抗损失 Lambda: ${ADVERSARIAL_LAMBDA}"
echo "=========================================="

# 检查数据集路径
if [ ! -d "${DATASET_PATH}" ]; then
    echo "错误: 数据集目录 ${DATASET_PATH} 不存在!"
    exit 1
fi

# =============================================
# Stage 1: 训练分类器 (带领域对抗)
# =============================================
echo "========== Stage 1: 训练Mamba3D分类器 =========="
CLASSIFIER_PREFIX="/SAN/medic/MRpcr/results/new_mamba_c3vd/new_mamba_classifier_${DATE_TAG}"
${PY3} train_classifier.py \
  -o ${CLASSIFIER_PREFIX} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l ${CLASSIFIER_PREFIX}.log \
  --dataset-type c3vd \
  --num-points ${NUM_POINTS} \
  --device ${DEVICE} \
  --epochs 300 \
  --batch-size ${BATCH_SIZE} \
  --workers 4 \
  --drop-last \
  --verbose \
  ${CLASSIFIER_SCENE_SPLIT} \
  --model-type mamba3d \
  --dim-k ${DIM_K} \
  --num-mamba-blocks ${NUM_MAMBA_BLOCKS} \
  --d-state ${D_STATE} \
  --expand ${EXPAND} \
  --symfn ${SYMFN} \
  --adversarial-lambda ${ADVERSARIAL_LAMBDA} \
  --base-lr 0.0001 \
  --warmup-epochs 10 \
  --cosine-annealing

if [ $? -ne 0 ]; then
    echo "分类器训练失败，退出脚本"
    exit 1
fi

echo "分类器训练完成!" 