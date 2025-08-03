#!/usr/bin/bash
#$ -l tmem=64G
#$ -l h_vmem=64G            
#$ -l h_rt=720000  
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -pe gpu 1
#$ -N ljiang_train_new_mamba_pointlk_c3vd
#$ -o /SAN/medic/MRpcr/logs/new_mamba_pointlk_c3vd_output.log
#$ -e /SAN/medic/MRpcr/logs/new_mamba_pointlk_c3vd_error.log
#$ -wd /SAN/medic/MRpcr

#=============================================================================
# 新 Mamba3D C3VD PointLK 配准模型训练脚本
#=============================================================================
#
# 此脚本使用预训练的分类器权重，训练一个带有领域适应和几何对应损失的
# PointLK配准模型。
#
# 使用前请确保已完成分类器训练，并正确设置下面的 `CLASSIFIER_PREFIX` 变量。
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

# --- !!重要配置!! ---
# 恢复训练配置：设置要恢复的模型路径
RESUME_CHECKPOINT="/SAN/medic/MRpcr/results/new_mamba_c3vd/new_mamba_pointlk_0711_model_best.pth"
# 如果要从头开始训练，请设置为空字符串: RESUME_CHECKPOINT=""

# --- 分类器配置（仅用于从头训练） ---
# 只有当 RESUME_CHECKPOINT 为空时才需要设置此项
CLASSIFIER_DATE_TAG="0709"

# --- 基本配置 ---
DATASET_PATH="/SAN/medic/MRpcr/C3VD_datasets"
CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/c3vd.txt"
NUM_POINTS=1024
DEVICE="cuda:0"
BATCH_SIZE=16
DATE_TAG=$(date +"%m%d")
MAG=0.8 # 用于PointLK的变换幅度

# --- Mamba3D 模型配置 ---
DIM_K=1024
NUM_MAMBA_BLOCKS=1
D_STATE=8
EXPAND=2
SYMFN="max"

# --- 损失函数权重配置 ---
ADVERSARIAL_LAMBDA=0.1
CORRESPONDENCE_LAMBDA=0.05

# --- 体素化配置 ---
USE_VOXELIZATION=false
VOXEL_SIZE=4
VOXEL_GRID_SIZE=32
MAX_VOXEL_POINTS=100
MAX_VOXELS=20000
MIN_VOXEL_POINTS_RATIO=0.1

# --- 数据集处理配置 ---
PAIR_MODE="one_to_one"
REFERENCE_NAME=""
REFERENCE_PARAM=""
if [ "${PAIR_MODE}" = "scene_reference" ] && [ -n "${REFERENCE_NAME}" ]; then
  REFERENCE_PARAM="--reference-name ${REFERENCE_NAME}"
fi

VOXELIZATION_PARAMS=""
if [ "${USE_VOXELIZATION}" = "true" ]; then
    VOXELIZATION_PARAMS="--use-voxelization --voxel-size ${VOXEL_SIZE} --voxel-grid-size ${VOXEL_GRID_SIZE} --max-voxel-points ${MAX_VOXEL_POINTS} --max-voxels ${MAX_VOXELS} --min-voxel-points-ratio ${MIN_VOXEL_POINTS_RATIO}"
else
    VOXELIZATION_PARAMS="--no-voxelization"
fi

# 场景划分参数 (PointLK场景划分)
POINTLK_SCENE_SPLIT="--scene-split"

# --- 打印配置信息 ---
echo "========== 新Mamba3D C3VD PointLK训练配置 =========="
echo "模型: Mamba3D PointLK"
echo "数据集路径: ${DATASET_PATH}"
echo "批次大小: ${BATCH_SIZE}"
echo "体素化启用: ${USE_VOXELIZATION}"
echo "对抗损失 Lambda: ${ADVERSARIAL_LAMBDA}"
echo "对应损失 Lambda: ${CORRESPONDENCE_LAMBDA}"
echo "Delta: 1e-4"
echo "Max Iter: 10"
if [ -n "${RESUME_CHECKPOINT}" ] && [ -f "${RESUME_CHECKPOINT}" ]; then
    echo "训练模式: 🔄 恢复训练"
    echo "恢复检查点: ${RESUME_CHECKPOINT}"
    echo "注意: 恢复训练不需要分类器权重"
else
    echo "训练模式: 🆕 从头训练"
    echo "分类器日期标签: ${CLASSIFIER_DATE_TAG}"
    echo "特征权重文件: ${TRANSFER_FILE}"
fi
echo "==============================================="

# 检查数据集路径
if [ ! -d "${DATASET_PATH}" ]; then
    echo "错误: 数据集目录 ${DATASET_PATH} 不存在!"
    exit 1
fi

# 检查恢复训练配置
if [ -n "${RESUME_CHECKPOINT}" ] && [ -f "${RESUME_CHECKPOINT}" ]; then
    echo "========== 恢复训练模式 =========="
    echo "将从检查点恢复训练: ${RESUME_CHECKPOINT}"
    RESUME_PARAM="--resume ${RESUME_CHECKPOINT}"
    TRANSFER_PARAM=""
else
    echo "========== 从头训练模式 =========="
    if [ -n "${RESUME_CHECKPOINT}" ]; then
        echo "警告: 指定的恢复检查点文件不存在: ${RESUME_CHECKPOINT}"
        echo "将从预训练特征权重开始训练"
    fi
    
    # 检查分类器日期标签
    if [ -z "${CLASSIFIER_DATE_TAG}" ]; then
        echo "错误: 从头训练需要设置 CLASSIFIER_DATE_TAG 变量"
        echo "请编辑此脚本，设置正确的分类器训练日期标签 (格式 MMDD, 例如 0709)"
        exit 1
    fi
    
    # 检查预训练特征文件
    CLASSIFIER_PREFIX="/SAN/medic/MRpcr/results/new_mamba_c3vd/new_mamba_classifier_${CLASSIFIER_DATE_TAG}"
    TRANSFER_FILE="${CLASSIFIER_PREFIX}_feat_best.pth"
    
    if [ ! -f "${TRANSFER_FILE}" ]; then
        echo "错误: 找不到预训练的特征文件 ${TRANSFER_FILE}"
        exit 1
    fi
    
    RESUME_PARAM=""
    TRANSFER_PARAM="--transfer-from ${TRANSFER_FILE}"
fi

# =============================================
# 训练PointLK配准模型 (带完整损失)
# =============================================
echo "========== 训练Mamba3D PointLK =========="

# 根据训练模式设置输出文件名
if [ -n "${RESUME_CHECKPOINT}" ] && [ -f "${RESUME_CHECKPOINT}" ]; then
    POINTLK_PREFIX="/SAN/medic/MRpcr/results/new_mamba_c3vd/new_mamba_pointlk_resume_${DATE_TAG}"
else
    POINTLK_PREFIX="/SAN/medic/MRpcr/results/new_mamba_c3vd/new_mamba_pointlk_${DATE_TAG}"
fi

${PY3} train_pointlk.py \
  -o ${POINTLK_PREFIX} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l ${POINTLK_PREFIX}.log \
  --dataset-type c3vd \
  ${RESUME_PARAM} \
  ${TRANSFER_PARAM} \
  --num-points ${NUM_POINTS} \
  --mag ${MAG} \
  --delta 1e-4 \
  --max-iter 10 \
  --pointnet tune \
  --epochs 200 \
  --batch-size ${BATCH_SIZE} \
  --workers 4 \
  --device ${DEVICE} \
  --drop-last \
  --verbose \
  --pair-mode ${PAIR_MODE} \
  ${REFERENCE_PARAM} \
  ${POINTLK_SCENE_SPLIT} \
  --model-type mamba3d \
  --dim-k ${DIM_K} \
  --num-mamba-blocks ${NUM_MAMBA_BLOCKS} \
  --d-state ${D_STATE} \
  --expand ${EXPAND} \
  --symfn ${SYMFN} \
  ${VOXELIZATION_PARAMS} \
  --adversarial-lambda ${ADVERSARIAL_LAMBDA} \
  --correspondence-lambda ${CORRESPONDENCE_LAMBDA} \
  --optimizer Adam \
  --base-lr 0.0001 \
  --warmup-epochs 5 \
  --cosine-annealing

echo "训练完成!" 