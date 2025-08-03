#!/usr/bin/bash
#$ -l tmem=64G
#$ -l h_vmem=64G            
#$ -l h_rt=720000  
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -pe gpu 1
#$ -N ljiang_train_mamba_v2_pointlk_loss2
#$ -o /SAN/medic/MRpcr/logs/mamba_v2_pointlk_c3vd_loss2_output.log
#$ -e /SAN/medic/MRpcr/logs/mamba_v2_pointlk_c3vd_loss2_error.log
#$ -wd /SAN/medic/MRpcr

#=============================================================================
# Mamba3D-v2 C3VD PointLK 配准模型训练脚本 (双损失+全局特征约束)
#=============================================================================
#
# 此脚本使用预训练的 Mamba3D-v2 分类器权重，训练一个不使用领域适应的
# PointLK 配准模型，使用双损失结构：
# - loss_r: 特征残差损失
# - loss_g: 几何变换损失  
# - 全局特征一致性损失 (权重0.1)
#
# 脚本会自动寻找最新的分类器权重文件，无需手动指定日期标签。
# 请确保已完成分类器训练。
#=============================================================================

# Set working directory to project root
cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
echo "Current working directory: $(pwd)"

# Activate Conda environment
echo "Activating Conda environment..."
source /SAN/medic/CARES/mobarak/venvs/anaconda3/etc/profile.d/conda.sh
conda activate mamba
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Create result and log directories
mkdir -p /SAN/medic/MRpcr/results/mamba_v2_c3vd

# Python command
PY3="nice -n 10 python"

# --- !!重要配置!! ---
# 自动寻找最新的分类器权重文件，不再需要手动指定日期
# CLASSIFIER_DATE_TAG="0723" # <--- 不再需要手动设置日期

# 函数：自动寻找最新的分类器权重文件
find_latest_classifier() {
    local result_dir="$1"
    local model_prefix="$2"
    
    # 寻找所有匹配的 feat_best.pth 文件
    local feat_files=$(find "$result_dir" -name "${model_prefix}_*_feat_best.pth" -type f 2>/dev/null | sort -V)
    
    if [ -z "$feat_files" ]; then
        echo ""
        return 1
    fi
    
    # 返回最新的文件（按文件名排序的最后一个）
    echo "$feat_files" | tail -1
}

# 自动寻找最新的分类器权重
echo "正在寻找最新的Mamba3D-v2分类器权重..."
CLASSIFIER_RESULT_DIR="/SAN/medic/MRpcr/results/mamba_v2_c3vd"
LATEST_FEAT_FILE=$(find_latest_classifier "$CLASSIFIER_RESULT_DIR" "mamba_v2_classifier")

if [ -z "$LATEST_FEAT_FILE" ]; then
    echo "错误: 在 $CLASSIFIER_RESULT_DIR 中未找到任何 mamba_v2_classifier_*_feat_best.pth 文件"
    echo "请先运行分类器训练脚本: train_mamba_v2_c3vd.sh"
    exit 1
fi

echo "找到最新的分类器权重文件: $LATEST_FEAT_FILE"

# 提取分类器前缀（用于生成PointLK输出文件名）
CLASSIFIER_BASENAME=$(basename "$LATEST_FEAT_FILE" "_feat_best.pth")
echo "分类器权重基础名称: $CLASSIFIER_BASENAME"

# --- 基本配置 ---
DATASET_PATH="/SAN/medic/MRpcr/C3VD_datasets"
CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/c3vd.txt"
NUM_POINTS=1024
DEVICE="cuda:0"
BATCH_SIZE=16
DATE_TAG=$(date +"%m%d")
MAG=0.8 # 用于PointLK的变换幅度

# --- Mamba3D-v2 模型配置 ---
DIM_K=1024
NUM_MAMBA_BLOCKS=3
D_STATE=16
EXPAND=2
SYMFN="max"

# --- 删除对抗损失配置 ---
# 不再使用领域对抗训练和特征对应损失，仅使用标准几何损失

# --- 体素化配置 ---
USE_VOXELIZATION=true
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
echo "========== Mamba3D-v2 C3VD PointLK训练配置 =========="
echo "模型: Mamba3D-v2 PointLK (双损失+全局特征约束)"
echo "数据集路径: ${DATASET_PATH}"
echo "批次大小: ${BATCH_SIZE}"
echo "体素化启用: ${USE_VOXELIZATION}"
echo "损失结构: loss_r + loss_g + 0.0*全局特征一致性损失 (loss2)"
echo "==============================================="

# 检查数据集路径
if [ ! -d "${DATASET_PATH}" ]; then
    echo "错误: 数据集目录 ${DATASET_PATH} 不存在!"
    exit 1
fi

# 删除原有的CLASSIFIER_DATE_TAG检查，因为现在自动寻找权重文件
# 检查是否找到了有效的权重文件
if [ ! -f "${LATEST_FEAT_FILE}" ]; then
    echo "错误: 权重文件不存在: ${LATEST_FEAT_FILE}"
    exit 1
fi

# =============================================
# 训练PointLK配准模型 (双损失+全局特征约束)
# =============================================
echo "========== 训练Mamba3D-v2 PointLK =========="
POINTLK_PREFIX="/SAN/medic/MRpcr/results/mamba_v2_c3vd/mamba_v2_pointlk_${DATE_TAG}_loss2"
TRANSFER_FILE="${LATEST_FEAT_FILE}"

echo "使用预训练权重文件: ${TRANSFER_FILE}"

${PY3} train_pointlk.py \
  -o ${POINTLK_PREFIX} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l ${POINTLK_PREFIX}.log \
  --dataset-type c3vd \
  --transfer-from ${TRANSFER_FILE} \
  --num-points ${NUM_POINTS} \
  --mag ${MAG} \
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
  --model-type mamba3d_v2 \
  --dim-k ${DIM_K} \
  --num-mamba-blocks ${NUM_MAMBA_BLOCKS} \
  --d-state ${D_STATE} \
  --expand ${EXPAND} \
  --symfn ${SYMFN} \
  ${VOXELIZATION_PARAMS} \
  --optimizer Adam \
  --base-lr 0.000001 \
  --warmup-epochs 5 \
  --cosine-annealing \
  --global-consistency-weight 0.0

echo "训练完成!" 