#! /usr/bin/bash
#$ -l tmem=64G
#$ -l h_vmem=64G            
#$ -l h_rt=720000  
#$ -l gpu=true

#$ -pe gpu 1
#$ -N ljiang_train_models_c3vd
#$ -o /SAN/medic/MRpcr/logs/c_train_models_c3vd_output.log
#$ -e /SAN/medic/MRpcr/logs/c_train_models_c3vd_error.log
#$ -wd /SAN/medic/MRpcr

  
# Set working directory to project root
cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
echo "Current working directory: $(pwd)"

# Activate Conda environment
echo "Activating Conda environment..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Create result and log directories
mkdir -p /SAN/medic/MRpcr/results/c3vd

# Python command
PY3="nice -n 10 python"

# Set variables
DATASET_PATH="/SAN/medic/MRpcr/C3VD_datasets"
CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK/experiments/sampledata/c3vd.txt"
NUM_POINTS=1024
DEVICE="cuda:0"
BATCH_SIZE=16
DATE_TAG=$(date +"%m%d")
MAG=0.8 # 用于PointLK的变换幅度

# C3VD配对模式设置
PAIR_MODE="one_to_one"  # 使用点对点配对模式
REFERENCE_NAME=""  # 清空参考点云名称
REFERENCE_PARAM=""
if [ "${PAIR_MODE}" = "scene_reference" ] && [ -n "${REFERENCE_NAME}" ]; then
  REFERENCE_PARAM="--reference-name ${REFERENCE_NAME}"
fi

# 场景划分参数（使用--scene-split）
SCENE_SPLIT="--scene-split"
# 分类器不使用场景划分，使用随机划分
CLASSIFIER_SCENE_SPLIT=""
# PointLK使用场景划分
POINTLK_SCENE_SPLIT="${SCENE_SPLIT}"

# 打印配置信息
echo "========== 训练配置 =========="
echo "数据集路径: ${DATASET_PATH}"
echo "类别文件: ${CATEGORY_FILE}"
echo "配对模式: ${PAIR_MODE}"
echo "参考点云: ${REFERENCE_NAME:-'自动选择'}"
echo "分类器数据划分: 随机划分"
echo "PointLK数据划分: 场景划分"
echo "点云数量: ${NUM_POINTS}"
echo "批次大小: ${BATCH_SIZE}"
echo "设备: ${DEVICE}"

# 检查数据集路径
echo "========== 数据集路径检查 =========="
if [ -d "${DATASET_PATH}" ]; then
    echo "数据集目录存在"
    echo "源点云目录: ${DATASET_PATH}/C3VD_ply_source"
    echo "目标点云目录: ${DATASET_PATH}/visible_point_cloud_ply_depth"
    
    if [ -d "${DATASET_PATH}/C3VD_ply_source" ]; then
        echo "源点云目录存在"
    else
        echo "警告: 源点云目录不存在!"
    fi
    
    if [ -d "${DATASET_PATH}/visible_point_cloud_ply_depth" ]; then
        echo "目标点云目录存在"
    else
        echo "警告: 目标点云目录不存在!"
    fi
else
    echo "警告: 数据集目录不存在!"
fi


# Stage 1: Train classifier
echo "========== 训练分类器 =========="
${PY3} train_classifier.py \
  -o /SAN/medic/MRpcr/results/c3vd/c3vd_classifier_${DATE_TAG} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l /SAN/medic/MRpcr/results/c3vd/c3vd_classifier_${DATE_TAG}.log \
  --dataset-type c3vd \
  --num-points ${NUM_POINTS} \
  --device ${DEVICE} \
  --epochs 300 \
  --batch-size ${BATCH_SIZE} \
  --workers 4 \
  --drop-last \
  --verbose \
  --pair-mode ${PAIR_MODE} \
  ${REFERENCE_PARAM} \
  ${CLASSIFIER_SCENE_SPLIT} \
  --base-lr 0.00001 \
  --warmup-epochs 5 \
  --cosine-annealing

# Check if previous command succeeded
if [ $? -ne 0 ]; then
    echo "分类器训练失败，退出脚本"
    exit 1
fi

# Stage 2: Train PointLK
echo "========== 训练 PointLK =========="
${PY3} train_pointlk.py \
  -o /SAN/medic/MRpcr/results/c3vd/c3vd_pointlk_${DATE_TAG} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l /SAN/medic/MRpcr/results/c3vd/c3vd_pointlk_${DATE_TAG}.log \
  --dataset-type c3vd \
  --transfer-from /SAN/medic/MRpcr/results/c3vd/c3vd_classifier_${DATE_TAG}_feat_best.pth \
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
  ${POINTLK_SCENE_SPLIT}

echo "训练完成!"

# 保存配置信息
CONFIG_FILE="/SAN/medic/MRpcr/results/c3vd/c3vd_training_${DATE_TAG}_config.txt"
echo "训练配置信息" > ${CONFIG_FILE}
echo "=============================" >> ${CONFIG_FILE}
echo "日期: $(date)" >> ${CONFIG_FILE}
echo "数据集路径: ${DATASET_PATH}" >> ${CONFIG_FILE}
echo "类别文件: ${CATEGORY_FILE}" >> ${CONFIG_FILE}
echo "配对模式: ${PAIR_MODE}" >> ${CONFIG_FILE}
if [ "${PAIR_MODE}" = "scene_reference" ] && [ -n "${REFERENCE_NAME}" ]; then
  echo "参考点云名称: ${REFERENCE_NAME}" >> ${CONFIG_FILE}
fi
echo "分类器数据划分: 随机划分" >> ${CONFIG_FILE}
echo "PointLK数据划分: 场景划分" >> ${CONFIG_FILE}
echo "点云数量: ${NUM_POINTS}" >> ${CONFIG_FILE}
echo "批次大小: ${BATCH_SIZE}" >> ${CONFIG_FILE}
echo "设备: ${DEVICE}" >> ${CONFIG_FILE}
echo "变换幅度: ${MAG}" >> ${CONFIG_FILE}
echo "总训练时间: $(date)" >> ${CONFIG_FILE}

# Disabled after training
# sudo swapoff /swapfile
# sudo rm /swapfile
