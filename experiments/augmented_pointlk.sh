#! /usr/bin/bash
#$ -l tmem=64G
#$ -l h_vmem=64G            
#$ -l h_rt=720000  
#$ -l gpu=true
#$ -pe gpu 1
#$ -N ljiang_train_augmented_c3vd
#$ -o /SAN/medic/MRpcr/logs/c_train_augmented_c3vd_output.log
#$ -e /SAN/medic/MRpcr/logs/c_train_augmented_c3vd_error.log
#$ -wd /SAN/medic/MRpcr

  
# 设置工作目录为项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WORKSPACE="/SAN/medic/MRpcr"
cd ${SCRIPT_DIR}
echo "当前工作目录: $(pwd)"

source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

# 设置变量
DATASET_PATH="/SAN/medic/MRpcr/C3VD_datasets"
CATEGORY_FILE="${WORKSPACE}/PointNetLK_c3vd/experiments/sampledata/c3vd.txt"
NUM_POINTS=1024
DEVICE="cuda:0"
BATCH_SIZE=16
DATE_TAG=$(date +"%m%d")
EMBEDDING="pointnet"
DATASET_TYPE="c3vd"
JAC_METHOD="approx"  # ['approx', 'feature']

# 分类器使用随机分割，PointLK使用场景分割
CLASSIFIER_SPLIT=""  # 分类器使用随机分割
POINTLK_SPLIT="--scene-split"  # PointLK使用场景分割

# 初始配对模式设置 (为分类器训练设置，PointLK将使用all模式)
PAIR_MODE="one_to_one"  # 使用点对点配对模式
REFERENCE_NAME=""  # 清空参考点云名称
REFERENCE_PARAM=""
if [ "${PAIR_MODE}" = "scene_reference" ] && [ -n "${REFERENCE_NAME}" ]; then
  REFERENCE_PARAM="--reference-name ${REFERENCE_NAME}"
fi

# 定义保存路径格式
SAVE_PREFIX="${EMBEDDING}_${DATASET_TYPE}_${JAC_METHOD}_${DATE_TAG}"
RESULTS_DIR="${WORKSPACE}/PointNetLK_c3vd/results"
mkdir -p ${RESULTS_DIR}/${SAVE_PREFIX}

# 设置Python脚本路径
SCRIPTS_PATH="${WORKSPACE}/PointNetLK_c3vd/experiments"
PY3="python3" 

# Stage 1: 训练分类器 (使用随机分割)
echo "========== 训练分类器 =========="
${PY3} ${SCRIPTS_PATH}/train_classifier.py \
  -o ${RESULTS_DIR}/${SAVE_PREFIX}/classifier \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l ${RESULTS_DIR}/${SAVE_PREFIX}/classifier.log \
  --dataset-type ${DATASET_TYPE} \
  --num-points ${NUM_POINTS} \
  --device ${DEVICE} \
  --epochs 200 \
  --batch-size ${BATCH_SIZE} \
  --workers 4 \
  --drop-last \
  --verbose


# ======================================================
# Stage 2: 使用所有配对方式训练PointLK
# ======================================================
echo "========== 训练PointLK (使用所有配对方式增强数据) =========="
AUGMENTED_PAIR_MODE="all"
echo "配对模式: ${AUGMENTED_PAIR_MODE}"
echo "雅可比矩阵计算方法: ${JAC_METHOD}"

${PY3} ${SCRIPTS_PATH}/train_pointlk.py \
  -o ${RESULTS_DIR}/${SAVE_PREFIX}/pointlk \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l ${RESULTS_DIR}/${SAVE_PREFIX}/pointlk.log \
  --dataset-type ${DATASET_TYPE} \
  --transfer-from ${RESULTS_DIR}/${SAVE_PREFIX}/classifier_feat_best.pth \
  --num-points ${NUM_POINTS} \
  --mag 0.8 \
  --pointnet "tune" \
  --epochs 150 \
  --batch-size ${BATCH_SIZE} \
  --workers 4 \
  --device ${DEVICE} \
  --drop-last \
  ${POINTLK_SPLIT} \
  --verbose \
  --pair-mode ${AUGMENTED_PAIR_MODE} \
  --jac-method ${JAC_METHOD} \
  ${REFERENCE_PARAM}

echo "增强训练完成！"

# 保存配置信息
echo "数据集路径: ${DATASET_PATH}" > ${RESULTS_DIR}/${SAVE_PREFIX}/${SAVE_PREFIX}_training_all_config.txt
echo "类别文件: ${CATEGORY_FILE}" >> ${RESULTS_DIR}/${SAVE_PREFIX}/${SAVE_PREFIX}_training_all_config.txt
echo "点云数量: ${NUM_POINTS}" >> ${RESULTS_DIR}/${SAVE_PREFIX}/${SAVE_PREFIX}_training_all_config.txt
echo "批次大小: ${BATCH_SIZE}" >> ${RESULTS_DIR}/${SAVE_PREFIX}/${SAVE_PREFIX}_training_all_config.txt
echo "设备: ${DEVICE}" >> ${RESULTS_DIR}/${SAVE_PREFIX}/${SAVE_PREFIX}_training_all_config.txt
echo "配对模式: ${AUGMENTED_PAIR_MODE}" >> ${RESULTS_DIR}/${SAVE_PREFIX}/${SAVE_PREFIX}_training_all_config.txt
echo "雅可比矩阵计算方法: ${JAC_METHOD}" >> ${RESULTS_DIR}/${SAVE_PREFIX}/${SAVE_PREFIX}_training_all_config.txt
echo "训练日期: $(date)" >> ${RESULTS_DIR}/${SAVE_PREFIX}/${SAVE_PREFIX}_training_all_config.txt
echo "数据增强说明: 包含一对一配对、源点云到源点云配对和目标点云到目标点云配对" >> ${RESULTS_DIR}/${SAVE_PREFIX}/${SAVE_PREFIX}_training_all_config.txt
