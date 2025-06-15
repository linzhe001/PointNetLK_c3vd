#! /usr/bin/bash
#$ -l tmem=64G
#$ -l h_vmem=64G            
#$ -l h_rt=720000  
#$ -l gpu=true
#$ -pe gpu 1
#$ -N ljiang_finetune_c3vd
#$ -o /SAN/medic/MRpcr/logs/finetune_c3vd_output.log
#$ -e /SAN/medic/MRpcr/logs/finetune_c3vd_error.log
#$ -wd /SAN/medic/MRpcr

  
# 设置工作目录为项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
echo "当前工作目录: $(pwd)"

# 激活Conda环境
echo "激活Conda环境..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 创建结果和日志目录
mkdir -p /SAN/medic/MRpcr/results/modelnet
mkdir -p /SAN/medic/MRpcr/results/c3vd

# Python命令
PY3="nice -n 10 python"

# 设置变量
# ModelNet数据集路径 - 更新为正确路径
MODELNET_PATH="/SAN/medic/MRpcr/ModelNet40"
MODELNET_CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/modelnet22.txt"

# C3VD数据集路径
C3VD_PATH="/SAN/medic/MRpcr/C3VD_datasets"
C3VD_CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/c3vd.txt"

NUM_POINTS=1024
DEVICE="cuda:0"
BATCH_SIZE=16
DATE_TAG=$(date +"%m%d")

# 保存路径
MODELNET_OUTPUT="/SAN/medic/MRpcr/results/modelnet/modelnet_finetune_classifier_${DATE_TAG}"
C3VD_CLASSIFIER_OUTPUT="/SAN/medic/MRpcr/results/c3vd/c3vd_finetune_classifier_${DATE_TAG}"
C3VD_POINTLK_OUTPUT="/SAN/medic/MRpcr/results/c3vd/c3vd_finetune_pointlk_${DATE_TAG}"

# Classifier使用随机分割，PointLK使用场景分割
CLASSIFIER_SPLIT=""  # Classifier使用随机分割
POINTLK_SPLIT="--scene-split"  # PointLK使用场景分割

# 配对模式设置 (one_to_one 或 scene_reference)
PAIR_MODE="one_to_one"  # 使用点对点配对模式
REFERENCE_NAME=""  # 清空参考点云名称
REFERENCE_PARAM=""
if [ "${PAIR_MODE}" = "scene_reference" ] && [ -n "${REFERENCE_NAME}" ]; then
  REFERENCE_PARAM="--reference-name ${REFERENCE_NAME}"
fi

# 打印配对模式信息
echo "========== 配对模式配置 =========="
echo "使用配对模式: ${PAIR_MODE}"
if [ "${PAIR_MODE}" = "scene_reference" ]; then
  echo "目标数据集目录: C3VD_ref"
  if [ -n "${REFERENCE_NAME}" ]; then
    echo "参考点云: ${REFERENCE_NAME}"
  else
    echo "参考点云: 自动选择每个场景的第一个点云"
  fi
else
  echo "目标数据集目录: visible_point_cloud_ply_depth"
  echo "配对方式: 每个源点云匹配对应帧号的目标点云"
fi

# 第1阶段: 在ModelNet上训练分类器
echo "========== 在ModelNet上训练分类器 =========="
${PY3} train_classifier.py \
  -o ${MODELNET_OUTPUT} \
  -i ${MODELNET_PATH} \
  -c ${MODELNET_CATEGORY_FILE} \
  -l ${MODELNET_OUTPUT}.log \
  --dataset-type modelnet \
  --num-points ${NUM_POINTS} \
  --device ${DEVICE} \
  --epochs 150 \
  --batch-size ${BATCH_SIZE} \
  --workers 4 \
  --drop-last \
  --verbose

# 检查前一个命令是否成功
if [ $? -ne 0 ]; then
    echo "ModelNet分类器训练失败，退出脚本"
    exit 1
fi

echo "ModelNet分类器训练完成，最佳模型保存为: ${MODELNET_OUTPUT}_model_best.pth"

# 第2阶段: 在C3VD上微调分类器
echo "========== 在C3VD上微调分类器 =========="
${PY3} train_classifier.py \
  -o ${C3VD_CLASSIFIER_OUTPUT} \
  -i ${C3VD_PATH} \
  -c ${C3VD_CATEGORY_FILE} \
  -l ${C3VD_CLASSIFIER_OUTPUT}.log \
  --dataset-type c3vd \
  --num-points ${NUM_POINTS} \
  --device ${DEVICE} \
  --epochs 100 \
  --batch-size ${BATCH_SIZE} \
  --workers 4 \
  --drop-last \
  ${CLASSIFIER_SPLIT} \
  --verbose \
  --pair-mode ${PAIR_MODE} \
  --pretrained ${MODELNET_OUTPUT}_model_best.pth \
  ${REFERENCE_PARAM}

# 检查前一个命令是否成功
if [ $? -ne 0 ]; then
    echo "C3VD分类器微调失败，退出脚本"
    exit 1
fi

echo "C3VD分类器微调完成，最佳模型保存为: ${C3VD_CLASSIFIER_OUTPUT}_model_best.pth"
echo "C3VD分类器特征提取器保存为: ${C3VD_CLASSIFIER_OUTPUT}_feat_best.pth"

# 第3阶段: 使用微调后的分类器特征训练PointLK
echo "========== 训练PointLK =========="
${PY3} train_pointlk.py \
  -o ${C3VD_POINTLK_OUTPUT} \
  -i ${C3VD_PATH} \
  -c ${C3VD_CATEGORY_FILE} \
  -l ${C3VD_POINTLK_OUTPUT}.log \
  --dataset-type c3vd \
  --transfer-from ${C3VD_CLASSIFIER_OUTPUT}_feat_best.pth \
  --num-points ${NUM_POINTS} \
  --mag 0.5 \
  --pointnet tune \
  --epochs 400 \
  --batch-size ${BATCH_SIZE} \
  --workers 4 \
  --device ${DEVICE} \
  --drop-last \
  --optimizer Adam \
  ${POINTLK_SPLIT} \
  --verbose \
  --pair-mode ${PAIR_MODE} \
  ${REFERENCE_PARAM}

echo "训练完成!"

# 保存配置信息
CONFIG_FILE="/SAN/medic/MRpcr/results/c3vd/finetune_c3vd_${DATE_TAG}_config.txt"
echo "========== 训练配置信息 ==========" > ${CONFIG_FILE}
echo "ModelNet数据集路径: ${MODELNET_PATH}" >> ${CONFIG_FILE}
echo "ModelNet类别文件: ${MODELNET_CATEGORY_FILE}" >> ${CONFIG_FILE}
echo "C3VD数据集路径: ${C3VD_PATH}" >> ${CONFIG_FILE}
echo "C3VD类别文件: ${C3VD_CATEGORY_FILE}" >> ${CONFIG_FILE}
echo "点云数量: ${NUM_POINTS}" >> ${CONFIG_FILE}
echo "批次大小: ${BATCH_SIZE}" >> ${CONFIG_FILE}
echo "设备: ${DEVICE}" >> ${CONFIG_FILE}
echo "配对模式: ${PAIR_MODE}" >> ${CONFIG_FILE}
if [ "${PAIR_MODE}" = "scene_reference" ] && [ -n "${REFERENCE_NAME}" ]; then
  echo "参考点云名称: ${REFERENCE_NAME}" >> ${CONFIG_FILE}
fi
echo "训练日期: $(date)" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "ModelNet分类器输出: ${MODELNET_OUTPUT}" >> ${CONFIG_FILE}
echo "C3VD分类器输出: ${C3VD_CLASSIFIER_OUTPUT}" >> ${CONFIG_FILE}
echo "PointLK模型输出: ${C3VD_POINTLK_OUTPUT}" >> ${CONFIG_FILE}

echo "配置信息已保存到: ${CONFIG_FILE}"
