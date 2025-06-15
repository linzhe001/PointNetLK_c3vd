#!/usr/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G            
#$ -l h_rt=174000  # 增加到4小时以支持两阶段训练
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -pe gpu 1
#$ -N ljiang_train_mamba3d_c3vd_full
#$ -o /SAN/medic/MRpcr/logs/f_mamba3d_c3vd_output.log
#$ -e /SAN/medic/MRpcr/logs/f_mamba3d_c3vd_error.log
#$ -wd /SAN/medic/MRpcr

#=============================================================================
# Mamba3D C3VD 训练脚本
#=============================================================================
# 
# 此脚本支持两种训练模式：
# 
# 1. 从头开始训练：
#    - 设置 RESUME_CHECKPOINT="" 或注释掉该行
#    - 需要预训练的特征权重文件 (FEAT_WEIGHTS)
# 
# 2. 恢复训练（从检查点继续）：
#    - 设置 RESUME_CHECKPOINT="/path/to/checkpoint.pth"
#    - 不需要特征权重文件，所有状态都从检查点恢复
# 
# 检查点文件包含：
#    - 模型权重
#    - 优化器状态
#    - 当前epoch
#    - 最佳损失值
#    - 最佳epoch信息
# 
# 使用示例：
#    bash train_mamba3d_c3vd.sh  # 自动检测模式并开始训练
#
#=============================================================================

cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
# Set working directory to project root
echo "Current working directory: $(pwd)"

# Activate Conda environment
echo "Activating Conda environment..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

# Create result and log directories
mkdir -p /SAN/medic/MRpcr/results/mamba3d_c3vd

# Python command
PY3="nice -n 10 python"

# Set variables
DATASET_PATH="/SAN/medic/MRpcr/C3VD_datasets"
CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/c3vd.txt"
NUM_POINTS=1024
DEVICE="cuda:0"
DATE_TAG="0610"
# DATE_TAG=$(date +"%m%d")
MAG=0.8 # 用于PointLK的变换幅度

# ===== 恢复训练配置 =====
# 设置要恢复的检查点文件路径（如果要从头开始训练，请留空）
RESUME_CHECKPOINT="/SAN/medic/MRpcr/results/mamba3d_c3vd/mamba3d_pointlk_0605_model_best.pth"
# RESUME_CHECKPOINT=""  # 取消注释此行来从头开始训练

# 检查恢复检查点文件
RESUME_PARAM=""
if [ -n "${RESUME_CHECKPOINT}" ]; then
    if [ -f "${RESUME_CHECKPOINT}" ]; then
        RESUME_PARAM="--resume ${RESUME_CHECKPOINT}"
        echo "✅ 将从检查点恢复训练: ${RESUME_CHECKPOINT}"
    else
        echo "❌ 警告: 指定的检查点文件不存在: ${RESUME_CHECKPOINT}"
        echo "   将从头开始训练"
        RESUME_PARAM=""
    fi
else
    echo "ℹ️  未指定检查点，将从头开始训练"
fi

# Mamba3D特有配置
DIM_K=1024                    # 特征维度
NUM_MAMBA_BLOCKS=1            # Mamba块数量
D_STATE=8                    # 状态空间维度
EXPAND=2                      # 扩展因子
SYMFN="max"                   # 聚合函数：max, avg, 或 selective
SCALE=1                       # 模型缩放因子
BATCH_SIZE=16                  # 批次大小 (注意：Mamba可能比Attention更内存高效)
LEARNING_RATE=0.00001           # 学习率
WEIGHT_DECAY=1e-6             # 权重衰减
OPTIMIZER="Adam"              # 优化器
EPOCHS_CLASSIFIER=150         # 分类器训练轮数
EPOCHS_POINTLK=200            # PointLK训练轮数
MAX_ITER=5

# C3VD配对模式设置
PAIR_MODE="one_to_one"  # 使用点对点配对模式
REFERENCE_NAME=""  # 清空参考点云名称
REFERENCE_PARAM=""
if [ "${PAIR_MODE}" = "scene_reference" ] && [ -n "${REFERENCE_NAME}" ]; then
  REFERENCE_PARAM="--reference-name ${REFERENCE_NAME}"
fi

# 场景划分参数（使用--scene-split）
SCENE_SPLIT="--scene-split"

# 定义输出路径
CLASSIFIER_PREFIX="/SAN/medic/MRpcr/results/mamba3d_c3vd/mamba3d_classifier_${DATE_TAG}"
POINTLK_PREFIX="/SAN/medic/MRpcr/results/mamba3d_c3vd/mamba3d_pointlk_${DATE_TAG}"
CLASSIFIER_LOG="${CLASSIFIER_PREFIX}.log"
POINTLK_LOG="${POINTLK_PREFIX}.log"

# 打印配置信息
echo "========== Mamba3D 两阶段训练配置 =========="
echo "🧠 模型类型: Mamba3D (两阶段训练)"
echo "📂 数据集路径: ${DATASET_PATH}"
echo "📄 类别文件: ${CATEGORY_FILE}"
echo "🔗 配对模式: ${PAIR_MODE}"
echo "🎯 参考点云: ${REFERENCE_NAME:-'自动选择'}"
echo "📊 数据划分: 场景划分"
echo "🎲 点云数量: ${NUM_POINTS}"
echo "📦 批次大小: ${BATCH_SIZE}"
echo "🖥️  设备: ${DEVICE}"
echo ""
echo "🔧 Mamba3D参数:"
echo "   - 特征维度: ${DIM_K}"
echo "   - Mamba块数量: ${NUM_MAMBA_BLOCKS}"
echo "   - 状态空间维度: ${D_STATE}"
echo "   - 扩展因子: ${EXPAND}"
echo "   - 聚合函数: ${SYMFN}"
echo "   - 模型缩放: ${SCALE}"
echo ""
echo "🎯 训练参数:"
echo "   - 学习率: ${LEARNING_RATE}"
echo "   - 优化器: ${OPTIMIZER}"
echo "   - 权重衰减: ${WEIGHT_DECAY}"
echo "   - 分类器训练轮数: ${EPOCHS_CLASSIFIER}"
echo "   - PointLK训练轮数: ${EPOCHS_POINTLK}"
echo ""
echo "🔄 恢复训练设置:"
if [ -n "${RESUME_PARAM}" ]; then
echo "   - 恢复检查点: ${RESUME_CHECKPOINT}"
else
echo "   - 训练模式: 从头开始"
fi
echo ""
echo "📁 输出路径:"
echo "   - 分类器前缀: ${CLASSIFIER_PREFIX}"
echo "   - PointLK前缀: ${POINTLK_PREFIX}"

# 检查数据集路径
echo ""
echo "========== 数据集路径检查 =========="
if [ -d "${DATASET_PATH}" ]; then
    echo "✅ 数据集目录存在"
    echo "📁 源点云目录: ${DATASET_PATH}/C3VD_ply_source"
    echo "📁 目标点云目录: ${DATASET_PATH}/visible_point_cloud_ply_depth"
    
    if [ -d "${DATASET_PATH}/C3VD_ply_source" ]; then
        echo "✅ 源点云目录存在"
        SOURCE_COUNT=$(find "${DATASET_PATH}/C3VD_ply_source" -name "*.ply" | wc -l)
        echo "📊 源点云文件数量: ${SOURCE_COUNT}"
    else
        echo "❌ 警告: 源点云目录不存在!"
    fi
    
    if [ -d "${DATASET_PATH}/visible_point_cloud_ply_depth" ]; then
        echo "✅ 目标点云目录存在"
        TARGET_COUNT=$(find "${DATASET_PATH}/visible_point_cloud_ply_depth" -name "*.ply" | wc -l)
        echo "📊 目标点云文件数量: ${TARGET_COUNT}"
    else
        echo "❌ 警告: 目标点云目录不存在!"
    fi
else
    echo "❌ 警告: 数据集目录不存在!"
fi

# 检查类别文件
echo ""
echo "========== 类别文件检查 =========="
if [ -f "${CATEGORY_FILE}" ]; then
    echo "✅ 类别文件存在"
    CATEGORY_COUNT=$(wc -l < "${CATEGORY_FILE}")
    echo "📊 类别数量: ${CATEGORY_COUNT}"
    echo "📋 类别列表:"
    cat "${CATEGORY_FILE}" | head -10
    if [ ${CATEGORY_COUNT} -gt 10 ]; then
        echo "   ... (共${CATEGORY_COUNT}个类别)"
    fi
else
    echo "❌ 警告: 类别文件不存在!"
fi

# =============================================
# 第一阶段：训练Mamba3D分类器
# =============================================
# echo "========== 第一阶段：训练Mamba3D分类器 =========="
# echo "🧠 开始训练Mamba3D分类器..."
# echo "📁 输出前缀: ${CLASSIFIER_PREFIX}"
# echo "📋 日志文件: ${CLASSIFIER_LOG}"
# echo ""

# ${PY3} train_classifier.py \
#   -o ${CLASSIFIER_PREFIX} \
#   -i ${DATASET_PATH} \
#   -c ${CATEGORY_FILE} \
#   -l ${CLASSIFIER_LOG} \
#   --dataset-type c3vd \
#   --num-points ${NUM_POINTS} \
#   --epochs ${EPOCHS_CLASSIFIER} \
#   --batch-size ${BATCH_SIZE} \
#   --workers 4 \
#   --device ${DEVICE} \
#   --drop-last \
#   --verbose \
#   --model-type mamba3d \
#   --dim-k ${DIM_K} \
#   --num-mamba-blocks ${NUM_MAMBA_BLOCKS} \
#   --d-state ${D_STATE} \
#   --expand ${EXPAND} \
#   --symfn ${SYMFN} \
#   --optimizer ${OPTIMIZER} \
#   --base-lr ${LEARNING_RATE} \
#   --warmup-epochs 5 \
#   --cosine-annealing

# 直接使用指定的特征权重文件
FEAT_WEIGHTS="/SAN/medic/MRpcr/results/mamba3d_c3vd/mamba3d_classifier_0605_feat_best.pth"

# 如果要恢复训练，检查是否需要特征权重文件
if [ -n "${RESUME_PARAM}" ]; then
    echo "🔄 恢复训练模式：将从检查点加载完整模型状态"
    echo "   注意：恢复训练时不需要单独的特征权重文件"
    # 恢复训练时不使用transfer-from参数
    TRANSFER_PARAM=""
else
    if [ -f "${FEAT_WEIGHTS}" ]; then
        echo "✅ 找到特征权重文件: ${FEAT_WEIGHTS}"
        TRANSFER_PARAM="--transfer-from ${FEAT_WEIGHTS}"
    else
        echo "❌ 警告: 未找到特征权重文件 ${FEAT_WEIGHTS}"
        echo "   请确认文件路径是否正确"
        exit 1
    fi
fi

# =============================================
# 第二阶段：训练Mamba3D-PointLK配准模型
# =============================================
echo ""
echo "========== 训练Mamba3D-PointLK配准模型 =========="
echo "🧠 开始训练Mamba3D-PointLK配准模型..."
echo "📁 输出前缀: ${POINTLK_PREFIX}"
echo "📋 日志文件: ${POINTLK_LOG}"
if [ -n "${RESUME_PARAM}" ]; then
    echo "🔄 恢复训练: ${RESUME_CHECKPOINT}"
else
    echo "🔄 使用预训练权重: ${FEAT_WEIGHTS}"
fi
echo ""

# 确保日志文件可写
rm -f ${POINTLK_LOG}
touch ${POINTLK_LOG}
chmod 664 ${POINTLK_LOG}

# 如果不是恢复训练，确保特征权重文件存在
if [ -z "${RESUME_PARAM}" ] && [ ! -f "${FEAT_WEIGHTS}" ]; then
    echo "❌ 错误: 特征权重文件不存在: ${FEAT_WEIGHTS}"
    echo "   无法进行第二阶段训练"
    exit 1
fi

${PY3} train_pointlk.py \
  -o ${POINTLK_PREFIX} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l ${POINTLK_LOG} \
  --dataset-type c3vd \
  --num-points ${NUM_POINTS} \
  --mag ${MAG} \
  --epochs ${EPOCHS_POINTLK} \
  --batch-size ${BATCH_SIZE} \
  --workers 4 \
  --device ${DEVICE} \
  --drop-last \
  --verbose \
  --pair-mode ${PAIR_MODE} \
  ${REFERENCE_PARAM} \
  ${SCENE_SPLIT} \
  --model-type mamba3d \
  --dim-k ${DIM_K} \
  --num-mamba-blocks ${NUM_MAMBA_BLOCKS} \
  --d-state ${D_STATE} \
  --expand ${EXPAND} \
  --symfn ${SYMFN} \
  --pointnet tune \
  --max-iter ${MAX_ITER} \
  --base-lr ${LEARNING_RATE} \
  ${RESUME_PARAM} \
  ${TRANSFER_PARAM}

# 检查PointLK训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 第二阶段：Mamba3D-PointLK配准训练完成!"
    echo "📁 模型保存位置: ${POINTLK_PREFIX}_*.pth"
    echo "📋 日志文件: ${POINTLK_LOG}"
    
    # 检查生成的PointLK模型文件
    echo ""
    echo "📊 生成的PointLK模型文件:"
    ls -lh ${POINTLK_PREFIX}_*.pth 2>/dev/null || echo "❌ 未找到PointLK模型文件"
    
else
    echo ""
    echo "❌ 第二阶段：Mamba3D-PointLK配准训练失败!"
    echo "请检查错误日志: ${POINTLK_LOG}"
    exit 1
fi

# =============================================
# 训练完成总结
# =============================================
echo ""
echo "🎉🎉🎉 Mamba3D两阶段训练全部完成! 🎉🎉🎉"
echo ""
echo "📊 训练总结:"
echo "✅ 第一阶段：Mamba3D分类器训练成功"
echo "   📁 分类器模型: ${CLASSIFIER_PREFIX}_*.pth"
echo "   📋 分类器日志: ${CLASSIFIER_LOG}"
echo ""
echo "✅ 第二阶段：Mamba3D-PointLK配准训练成功"
echo "   📁 配准模型: ${POINTLK_PREFIX}_*.pth"
echo "   📋 配准日志: ${POINTLK_LOG}"
echo ""
echo "🏷️  标签: ${DATE_TAG}"
echo "⏰ 完成时间: $(date)"

# 保存详细配置信息
CONFIG_FILE="/SAN/medic/MRpcr/results/mamba3d_c3vd/mamba3d_c3vd_${DATE_TAG}_config.txt"
echo "🧠 Mamba3D 两阶段训练配置信息" > ${CONFIG_FILE}
echo "=====================================" >> ${CONFIG_FILE}
echo "训练完成时间: $(date)" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "🔧 模型配置:" >> ${CONFIG_FILE}
echo "模型类型: Mamba3D (两阶段训练)" >> ${CONFIG_FILE}
echo "特征维度: ${DIM_K}" >> ${CONFIG_FILE}
echo "Mamba块数量: ${NUM_MAMBA_BLOCKS}" >> ${CONFIG_FILE}
echo "状态空间维度: ${D_STATE}" >> ${CONFIG_FILE}
echo "扩展因子: ${EXPAND}" >> ${CONFIG_FILE}
echo "聚合函数: ${SYMFN}" >> ${CONFIG_FILE}
echo "模型缩放: ${SCALE}" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "📊 训练配置:" >> ${CONFIG_FILE}
echo "数据集路径: ${DATASET_PATH}" >> ${CONFIG_FILE}
echo "类别文件: ${CATEGORY_FILE}" >> ${CONFIG_FILE}
echo "配对模式: ${PAIR_MODE}" >> ${CONFIG_FILE}
if [ "${PAIR_MODE}" = "scene_reference" ] && [ -n "${REFERENCE_NAME}" ]; then
  echo "参考点云名称: ${REFERENCE_NAME}" >> ${CONFIG_FILE}
fi
echo "数据划分: 场景划分" >> ${CONFIG_FILE}
echo "点云数量: ${NUM_POINTS}" >> ${CONFIG_FILE}
echo "批次大小: ${BATCH_SIZE}" >> ${CONFIG_FILE}
echo "学习率: ${LEARNING_RATE}" >> ${CONFIG_FILE}
echo "优化器: ${OPTIMIZER}" >> ${CONFIG_FILE}
echo "权重衰减: ${WEIGHT_DECAY}" >> ${CONFIG_FILE}
echo "分类器训练轮数: ${EPOCHS_CLASSIFIER}" >> ${CONFIG_FILE}
echo "PointLK训练轮数: ${EPOCHS_POINTLK}" >> ${CONFIG_FILE}
echo "设备: ${DEVICE}" >> ${CONFIG_FILE}
echo "变换幅度: ${MAG}" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "📁 输出文件:" >> ${CONFIG_FILE}
echo "分类器前缀: ${CLASSIFIER_PREFIX}" >> ${CONFIG_FILE}
echo "PointLK前缀: ${POINTLK_PREFIX}" >> ${CONFIG_FILE}
echo "分类器日志: ${CLASSIFIER_LOG}" >> ${CONFIG_FILE}
echo "PointLK日志: ${POINTLK_LOG}" >> ${CONFIG_FILE}
echo "特征权重文件: ${FEAT_WEIGHTS}" >> ${CONFIG_FILE}

echo ""
echo "💾 配置信息已保存到: ${CONFIG_FILE}"

echo ""
echo "🎯 最终结果文件:"
echo "📂 结果目录: /SAN/medic/MRpcr/results/mamba3d_c3vd/"
echo "📄 配置文件: ${CONFIG_FILE}"
echo "📊 分类器模型: ${CLASSIFIER_PREFIX}_model_best.pth"
echo "🔧 特征权重: ${FEAT_WEIGHTS}"
echo "🎯 配准模型: ${POINTLK_PREFIX}_model_best.pth"
echo "📋 分类器日志: ${CLASSIFIER_LOG}"
echo "📋 配准日志: ${POINTLK_LOG}" 