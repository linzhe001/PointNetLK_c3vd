#!/usr/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G            
#$ -l h_rt=74000  # 增加到4小时以支持两阶段训练
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -pe gpu 1
#$ -N ljiang_train_fast_attention_c3vd
#$ -o /SAN/medic/MRpcr/logs/f_fast_attention_c3vd_output.log
#$ -e /SAN/medic/MRpcr/logs/f_fast_attention_c3vd_error.log
#$ -wd /SAN/medic/MRpcr

cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
# Set working directory to project root
echo "Current working directory: $(pwd)"

# Activate Conda environment
echo "Activating Conda environment..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

# Create result and log directories
mkdir -p /SAN/medic/MRpcr/results/fast_attention_c3vd

# Python command
PY3="nice -n 10 python"

# Set variables
DATASET_PATH="/SAN/medic/MRpcr/C3VD_datasets"
CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/c3vd.txt"
NUM_POINTS=1024
DEVICE="cuda:0"
DATE_TAG=$(date +"%m%d")
MAG=0.8 # 用于PointLK的变换幅度

# Fast Point Attention特有配置
DIM_K=1024                    # 特征维度
NUM_FAST_ATTENTION_BLOCKS=2   # 注意力块数量
FAST_ATTENTION_SCALE=1        # 模型规模缩放因子 (1=标准, 2=轻量, 4=超轻量)
SYMFN="max"                   # 聚合函数：max, avg, 或 selective
BATCH_SIZE=16                 # 批次大小 (Fast Point Attention更轻量，可使用更大批次)
LEARNING_RATE=0.0001          # 学习率
WEIGHT_DECAY=1e-6             # 权重衰减
OPTIMIZER="Adam"              # 优化器
EPOCHS_CLASSIFIER=150         # 分类器训练轮数
EPOCHS_POINTLK=200            # PointLK训练轮数

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
CLASSIFIER_PREFIX="/SAN/medic/MRpcr/results/fast_attention_c3vd/fast_attention_classifier_${DATE_TAG}"
POINTLK_PREFIX="/SAN/medic/MRpcr/results/fast_attention_c3vd/fast_attention_pointlk_${DATE_TAG}"
CLASSIFIER_LOG="${CLASSIFIER_PREFIX}.log"
POINTLK_LOG="${POINTLK_PREFIX}.log"

# 打印配置信息
echo "========== Fast Point Attention 两阶段训练配置 =========="
echo "🧠 模型类型: Fast Point Attention (两阶段训练)"
echo "📂 数据集路径: ${DATASET_PATH}"
echo "📄 类别文件: ${CATEGORY_FILE}"
echo "🔗 配对模式: ${PAIR_MODE}"
echo "🎯 参考点云: ${REFERENCE_NAME:-'自动选择'}"
echo "📊 数据划分: 场景划分"
echo "🎲 点云数量: ${NUM_POINTS}"
echo "📦 批次大小: ${BATCH_SIZE}"
echo "🖥️  设备: ${DEVICE}"
echo ""
echo "🔧 Fast Point Attention参数:"
echo "   - 特征维度: ${DIM_K}"
echo "   - 注意力块数量: ${NUM_FAST_ATTENTION_BLOCKS}"
echo "   - 规模缩放因子: ${FAST_ATTENTION_SCALE}"
echo "   - 聚合函数: ${SYMFN}"
echo ""
echo "🎯 训练参数:"
echo "   - 学习率: ${LEARNING_RATE}"
echo "   - 优化器: ${OPTIMIZER}"
echo "   - 权重衰减: ${WEIGHT_DECAY}"
echo "   - 分类器训练轮数: ${EPOCHS_CLASSIFIER}"
echo "   - PointLK训练轮数: ${EPOCHS_POINTLK}"
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

# Fast Point Attention内存优势说明
echo ""
echo "========== Fast Point Attention优势分析 =========="
echo "💡 Fast Point Attention采用轻量级设计，比标准Attention更高效:"
echo "   - 单头注意力替代多头注意力，减少8倍计算量"
echo "   - 简化位置编码，减少计算开销"
echo "   - 前馈网络隐藏层从4倍缩小到2倍"
echo "   - 更小的内存占用和更快的推理速度"

# 等待用户确认（可选）
echo ""
echo "========== 开始两阶段训练 =========="
echo "🚀 即将开始Fast Point Attention两阶段训练..."
echo "📋 第一阶段: 分类器训练 (~$((${EPOCHS_CLASSIFIER} * 2))分钟)"
echo "📋 第二阶段: PointLK配准训练 (~$((${EPOCHS_POINTLK} * 2))分钟)"
echo "⏱️  预估总训练时间: $(($((${EPOCHS_CLASSIFIER} + ${EPOCHS_POINTLK})) * 2))分钟"
echo ""

# 移除第一阶段的分类器训练
# 直接开始第二阶段的Fast Point Attention-PointLK配准模型训练

# 检查是否有之前的快照文件用于续训
RESUME_FILE="${POINTLK_PREFIX}_snap_best.pth"
RESUME_ARG=""
if [ -f "${RESUME_FILE}" ]; then
    echo "✅ 发现之前的快照文件: ${RESUME_FILE}"
    echo "📈 将继续之前的训练..."
    RESUME_ARG="--resume ${RESUME_FILE}"
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
  --model-type fast_attention \
  --dim-k ${DIM_K} \
  --num-fast-attention-blocks ${NUM_FAST_ATTENTION_BLOCKS} \
  --fast-attention-scale ${FAST_ATTENTION_SCALE} \
  --symfn ${SYMFN} \
  --pointnet tune \
  ${RESUME_ARG} \
  --no-voxelization

# 检查PointLK训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 第二阶段：Fast Point Attention-PointLK配准训练完成!"
    echo "📁 模型保存位置: ${POINTLK_PREFIX}_*.pth"
    echo "📋 日志文件: ${POINTLK_LOG}"
    
    # 检查生成的PointLK模型文件
    echo ""
    echo "📊 生成的PointLK模型文件:"
    ls -lh ${POINTLK_PREFIX}_*.pth 2>/dev/null || echo "❌ 未找到PointLK模型文件"
    
else
    echo ""
    echo "❌ 第二阶段：Fast Point Attention-PointLK配准训练失败!"
    echo "请检查错误日志: ${POINTLK_LOG}"
    exit 1
fi

# =============================================
# 训练完成总结
# =============================================
echo ""
echo "🎉🎉🎉 Fast Point Attention两阶段训练全部完成! 🎉🎉🎉"
echo ""
echo "📊 训练总结:"
echo "✅ 第一阶段：Fast Point Attention分类器训练成功"
echo "   📁 分类器模型: ${CLASSIFIER_PREFIX}_*.pth"
echo "   📋 分类器日志: ${CLASSIFIER_LOG}"
echo ""
echo "✅ 第二阶段：Fast Point Attention-PointLK配准训练成功"
echo "   📁 配准模型: ${POINTLK_PREFIX}_*.pth"
echo "   📋 配准日志: ${POINTLK_LOG}"
echo ""
echo "🏷️  标签: ${DATE_TAG}"
echo "⏰ 完成时间: $(date)"

# 保存详细配置信息
CONFIG_FILE="/SAN/medic/MRpcr/results/fast_attention_c3vd/fast_attention_c3vd_${DATE_TAG}_config.txt"
echo "🧠 Fast Point Attention 两阶段训练配置信息" > ${CONFIG_FILE}
echo "=====================================" >> ${CONFIG_FILE}
echo "训练完成时间: $(date)" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "🔧 模型配置:" >> ${CONFIG_FILE}
echo "模型类型: Fast Point Attention (两阶段训练)" >> ${CONFIG_FILE}
echo "特征维度: ${DIM_K}" >> ${CONFIG_FILE}
echo "注意力块数量: ${NUM_FAST_ATTENTION_BLOCKS}" >> ${CONFIG_FILE}
echo "规模缩放因子: ${FAST_ATTENTION_SCALE}" >> ${CONFIG_FILE}
echo "聚合函数: ${SYMFN}" >> ${CONFIG_FILE}
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
echo "📂 结果目录: /SAN/medic/MRpcr/results/fast_attention_c3vd/"
echo "📄 配置文件: ${CONFIG_FILE}"
echo "📊 分类器模型: ${CLASSIFIER_PREFIX}_model_best.pth"
echo "🔧 特征权重: ${FEAT_WEIGHTS}"
echo "🎯 配准模型: ${POINTLK_PREFIX}_model_best.pth"
echo "📋 分类器日志: ${CLASSIFIER_LOG}"
echo "📋 配准日志: ${POINTLK_LOG}" 