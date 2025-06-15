#!/usr/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G            
#$ -l h_rt=74000  # 增加到4小时以支持两阶段训练
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -pe gpu 1
#$ -N ljiang_train_attention_c3vd_full
#$ -o /SAN/medic/MRpcr/logs/f_attention_c3vd_output.log
#$ -e /SAN/medic/MRpcr/logs/f_attention_c3vd_error.log
#$ -wd /SAN/medic/MRpcr

cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
# Set working directory to project root
echo "Current working directory: $(pwd)"

# Activate Conda environment
echo "Activating Conda environment..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

# Create result and log directories
mkdir -p /SAN/medic/MRpcr/results/attention_c3vd

# Python command
PY3="nice -n 10 python"

# Set variables
DATASET_PATH="/SAN/medic/MRpcr/C3VD_datasets"
CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/c3vd.txt"
NUM_POINTS=1024
DEVICE="cuda:0"
DATE_TAG="0602"  # 固定使用0602日期的分类器训练结果
MAG=0.8 # 用于PointLK的变换幅度

# AttentionNet特有配置
DIM_K=1024                    # 特征维度
NUM_ATTENTION_BLOCKS=2        # Attention块数量
NUM_HEADS=4                   # 多头注意力头数
SYMFN="max"                   # 聚合函数：max, avg, 或 attention
SCALE=1                       # 模型缩放因子
BATCH_SIZE=4                 # 批次大小
LEARNING_RATE=0.0001           # 学习率
WEIGHT_DECAY=1e-6            # 权重衰减
OPTIMIZER="Adam"             # 优化器
EPOCHS_CLASSIFIER=150        # 分类器训练轮数
EPOCHS_POINTLK=200           # PointLK训练轮数

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
CLASSIFIER_PREFIX="/SAN/medic/MRpcr/results/attention_c3vd/attention_classifier_${DATE_TAG}"
POINTLK_PREFIX="/SAN/medic/MRpcr/results/attention_c3vd/attention_pointlk_${DATE_TAG}"
CLASSIFIER_LOG="${CLASSIFIER_PREFIX}.log"
POINTLK_LOG="${POINTLK_PREFIX}.log"

# 打印配置信息
echo "========== AttentionNet 两阶段训练配置 =========="
echo "🧠 模型类型: AttentionNet (两阶段训练)"
echo "📂 数据集路径: ${DATASET_PATH}"
echo "📄 类别文件: ${CATEGORY_FILE}"
echo "🔗 配对模式: ${PAIR_MODE}"
echo "🎯 参考点云: ${REFERENCE_NAME:-'自动选择'}"
echo "📊 数据划分: 场景划分"
echo "🎲 点云数量: ${NUM_POINTS}"
echo "📦 批次大小: ${BATCH_SIZE}"
echo "🖥️  设备: ${DEVICE}"
echo ""
echo "🔧 AttentionNet参数:"
echo "   - 特征维度: ${DIM_K}"
echo "   - Attention块数量: ${NUM_ATTENTION_BLOCKS}"
echo "   - 多头注意力头数: ${NUM_HEADS}"
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

# 估算GPU内存需求
echo ""
echo "========== GPU内存需求估算 =========="
ESTIMATED_MEMORY=$((${BATCH_SIZE} * ${NUM_POINTS} * ${NUM_POINTS} * ${NUM_HEADS} * 4 / 1024 / 1024))
RECOMMENDED_MEMORY=$((${ESTIMATED_MEMORY} * 2))
echo "💾 预估单批次内存: ~${ESTIMATED_MEMORY}MB"
echo "🎯 建议GPU内存: >${RECOMMENDED_MEMORY}MB"

if [ ${ESTIMATED_MEMORY} -gt 8000 ]; then
    echo "⚠️  警告: 内存需求较高，建议减小批次大小"
    SUGGESTED_BATCH=$((${BATCH_SIZE} * 8000 / ${ESTIMATED_MEMORY}))
    echo "💡 建议批次大小: ${SUGGESTED_BATCH}"
fi

# 等待用户确认（可选）
echo ""
echo "========== 开始两阶段训练 =========="
echo "🚀 即将开始AttentionNet两阶段训练..."
echo "📋 第一阶段: 分类器训练 (~$((${EPOCHS_CLASSIFIER} * 2))分钟)"
echo "📋 第二阶段: PointLK配准训练 (~$((${EPOCHS_POINTLK} * 2))分钟)"
echo "⏱️  预估总训练时间: $(($((${EPOCHS_CLASSIFIER} + ${EPOCHS_POINTLK})) * 2))分钟"
echo ""

# =============================================
# 第一阶段：训练AttentionNet分类器
# =============================================
echo "========== 第一阶段：训练AttentionNet分类器 =========="
echo "🧠 开始训练AttentionNet分类器..."
echo "📁 输出前缀: ${CLASSIFIER_PREFIX}"
echo "📋 日志文件: ${CLASSIFIER_LOG}"
echo ""

${PY3} train_classifier.py \
  -o ${CLASSIFIER_PREFIX} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l ${CLASSIFIER_LOG} \
  --dataset-type c3vd \
  --num-points ${NUM_POINTS} \
  --epochs ${EPOCHS_CLASSIFIER} \
  --batch-size ${BATCH_SIZE} \
  --workers 2 \
  --device ${DEVICE} \
  --drop-last \
  --verbose \
  --model-type attention \
  --dim-k ${DIM_K} \
  --num-attention-blocks ${NUM_ATTENTION_BLOCKS} \
  --num-heads ${NUM_HEADS} \
  --symfn ${SYMFN} \
  --optimizer ${OPTIMIZER} \
  --base-lr ${LEARNING_RATE} \
  --warmup-epochs 5 \
  --cosine-annealing

# 检查分类器训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 第一阶段：AttentionNet分类器训练完成!"
    echo "📁 模型保存位置: ${CLASSIFIER_PREFIX}_*.pth"
    echo "📋 日志文件: ${CLASSIFIER_LOG}"
    
    # 检查生成的分类器模型文件
    echo ""
    echo "📊 生成的分类器模型文件:"
    ls -lh ${CLASSIFIER_PREFIX}_*.pth 2>/dev/null || echo "❌ 未找到分类器模型文件"
    
    # 查找特征权重文件
    FEAT_WEIGHTS="${CLASSIFIER_PREFIX}_feat_best.pth"
    if [ -f "${FEAT_WEIGHTS}" ]; then
        echo "✅ 找到特征权重文件: ${FEAT_WEIGHTS}"
    else
        echo "❌ 警告: 未找到特征权重文件 ${FEAT_WEIGHTS}"
        echo "   尝试查找其他权重文件..."
        ls -la ${CLASSIFIER_PREFIX}_*.pth
    fi
    
else
    echo ""
    echo "❌ 第一阶段：AttentionNet分类器训练失败!"
    echo "请检查错误日志: ${CLASSIFIER_LOG}"
    exit 1
fi

# =============================================
# 第二阶段：训练AttentionNet-PointLK配准模型
# =============================================
echo ""
echo "========== 第二阶段：训练AttentionNet-PointLK配准模型 =========="
echo "🧠 开始训练AttentionNet-PointLK配准模型..."
echo "📁 输出前缀: ${POINTLK_PREFIX}"
echo "📋 日志文件: ${POINTLK_LOG}"
echo "🔄 使用预训练权重: ${FEAT_WEIGHTS}"
echo ""

# 确保特征权重文件存在
if [ ! -f "${FEAT_WEIGHTS}" ]; then
    echo "❌ 错误: 特征权重文件不存在: ${FEAT_WEIGHTS}"
    echo "   无法进行第二阶段训练"
    exit 1
fi

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
  --workers 2 \
  --device ${DEVICE} \
  --drop-last \
  --verbose \
  --pair-mode ${PAIR_MODE} \
  ${REFERENCE_PARAM} \
  ${SCENE_SPLIT} \
  --model-type attention \
  --dim-k ${DIM_K} \
  --num-attention-blocks ${NUM_ATTENTION_BLOCKS} \
  --num-heads ${NUM_HEADS} \
  --symfn ${SYMFN} \
  --pointnet tune \
  --transfer-from ${FEAT_WEIGHTS} \
  ${RESUME_ARG}

# 检查PointLK训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 第二阶段：AttentionNet-PointLK配准训练完成!"
    echo "📁 模型保存位置: ${POINTLK_PREFIX}_*.pth"
    echo "📋 日志文件: ${POINTLK_LOG}"
    
    # 检查生成的PointLK模型文件
    echo ""
    echo "📊 生成的PointLK模型文件:"
    ls -lh ${POINTLK_PREFIX}_*.pth 2>/dev/null || echo "❌ 未找到PointLK模型文件"
    
else
    echo ""
    echo "❌ 第二阶段：AttentionNet-PointLK配准训练失败!"
    echo "请检查错误日志: ${POINTLK_LOG}"
    exit 1
fi

# =============================================
# 训练完成总结
# =============================================
echo ""
echo "🎉🎉🎉 AttentionNet两阶段训练全部完成! 🎉🎉🎉"
echo ""
echo "📊 训练总结:"
echo "✅ 第一阶段：AttentionNet分类器训练成功"
echo "   📁 分类器模型: ${CLASSIFIER_PREFIX}_*.pth"
echo "   📋 分类器日志: ${CLASSIFIER_LOG}"
echo ""
echo "✅ 第二阶段：AttentionNet-PointLK配准训练成功"
echo "   📁 配准模型: ${POINTLK_PREFIX}_*.pth"
echo "   📋 配准日志: ${POINTLK_LOG}"
echo ""
echo "🏷️  标签: ${DATE_TAG}"
echo "⏰ 完成时间: $(date)"

# 保存详细配置信息
CONFIG_FILE="/SAN/medic/MRpcr/results/attention_c3vd/attention_c3vd_${DATE_TAG}_config.txt"
echo "🧠 AttentionNet 两阶段训练配置信息" > ${CONFIG_FILE}
echo "=====================================" >> ${CONFIG_FILE}
echo "训练完成时间: $(date)" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "🔧 模型配置:" >> ${CONFIG_FILE}
echo "模型类型: AttentionNet (两阶段训练)" >> ${CONFIG_FILE}
echo "特征维度: ${DIM_K}" >> ${CONFIG_FILE}
echo "Attention块数量: ${NUM_ATTENTION_BLOCKS}" >> ${CONFIG_FILE}
echo "多头注意力头数: ${NUM_HEADS}" >> ${CONFIG_FILE}
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
echo "💾 内存估算:" >> ${CONFIG_FILE}
echo "预估单批次内存: ~${ESTIMATED_MEMORY}MB" >> ${CONFIG_FILE}
echo "建议GPU内存: >${RECOMMENDED_MEMORY}MB" >> ${CONFIG_FILE}
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
echo "📂 结果目录: /SAN/medic/MRpcr/results/attention_c3vd/"
echo "📄 配置文件: ${CONFIG_FILE}"
echo "📊 分类器模型: ${CLASSIFIER_PREFIX}_model_best.pth"
echo "🔧 特征权重: ${FEAT_WEIGHTS}"
echo "🎯 配准模型: ${POINTLK_PREFIX}_model_best.pth"
echo "📋 分类器日志: ${CLASSIFIER_LOG}"
echo "📋 配准日志: ${POINTLK_LOG}"

# Disabled after training
# sudo swapoff /swapfile
# sudo rm /swapfile
