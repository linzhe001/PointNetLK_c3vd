#!/usr/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G            
#$ -l h_rt=174000  # 增加到4小时以支持恢复训练
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -pe gpu 1
#$ -N ljiang_resume_cformer_c3vd
#$ -o /SAN/medic/MRpcr/logs/f_cformer_c3vd_resume_output.log
#$ -e /SAN/medic/MRpcr/logs/f_cformer_c3vd_resume_error.log
#$ -wd /SAN/medic/MRpcr

cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
# Set working directory to project root
echo "Current working directory: $(pwd)"

# Activate Conda environment
echo "Activating Conda environment..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

# Create result and log directories
mkdir -p /SAN/medic/MRpcr/results/cformer_c3vd

# Python command
PY3="nice -n 10 python"

# Set variables
DATASET_PATH="/SAN/medic/MRpcr/C3VD_datasets"
CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/c3vd.txt"
NUM_POINTS=1024
DEVICE="cuda:0"
DATE_TAG=$(date +"%m%d")
MAG=0.8 # 用于PointLK的变换幅度

# ===== 恢复训练配置 =====
# 设置要恢复的模型路径 - 请根据实际情况修改这些路径
RESUME_CHECKPOINT=""  # 如果要从checkpoint恢复完整训练状态，设置checkpoint路径
PRETRAINED_MODEL="/SAN/medic/MRpcr/results/cformer_c3vd/cformer_pointlk_0615_model_best.pth"  # 要继续训练的配准模型权重路径

# 体素化配置参数
USE_VOXELIZATION=true           # 是否启用体素化（true/false）
VOXEL_SIZE=4                 # 体素大小 (修改为0.05，更适合医学点云)
VOXEL_GRID_SIZE=32              # 体素网格尺寸
MAX_VOXEL_POINTS=100            # 每个体素最大点数
MAX_VOXELS=20000                # 最大体素数量
MIN_VOXEL_POINTS_RATIO=0.1      # 最小体素点数比例

# CFormer特有配置
DIM_K=1024                    # 特征维度
NUM_PROXY_POINTS=8            # 代理点数量
NUM_BLOCKS=2                  # CFormer块数量
SYMFN="max"                   # 聚合函数：max, avg, 或 cd_pool
SCALE=1                       # 模型缩放因子
BATCH_SIZE=16                 # 批次大小
LEARNING_RATE=0.0001          # 学习率
WEIGHT_DECAY=1e-6             # 权重衰减
OPTIMIZER="Adam"              # 优化器
EPOCHS_POINTLK=200            # PointLK训练轮数

# PointLK使用场景分割和数据增强
POINTLK_SPLIT="--scene-split"  # PointLK使用场景分割
AUGMENTED_PAIR_MODE="all"  # PointLK使用所有配对方式进行数据增强
REFERENCE_NAME=""  # 清空参考点云名称
REFERENCE_PARAM=""
if [ "${PAIR_MODE}" = "scene_reference" ] && [ -n "${REFERENCE_NAME}" ]; then
  REFERENCE_PARAM="--reference-name ${REFERENCE_NAME}"
fi

# 构建体素化参数字符串
VOXELIZATION_PARAMS=""
if [ "${USE_VOXELIZATION}" = "true" ]; then
    VOXELIZATION_PARAMS="--use-voxelization --voxel-size ${VOXEL_SIZE} --voxel-grid-size ${VOXEL_GRID_SIZE} --max-voxel-points ${MAX_VOXEL_POINTS} --max-voxels ${MAX_VOXELS} --min-voxel-points-ratio ${MIN_VOXEL_POINTS_RATIO}"
else
    VOXELIZATION_PARAMS="--no-voxelization"
fi

# 定义输出路径
POINTLK_PREFIX="/SAN/medic/MRpcr/results/cformer_c3vd/cformer_pointlk_resume_${DATE_TAG}"
POINTLK_LOG="${POINTLK_PREFIX}.log"

# 打印配置信息
echo "========== CFormer 配准模型恢复训练配置 =========="
echo "🔄 训练模式: 恢复配准模型训练 (跳过分类器训练)"
echo "🧠 模型类型: CFormer PointLK 配准"
echo "📂 数据集路径: ${DATASET_PATH}"
echo "📄 类别文件: ${CATEGORY_FILE}"
echo "🔗 配对模式: ${AUGMENTED_PAIR_MODE} (数据增强)"
echo "📊 数据划分: 场景划分"
echo "🎯 参考点云: ${REFERENCE_NAME:-'自动选择'}"
echo "🎲 点云数量: ${NUM_POINTS}"
echo "📦 批次大小: ${BATCH_SIZE}"
echo "🖥️  设备: ${DEVICE}"
echo ""
echo "🔄 恢复训练配置:"
if [ -n "${RESUME_CHECKPOINT}" ] && [ -f "${RESUME_CHECKPOINT}" ]; then
    echo "   - 恢复模式: 从Checkpoint恢复 (包含优化器状态)"
    echo "   - Checkpoint路径: ${RESUME_CHECKPOINT}"
elif [ -n "${PRETRAINED_MODEL}" ] && [ -f "${PRETRAINED_MODEL}" ]; then
    echo "   - 恢复模式: 从预训练模型继续训练"
    echo "   - 预训练模型路径: ${PRETRAINED_MODEL}"
else
    echo "   - ⚠️  警告: 未指定有效的恢复路径，将从头开始训练"
fi
echo ""
echo "🔧 体素化配置:"
echo "   - 启用体素化: ${USE_VOXELIZATION}"
if [ "${USE_VOXELIZATION}" = "true" ]; then
    echo "   - 体素大小: ${VOXEL_SIZE}"
    echo "   - 体素网格尺寸: ${VOXEL_GRID_SIZE}"
    echo "   - 每个体素最大点数: ${MAX_VOXEL_POINTS}"
    echo "   - 最大体素数量: ${MAX_VOXELS}"
    echo "   - 最小体素点数比例: ${MIN_VOXEL_POINTS_RATIO}"
fi
echo ""
echo "🔧 CFormer参数:"
echo "   - 特征维度: ${DIM_K}"
echo "   - 代理点数量: ${NUM_PROXY_POINTS}"
echo "   - CFormer块数量: ${NUM_BLOCKS}"
echo "   - 聚合函数: ${SYMFN}"
echo "   - 模型缩放: ${SCALE}"
echo ""
echo "🎯 训练参数:"
echo "   - 学习率: ${LEARNING_RATE}"
echo "   - 优化器: ${OPTIMIZER}"
echo "   - 权重衰减: ${WEIGHT_DECAY}"
echo "   - 训练轮数: ${EPOCHS_POINTLK}"
echo ""
echo "📁 输出路径:"
echo "   - 输出前缀: ${POINTLK_PREFIX}"
echo "   - 日志文件: ${POINTLK_LOG}"

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

# 检查恢复训练文件
echo ""
echo "========== 恢复训练文件检查 =========="
if [ -n "${RESUME_CHECKPOINT}" ]; then
    if [ -f "${RESUME_CHECKPOINT}" ]; then
        echo "✅ Checkpoint文件存在: ${RESUME_CHECKPOINT}"
        CHECKPOINT_SIZE=$(du -h "${RESUME_CHECKPOINT}" | cut -f1)
        echo "📊 文件大小: ${CHECKPOINT_SIZE}"
    else
        echo "❌ 错误: 指定的Checkpoint文件不存在: ${RESUME_CHECKPOINT}"
        echo "🔄 将检查预训练模型..."
        RESUME_CHECKPOINT=""
    fi
fi

if [ -n "${PRETRAINED_MODEL}" ]; then
    if [ -f "${PRETRAINED_MODEL}" ]; then
        echo "✅ 预训练模型文件存在: ${PRETRAINED_MODEL}"
        MODEL_SIZE=$(du -h "${PRETRAINED_MODEL}" | cut -f1)
        echo "📊 文件大小: ${MODEL_SIZE}"
    else
        echo "❌ 警告: 指定的预训练模型文件不存在: ${PRETRAINED_MODEL}"
        echo "🔄 将从头开始训练..."
        PRETRAINED_MODEL=""
    fi
fi

# 等待用户确认（可选）
echo ""
echo "========== 开始恢复训练 =========="
echo "🚀 即将开始CFormer配准模型恢复训练..."
echo "⏱️  预估训练时间: $((${EPOCHS_POINTLK} * 2))分钟"
echo ""

# =============================================
# 直接开始PointLK配准训练（跳过分类器训练）
# =============================================
echo "========== CFormer PointLK 配准恢复训练 =========="
echo "🎯 开始恢复训练CFormer PointLK配准模型..."
echo "📁 输出前缀: ${POINTLK_PREFIX}"
echo "📋 日志文件: ${POINTLK_LOG}"
echo "📊 使用场景数据划分"
echo "🔗 使用数据增强配对模式: ${AUGMENTED_PAIR_MODE}"
if [ "${USE_VOXELIZATION}" = "true" ]; then
    echo "🔧 启用体素化预处理"
else
    echo "🔧 使用标准重采样预处理"
fi
echo ""

# 构建恢复训练参数
RESUME_PARAM=""
PRETRAINED_PARAM=""

if [ -n "${RESUME_CHECKPOINT}" ] && [ -f "${RESUME_CHECKPOINT}" ]; then
    RESUME_PARAM="--resume ${RESUME_CHECKPOINT}"
    echo "🔄 从Checkpoint恢复完整训练状态: ${RESUME_CHECKPOINT}"
elif [ -n "${PRETRAINED_MODEL}" ] && [ -f "${PRETRAINED_MODEL}" ]; then
    PRETRAINED_PARAM="--pretrained ${PRETRAINED_MODEL}"
    echo "🔄 从预训练模型继续训练: ${PRETRAINED_MODEL}"
else
    echo "⚠️  未找到有效的恢复文件，从头开始训练"
fi

# PointLK配准训练（带体素化参数和恢复训练参数）
${PY3} train_pointlk.py \
  -o ${POINTLK_PREFIX} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l ${POINTLK_LOG} \
  --dataset-type c3vd \
  --num-points ${NUM_POINTS} \
  --epochs ${EPOCHS_POINTLK} \
  --batch-size ${BATCH_SIZE} \
  --workers 4 \
  --device ${DEVICE} \
  --drop-last \
  --verbose \
  --mag ${MAG} \
  --model-type cformer \
  --dim-k ${DIM_K} \
  --num-proxy-points ${NUM_PROXY_POINTS} \
  --num-blocks ${NUM_BLOCKS} \
  --symfn ${SYMFN} \
  --optimizer ${OPTIMIZER} \
  --base-lr ${LEARNING_RATE} \
  --warmup-epochs 5 \
  --cosine-annealing \
  --pair-mode ${AUGMENTED_PAIR_MODE} \
  ${POINTLK_SPLIT} \
  ${REFERENCE_PARAM} \
  ${RESUME_PARAM} \
  ${PRETRAINED_PARAM} \
  ${VOXELIZATION_PARAMS}

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 CFormer PointLK配准恢复训练完成!"
    echo ""
    echo "========== 训练总结 =========="
    echo "✅ 配准模型恢复训练完成"
    echo "📁 输出模型: ${POINTLK_PREFIX}_model_best.pth"
    echo "📋 训练日志: ${POINTLK_LOG}"
    if [ "${USE_VOXELIZATION}" = "true" ]; then
        echo "🔧 使用了体素化预处理技术"
    fi
    if [ -n "${RESUME_CHECKPOINT}" ]; then
        echo "🔄 从Checkpoint恢复训练: ${RESUME_CHECKPOINT}"
    elif [ -n "${PRETRAINED_MODEL}" ]; then
        echo "🔄 从预训练模型继续训练: ${PRETRAINED_MODEL}"
    fi
    echo ""
    echo "🚀 训练完成! 可以使用生成的模型进行测试。"
else
    echo ""
    echo "❌ CFormer PointLK配准恢复训练失败!"
    exit 1
fi

# =============================================
# 训练完成总结
# =============================================
echo ""
echo "🎉🎉🎉 CFormer配准模型恢复训练完成! 🎉🎉🎉"
echo ""
echo "📊 训练总结:"
echo "✅ CFormer-PointLK配准恢复训练成功"
echo "   📁 配准模型: ${POINTLK_PREFIX}_*.pth"
echo "   📋 配准日志: ${POINTLK_LOG}"
echo ""
echo "🏷️  标签: resume_${DATE_TAG}"
echo "⏰ 完成时间: $(date)"

# 保存详细配置信息
CONFIG_FILE="/SAN/medic/MRpcr/results/cformer_c3vd/cformer_c3vd_resume_${DATE_TAG}_config.txt"
echo "🧠 CFormer PointLK 配准恢复训练配置信息" > ${CONFIG_FILE}
echo "=====================================" >> ${CONFIG_FILE}
echo "训练完成时间: $(date)" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "🔄 恢复训练信息:" >> ${CONFIG_FILE}
echo "训练模式: 恢复配准模型训练 (跳过分类器训练)" >> ${CONFIG_FILE}
if [ -n "${RESUME_CHECKPOINT}" ]; then
    echo "恢复模式: 从Checkpoint恢复" >> ${CONFIG_FILE}
    echo "Checkpoint路径: ${RESUME_CHECKPOINT}" >> ${CONFIG_FILE}
elif [ -n "${PRETRAINED_MODEL}" ]; then
    echo "恢复模式: 从预训练模型继续训练" >> ${CONFIG_FILE}
    echo "预训练模型路径: ${PRETRAINED_MODEL}" >> ${CONFIG_FILE}
fi
echo "" >> ${CONFIG_FILE}
echo "🔧 模型配置:" >> ${CONFIG_FILE}
echo "模型类型: CFormer PointLK 配准" >> ${CONFIG_FILE}
echo "特征维度: ${DIM_K}" >> ${CONFIG_FILE}
echo "代理点数量: ${NUM_PROXY_POINTS}" >> ${CONFIG_FILE}
echo "CFormer块数量: ${NUM_BLOCKS}" >> ${CONFIG_FILE}
echo "聚合函数: ${SYMFN}" >> ${CONFIG_FILE}
echo "模型缩放: ${SCALE}" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "📊 训练配置:" >> ${CONFIG_FILE}
echo "数据集路径: ${DATASET_PATH}" >> ${CONFIG_FILE}
echo "类别文件: ${CATEGORY_FILE}" >> ${CONFIG_FILE}
echo "配对模式: ${AUGMENTED_PAIR_MODE} (数据增强)" >> ${CONFIG_FILE}
if [ -n "${REFERENCE_NAME}" ]; then
  echo "参考点云名称: ${REFERENCE_NAME}" >> ${CONFIG_FILE}
fi
echo "数据划分: 场景划分" >> ${CONFIG_FILE}
echo "点云数量: ${NUM_POINTS}" >> ${CONFIG_FILE}
echo "批次大小: ${BATCH_SIZE}" >> ${CONFIG_FILE}
echo "学习率: ${LEARNING_RATE}" >> ${CONFIG_FILE}
echo "优化器: ${OPTIMIZER}" >> ${CONFIG_FILE}
echo "权重衰减: ${WEIGHT_DECAY}" >> ${CONFIG_FILE}
echo "训练轮数: ${EPOCHS_POINTLK}" >> ${CONFIG_FILE}
echo "设备: ${DEVICE}" >> ${CONFIG_FILE}
echo "变换幅度: ${MAG}" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "🔧 体素化配置:" >> ${CONFIG_FILE}
echo "启用体素化: ${USE_VOXELIZATION}" >> ${CONFIG_FILE}
if [ "${USE_VOXELIZATION}" = "true" ]; then
    echo "体素大小: ${VOXEL_SIZE}" >> ${CONFIG_FILE}
    echo "体素网格尺寸: ${VOXEL_GRID_SIZE}" >> ${CONFIG_FILE}
    echo "每个体素最大点数: ${MAX_VOXEL_POINTS}" >> ${CONFIG_FILE}
    echo "最大体素数量: ${MAX_VOXELS}" >> ${CONFIG_FILE}
    echo "最小体素点数比例: ${MIN_VOXEL_POINTS_RATIO}" >> ${CONFIG_FILE}
fi
echo "" >> ${CONFIG_FILE}
echo "📁 输出文件:" >> ${CONFIG_FILE}
echo "输出前缀: ${POINTLK_PREFIX}" >> ${CONFIG_FILE}
echo "训练日志: ${POINTLK_LOG}" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "🔧 数据增强说明:" >> ${CONFIG_FILE}
echo "包含一对一配对、源点云到源点云配对和目标点云到目标点云配对" >> ${CONFIG_FILE}
echo "通过'all'配对模式实现更丰富的训练数据" >> ${CONFIG_FILE}

echo ""
echo "💾 配置信息已保存到: ${CONFIG_FILE}"

echo ""
echo "🎯 最终结果文件:"
echo "📂 结果目录: /SAN/medic/MRpcr/results/cformer_c3vd/"
echo "📄 配置文件: ${CONFIG_FILE}"
echo "🎯 配准模型: ${POINTLK_PREFIX}_model_best.pth"
echo "📋 配准日志: ${POINTLK_LOG}"
if [ -n "${RESUME_CHECKPOINT}" ]; then
    echo "🔄 原Checkpoint: ${RESUME_CHECKPOINT}"
elif [ -n "${PRETRAINED_MODEL}" ]; then
    echo "🔄 原预训练模型: ${PRETRAINED_MODEL}"
fi 