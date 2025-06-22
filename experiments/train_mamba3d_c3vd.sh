#!/usr/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G            
#$ -l h_rt=174000  # 增加到4小时以支持恢复训练
#$ -l gpu=true

#$ -pe gpu 1
#$ -N ljiang_resume_mamba3d_c3vd
#$ -o /SAN/medic/MRpcr/logs/f_mamba3d_c3vd_resume_output.log
#$ -e /SAN/medic/MRpcr/logs/f_mamba3d_c3vd_resume_error.log
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
DATE_TAG=$(date +"%m%d_%H%M")  # 例如: 0620_1430
MAG=0.8 # 用于PointLK的变换幅度

# ===== 恢复训练配置 =====
# 设置要恢复的模型路径 - 请根据实际情况修改这些路径
RESUME_CHECKPOINT=""  # 如果要从checkpoint恢复完整训练状态，设置checkpoint路径
PRETRAINED_MODEL="/SAN/medic/MRpcr/results/mamba3d_c3vd/mamba3d_pointlk_resume_0620_model_best.pth"  # 要继续训练的配准模型权重路径

# 体素化配置参数
USE_VOXELIZATION=true           # 是否启用体素化（true/false）
VOXEL_SIZE=4                 # 体素大小 (适合医学点云)
VOXEL_GRID_SIZE=32              # 体素网格尺寸
MAX_VOXEL_POINTS=100            # 每个体素最大点数
MAX_VOXELS=20000                # 最大体素数量
MIN_VOXEL_POINTS_RATIO=0.1      # 最小体素点数比例

# Mamba3D特有配置
DIM_K=1024                    # 特征维度
NUM_MAMBA_BLOCKS=1            # Mamba块数量
D_STATE=8                     # 状态空间维度
EXPAND=2                      # 扩展因子
SYMFN="max"                   # 聚合函数：max, avg, 或 selective
BATCH_SIZE=16                 # 批次大小
LEARNING_RATE=0.0001          # 学习率
WEIGHT_DECAY=1e-6             # 权重衰减
OPTIMIZER="Adam"              # 优化器
EPOCHS_POINTLK=200            # PointLK训练轮数

# PointLK使用场景分割和数据增强
POINTLK_SPLIT="--scene-split"  # PointLK使用场景分割
AUGMENTED_PAIR_MODE="one_to_one"  # PointLK使用所有配对方式进行数据增强
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
POINTLK_PREFIX="/SAN/medic/MRpcr/results/mamba3d_c3vd/mamba3d_pointlk_resume_${DATE_TAG}"
POINTLK_LOG="${POINTLK_PREFIX}.log"

# 打印配置信息
echo "========== Mamba3D 配准模型恢复训练配置 =========="
echo "🔄 训练模式: 恢复配准模型训练 (跳过分类器训练)"
echo "🧠 模型类型: Mamba3D PointLK 配准"
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
echo "🔧 Mamba3D参数:"
echo "   - 特征维度: ${DIM_K}"
echo "   - Mamba块数量: ${NUM_MAMBA_BLOCKS}"
echo "   - 状态空间维度: ${D_STATE}"
echo "   - 扩展因子: ${EXPAND}"
echo "   - 聚合函数: ${SYMFN}"
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
echo "🚀 即将开始Mamba3D配准模型恢复训练..."
echo "⏱️  预估训练时间: $((${EPOCHS_POINTLK} * 2))分钟"
echo ""

# =============================================
# 直接开始PointLK配准训练（跳过分类器训练）
# =============================================
echo "========== Mamba3D PointLK 配准恢复训练 =========="
echo "🎯 开始恢复训练Mamba3D PointLK配准模型..."
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
    echo "⚠️  从头开始训练 (未指定恢复路径)"
fi

echo ""
echo "🎯 开始Mamba3D PointLK配准训练..."

# Mamba3D PointLK配准训练
${PY3} train_pointlk.py \
  -o ${POINTLK_PREFIX} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l ${POINTLK_LOG} \
  --dataset-type c3vd \
  --num-points ${NUM_POINTS} \
  --mag ${MAG} \
  --device ${DEVICE} \
  --batch-size ${BATCH_SIZE} \
  --epochs ${EPOCHS_POINTLK} \
  --optimizer ${OPTIMIZER} \
  --base-lr ${LEARNING_RATE} \
  --warmup-epochs 5 \
  --cosine-annealing \
  --workers 4 \
  --drop-last \
  --verbose \
  --pair-mode ${AUGMENTED_PAIR_MODE} \
  ${REFERENCE_PARAM} \
  ${POINTLK_SPLIT} \
  --model-type mamba3d \
  --dim-k ${DIM_K} \
  --num-mamba-blocks ${NUM_MAMBA_BLOCKS} \
  --d-state ${D_STATE} \
  --expand ${EXPAND} \
  --symfn ${SYMFN} \
  ${VOXELIZATION_PARAMS} \
  ${RESUME_PARAM} \
  ${PRETRAINED_PARAM}

# 检查PointLK训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Mamba3D PointLK配准模型恢复训练完成!"
    echo "📁 模型保存路径: ${POINTLK_PREFIX}_model_best.pth"
    echo "📋 训练日志: ${POINTLK_LOG}"
    
    # 显示模型文件信息
    if [ -f "${POINTLK_PREFIX}_model_best.pth" ]; then
        echo "✅ 最佳配准模型已保存"
        MODEL_SIZE=$(du -h "${POINTLK_PREFIX}_model_best.pth" | cut -f1)
        echo "📊 模型文件大小: ${MODEL_SIZE}"
    fi
    
    if [ -f "${POINTLK_PREFIX}_snap_best.pth" ]; then
        echo "✅ 最佳训练快照已保存"
        SNAP_SIZE=$(du -h "${POINTLK_PREFIX}_snap_best.pth" | cut -f1)
        echo "📊 快照文件大小: ${SNAP_SIZE}"
    fi
    
    # 显示训练日志的最后几行
    if [ -f "${POINTLK_LOG}" ]; then
        echo ""
        echo "📋 训练完成信息:"
        tail -5 "${POINTLK_LOG}"
    fi
    
else
    echo ""
    echo "❌ Mamba3D PointLK配准模型恢复训练失败!"
    echo "请检查错误日志: ${POINTLK_LOG}"
    
    # 显示最后几行错误信息
    if [ -f "${POINTLK_LOG}" ]; then
        echo ""
        echo "📋 最新错误信息:"
        tail -10 "${POINTLK_LOG}"
    fi
    
    exit 1
fi

# 保存训练配置信息
CONFIG_FILE="/SAN/medic/MRpcr/results/mamba3d_c3vd/mamba3d_resume_${DATE_TAG}_config.txt"
echo "🧠 Mamba3D配准模型恢复训练配置" > ${CONFIG_FILE}
echo "=====================================" >> ${CONFIG_FILE}
echo "训练完成时间: $(date)" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "🔧 模型配置:" >> ${CONFIG_FILE}
echo "模型类型: Mamba3D PointLK 配准" >> ${CONFIG_FILE}
echo "特征维度: ${DIM_K}" >> ${CONFIG_FILE}
echo "Mamba块数量: ${NUM_MAMBA_BLOCKS}" >> ${CONFIG_FILE}
echo "状态空间维度: ${D_STATE}" >> ${CONFIG_FILE}
echo "扩展因子: ${EXPAND}" >> ${CONFIG_FILE}
echo "聚合函数: ${SYMFN}" >> ${CONFIG_FILE}
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
echo "🎯 训练配置:" >> ${CONFIG_FILE}
echo "数据集路径: ${DATASET_PATH}" >> ${CONFIG_FILE}
echo "类别文件: ${CATEGORY_FILE}" >> ${CONFIG_FILE}
echo "配对模式: ${AUGMENTED_PAIR_MODE}" >> ${CONFIG_FILE}
echo "数据划分: 场景划分" >> ${CONFIG_FILE}
if [ -n "${REFERENCE_NAME}" ]; then
  echo "参考点云名称: ${REFERENCE_NAME}" >> ${CONFIG_FILE}
fi
echo "点云数量: ${NUM_POINTS}" >> ${CONFIG_FILE}
echo "批次大小: ${BATCH_SIZE}" >> ${CONFIG_FILE}
echo "学习率: ${LEARNING_RATE}" >> ${CONFIG_FILE}
echo "优化器: ${OPTIMIZER}" >> ${CONFIG_FILE}
echo "权重衰减: ${WEIGHT_DECAY}" >> ${CONFIG_FILE}
echo "训练轮数: ${EPOCHS_POINTLK}" >> ${CONFIG_FILE}
echo "设备: ${DEVICE}" >> ${CONFIG_FILE}
echo "变换幅度: ${MAG}" >> ${CONFIG_FILE}
echo "联合归一化: 是" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "🔄 恢复训练配置:" >> ${CONFIG_FILE}
if [ -n "${RESUME_CHECKPOINT}" ] && [ -f "${RESUME_CHECKPOINT}" ]; then
    echo "恢复模式: 从Checkpoint恢复" >> ${CONFIG_FILE}
    echo "Checkpoint路径: ${RESUME_CHECKPOINT}" >> ${CONFIG_FILE}
elif [ -n "${PRETRAINED_MODEL}" ] && [ -f "${PRETRAINED_MODEL}" ]; then
    echo "恢复模式: 从预训练模型继续训练" >> ${CONFIG_FILE}
    echo "预训练模型路径: ${PRETRAINED_MODEL}" >> ${CONFIG_FILE}
else
    echo "恢复模式: 从头开始训练" >> ${CONFIG_FILE}
fi
echo "" >> ${CONFIG_FILE}
echo "📁 输出文件:" >> ${CONFIG_FILE}
echo "输出前缀: ${POINTLK_PREFIX}" >> ${CONFIG_FILE}
echo "训练日志: ${POINTLK_LOG}" >> ${CONFIG_FILE}
echo "最佳模型: ${POINTLK_PREFIX}_model_best.pth" >> ${CONFIG_FILE}
echo "最佳快照: ${POINTLK_PREFIX}_snap_best.pth" >> ${CONFIG_FILE}

echo ""
echo "💾 训练配置信息已保存到: ${CONFIG_FILE}"

echo ""
echo "🎯 恢复训练完成总结:"
echo "📂 结果目录: /SAN/medic/MRpcr/results/mamba3d_c3vd/"
echo "📄 配置文件: ${CONFIG_FILE}"
echo "🎯 最佳配准模型: ${POINTLK_PREFIX}_model_best.pth"
echo "📋 训练日志: ${POINTLK_LOG}"
if [ "${USE_VOXELIZATION}" = "true" ]; then
    echo "🔧 使用了体素化预处理技术"
fi
echo "⏰ 完成时间: $(date)"

echo ""
echo "🎉🎉🎉 Mamba3D配准模型恢复训练全部完成! 🎉🎉🎉" 