#!/usr/bin/bash
#$ -l tmem=16G
#$ -l h_vmem=16G            
#$ -l h_rt=3600  # 1小时测试时间
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -pe gpu 1
#$ -N ljiang_test_fixed_pair_mamba3d
#$ -o /SAN/medic/MRpcr/logs/f_test_fixed_pair_mamba3d_output.log
#$ -e /SAN/medic/MRpcr/logs/f_test_fixed_pair_mamba3d_error.log
#$ -wd /SAN/medic/MRpcr

cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
# 设置工作目录
echo "当前工作目录: $(pwd)"

# 激活Conda环境
echo "激活Conda环境..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

# 创建结果和日志目录
mkdir -p /SAN/medic/MRpcr/results/fixed_pair_test
mkdir -p /SAN/medic/MRpcr/results/fixed_pair_test/logs

# Python命令
PY3="nice -n 10 python"

# 基本设置
DEVICE="cuda:0"
DATE_TAG=$(date +"%m%d_%H%M")

# =============================================================================
# 固定点云对配置
# =============================================================================
# 数据集路径
DATASET_PATH="/SAN/medic/MRpcr/C3VD_datasets"

# 固定的源点云和目标点云路径 (用户可自定义)
SOURCE_CLOUD="${DATASET_PATH}/C3VD_ply_source/trans_t3_b/0000_depth_pcd.ply"
TARGET_CLOUD="${DATASET_PATH}/visible_point_cloud_ply_depth/trans_t3_b/frame_0000_visible.ply"

# 固定扰动值 (rx,ry,rz,tx,ty,tz) - 用户可自定义
# 格式：旋转角度(弧度),平移距离(毫米)
FIXED_PERTURBATION="0.1,0.15,0.05,2.0,1.5,0.8"

# 点云参数
NUM_POINTS=1024

# =============================================================================
# 模型配置 (与训练保持一致)
# =============================================================================
# 体素化配置参数
USE_VOXELIZATION=false            # 是否启用体素化（true/false）
VOXEL_SIZE=4                      # 体素大小 (适合医学点云)
VOXEL_GRID_SIZE=32               # 体素网格尺寸
MAX_VOXEL_POINTS=100             # 每个体素最大点数
MAX_VOXELS=20000                 # 最大体素数量
MIN_VOXEL_POINTS_RATIO=0.1       # 最小体素点数比例

# 构建体素化参数字符串
VOXELIZATION_PARAMS=""
if [ "${USE_VOXELIZATION}" = "true" ]; then
    VOXELIZATION_PARAMS="--use-voxelization --voxel-size ${VOXEL_SIZE} --voxel-grid-size ${VOXEL_GRID_SIZE} --max-voxel-points ${MAX_VOXEL_POINTS} --max-voxels ${MAX_VOXELS} --min-voxel-points-ratio ${MIN_VOXEL_POINTS_RATIO}"
else
    VOXELIZATION_PARAMS="--no-voxelization"
fi

# Mamba3D模型配置
DIM_K=1024                       # 特征维度
NUM_MAMBA_BLOCKS=1               # Mamba块数量
D_STATE=8                        # 状态空间维度
EXPAND=2                         # 扩展因子
SYMFN="max"                      # 聚合函数：max, avg, 或 selective
MAX_ITER=20                      # LK最大迭代次数
DELTA=1.0e-4                     # LK步长

# =============================================================================
# 模型路径配置
# =============================================================================
MODEL_DIR="/SAN/medic/MRpcr/results/mamba3d_c3vd"
MAMBA3D_MODEL_PREFIX="${MODEL_DIR}/mamba3d_pointlk_${DATE_TAG}"
MAMBA3D_MODEL="${MAMBA3D_MODEL_PREFIX}_model_best.pth"

# 自动查找最新模型文件
if [ ! -f "${MAMBA3D_MODEL}" ]; then
    echo "⚠️ 未找到今日Mamba3D模型: ${MAMBA3D_MODEL}"
    # 查找最新的模型文件
    LATEST_MODEL=$(find ${MODEL_DIR} -name "mamba3d_pointlk_*_model_best.pth" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -f2- -d' ')
    if [ -n "${LATEST_MODEL}" ] && [ -f "${LATEST_MODEL}" ]; then
        MAMBA3D_MODEL="${LATEST_MODEL}"
        echo "✅ 找到最新Mamba3D模型: ${MAMBA3D_MODEL}"
    else
        echo "❌ 错误: 未找到任何Mamba3D模型文件!"
        exit 1
    fi
else
    echo "✅ 使用今日Mamba3D模型: ${MAMBA3D_MODEL}"
fi

# =============================================================================
# 输出配置
# =============================================================================
# 结果输出路径
OUTPUT_CSV="/SAN/medic/MRpcr/results/fixed_pair_test/fixed_pair_result_${DATE_TAG}.csv"
LOG_FILE="/SAN/medic/MRpcr/results/fixed_pair_test/logs/fixed_pair_test_${DATE_TAG}.log"

# =============================================================================
# 预检查
# =============================================================================
echo ""
echo "========== 固定点云对测试配置检查 =========="
echo "📂 源点云文件: ${SOURCE_CLOUD}"
echo "📂 目标点云文件: ${TARGET_CLOUD}"
echo "🎯 固定扰动值: ${FIXED_PERTURBATION}"
echo "🔧 使用模型: ${MAMBA3D_MODEL}"
echo "💾 结果输出: ${OUTPUT_CSV}"

# 检查点云文件是否存在
if [ ! -f "${SOURCE_CLOUD}" ]; then
    echo "❌ 错误: 源点云文件不存在: ${SOURCE_CLOUD}"
    exit 1
fi

if [ ! -f "${TARGET_CLOUD}" ]; then
    echo "❌ 错误: 目标点云文件不存在: ${TARGET_CLOUD}"
    exit 1
fi

if [ ! -f "${MAMBA3D_MODEL}" ]; then
    echo "❌ 错误: 模型文件不存在: ${MAMBA3D_MODEL}"
    exit 1
fi

echo "✅ 所有文件检查通过!"

# GPU内存检查
echo ""
echo "========== GPU状态检查 =========="
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️  GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️  无法获取GPU信息"
fi

# =============================================================================
# 开始测试
# =============================================================================
echo ""
echo "========== 开始固定点云对配准测试 =========="
echo "🚀 测试开始时间: $(date)"
echo "🎯 测试目标: 使用固定扰动对固定点云对进行配准"
echo "⏱️  预计测试时间: 1-2分钟"
echo ""

# 运行固定点云对测试
echo "🚀 开始单对点云配准测试..."
${PY3} test_pointlk.py \
  -o ${OUTPUT_CSV} \
  -i ${DATASET_PATH} \
  -c /dev/null \
  -l ${LOG_FILE} \
  --dataset-type c3vd \
  --num-points ${NUM_POINTS} \
  --max-iter ${MAX_ITER} \
  --delta ${DELTA} \
  --device ${DEVICE} \
  --model-type mamba3d \
  --dim-k ${DIM_K} \
  --num-mamba-blocks ${NUM_MAMBA_BLOCKS} \
  --d-state ${D_STATE} \
  --expand ${EXPAND} \
  --symfn ${SYMFN} \
  --pretrained ${MAMBA3D_MODEL} \
  ${VOXELIZATION_PARAMS} \
  --single-pair-mode \
  --source-cloud ${SOURCE_CLOUD} \
  --target-cloud ${TARGET_CLOUD} \
  --single-perturbation ${FIXED_PERTURBATION} \
  --enhanced-output

# 检查测试结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 固定点云对测试完成!"
    echo "📊 结果文件: ${OUTPUT_CSV}"
    echo "📝 日志文件: ${LOG_FILE}"
    
    # 显示结果摘要
    if [ -f "${OUTPUT_CSV}" ]; then
        echo ""
        echo "========== 测试结果摘要 =========="
        echo "📋 结果文件大小: $(du -h ${OUTPUT_CSV} | cut -f1)"
        echo "📈 结果行数: $(wc -l < ${OUTPUT_CSV})"
        
        # 显示结果文件的前几行
        echo ""
        echo "📊 结果文件内容预览:"
        head -n 5 "${OUTPUT_CSV}"
        echo ""
    fi
    
    # 显示日志文件的最后几行
    if [ -f "${LOG_FILE}" ]; then
        echo "📋 测试日志最后几行:"
        tail -n 10 "${LOG_FILE}"
    fi
    
else
    echo ""
    echo "❌ 固定点云对测试失败!"
    echo "请检查错误日志: ${LOG_FILE}"
    
    # 显示最后几行错误信息
    if [ -f "${LOG_FILE}" ]; then
        echo ""
        echo "🔍 错误日志最后几行:"
        tail -n 10 "${LOG_FILE}"
    fi
    
    exit 1
fi

echo ""
echo "🎉 固定点云对测试完成时间: $(date)"
echo "==========================================" 