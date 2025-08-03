#!/usr/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=36000  # 1小时测试时间
#$ -l gpu=true

#$ -pe gpu 1
#$ -N ljiang_test_new_mamba_c3vd
#$ -o /SAN/medic/MRpcr/logs/f_test_new_mamba_c3vd_output.log
#$ -e /SAN/medic/MRpcr/logs/f_test_new_mamba_c3vd_error.log
#$ -wd /SAN/medic/MRpcr

cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
# 设置工作目录
echo "当前工作目录: $(pwd)"

# 激活Conda环境
echo "激活Conda环境..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

# 创建结果和日志目录
mkdir -p /SAN/medic/MRpcr/results/new_mamba_c3vd/test_results
mkdir -p /SAN/medic/MRpcr/results/new_mamba_c3vd/test_results/gt

# Python命令
PY3="nice -n 10 python"

# 设置变量
DATASET_PATH="/SAN/medic/MRpcr/C3VD_datasets"
CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/c3vd.txt"
NUM_POINTS=1024
DEVICE="cuda:0"
DATE_TAG=$(date +"%m%d")

# 体素化配置参数（与训练保持一致）
USE_VOXELIZATION=false           # 是否启用体素化（true/false），与new_mamba训练脚本保持一致
VOXEL_SIZE=4                 # 体素大小 (适合医学点云)
VOXEL_GRID_SIZE=32              # 体素网格尺寸
MAX_VOXEL_POINTS=100            # 每个体素最大点数
MAX_VOXELS=20000                # 最大体素数量
MIN_VOXEL_POINTS_RATIO=0.1      # 最小体素点数比例

# 构建体素化参数字符串
VOXELIZATION_PARAMS=""
if [ "${USE_VOXELIZATION}" = "true" ]; then
    VOXELIZATION_PARAMS="--use-voxelization --voxel-size ${VOXEL_SIZE} --voxel-grid-size ${VOXEL_GRID_SIZE} --max-voxel-points ${MAX_VOXEL_POINTS} --max-voxels ${MAX_VOXELS} --min-voxel-points-ratio ${MIN_VOXEL_POINTS_RATIO}"
else
    VOXELIZATION_PARAMS="--no-voxelization"
fi

# Mamba3D模型配置（与训练保持一致）
DIM_K=1024                    # 特征维度
NUM_MAMBA_BLOCKS=1            # Mamba块数量
D_STATE=8                     # 状态空间维度
EXPAND=2                      # 扩展因子
SYMFN="max"                   # 聚合函数：max, avg, 或 selective
MAX_ITER=10                   # LK最大迭代次数（与训练保持一致）
DELTA=1.0e-4                  # LK步长（与训练保持一致）

# C3VD配对模式设置（与训练保持一致）
PAIR_MODE="one_to_one"  # 使用点对点配对模式
REFERENCE_NAME=""  # 清空参考点云名称
REFERENCE_PARAM=""
if [ "${PAIR_MODE}" = "scene_reference" ] && [ -n "${REFERENCE_NAME}" ]; then
  REFERENCE_PARAM="--reference-name ${REFERENCE_NAME}"
fi

# 最大测试样本数
MAX_SAMPLES_ROUND1=1000  # 第一轮测试：角度扰动文件的最大样本数
MAX_SAMPLES_ROUND2=0     # 第二轮测试：GT姿态文件使用所有样本（0表示无限制）

# 可视化设置（可选）
VISUALIZE_PERT="" # 如需可视化，设置为 "--visualize-pert pert_010.csv pert_020.csv"
VISUALIZE_SAMPLES=3

# 模型路径（自动查找最新）
MODEL_DIR="/SAN/medic/MRpcr/results/new_mamba_c3vd"
MAMBA3D_MODEL_PREFIX="${MODEL_DIR}/new_mamba_pointlk_${DATE_TAG}"
MAMBA3D_MODEL="${MAMBA3D_MODEL_PREFIX}_model_best.pth"

if [ ! -f "${MAMBA3D_MODEL}" ]; then
    echo "⚠️ 未找到今日新Mamba3D模型: ${MAMBA3D_MODEL}"
    # 修改查找逻辑以匹配文件名 (new_mamba_pointlk_... 或 new_mamba_pointlk_resume_...)
    LATEST_MODEL=$(find ${MODEL_DIR} -name "new_mamba_pointlk_*_model_best.pth" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -f2- -d' ')
    if [ -n "${LATEST_MODEL}" ] && [ -f "${LATEST_MODEL}" ]; then
        MAMBA3D_MODEL="${LATEST_MODEL}"
        echo "✅ 找到最新新Mamba3D模型: ${MAMBA3D_MODEL}"
    else
        echo "❌ 错误: 未找到任何新Mamba3D模型文件!"
        exit 1
    fi
else
    echo "✅ 使用今日新Mamba3D模型: ${MAMBA3D_MODEL}"
fi

# 提取模型前缀用于日志记录
MAMBA3D_MODEL_PREFIX=$(basename "${MAMBA3D_MODEL}" | sed 's/_model_best\.pth$//')


# 检查指定的模型文件是否存在
if [ ! -f "${MAMBA3D_MODEL}" ]; then
    echo "❌ 错误: 指定的新Mamba3D模型文件不存在: ${MAMBA3D_MODEL}"
    echo "请确保模型文件路径正确"
    exit 1
else
    echo "✅ 使用指定的新Mamba3D模型: ${MAMBA3D_MODEL}"
    MODEL_SIZE=$(du -h "${MAMBA3D_MODEL}" | cut -f1)
    echo "📊 模型文件大小: ${MODEL_SIZE}"
fi

# 扰动文件配置
PERTURBATION_DIR="/SAN/medic/MRpcr/PointNetLK_c3vd/gt"
GT_POSES_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/gt_poses.csv"

# 测试结果输出目录
TEST_RESULTS_DIR="/SAN/medic/MRpcr/results/new_mamba_c3vd/test_results"
TEST_LOG="${TEST_RESULTS_DIR}/test_log_${DATE_TAG}.log"

# 打印配置信息
echo "========== 新Mamba3D配准模型测试配置 =========="
echo "🧠 模型类型: 新Mamba3D配准模型"
echo "📂 数据集路径: ${DATASET_PATH}"
echo "📄 类别文件: ${CATEGORY_FILE}"
echo "🔗 配对模式: ${PAIR_MODE}"
echo "🎯 参考点云: ${REFERENCE_NAME:-'自动选择'}"
echo "🎲 点云数量: ${NUM_POINTS}"
echo "🖥️  设备: ${DEVICE}"
echo "📊 第一轮最大测试样本: ${MAX_SAMPLES_ROUND1}"
echo "📊 第二轮最大测试样本: ${MAX_SAMPLES_ROUND2:-'无限制'}"
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
echo "   - LK最大迭代: ${MAX_ITER}"
echo "   - LK步长: ${DELTA}"
echo ""
echo "📁 模型文件:"
echo "   - 配准模型: ${MAMBA3D_MODEL}"
echo "   - 扰动目录: ${PERTURBATION_DIR}"
echo "   - GT姿态文件: ${GT_POSES_FILE}"
echo ""
echo "📁 输出路径:"
echo "   - 测试结果目录: ${TEST_RESULTS_DIR}"
echo "   - 测试日志: ${TEST_LOG}"

# 检查必要文件
echo ""
echo "========== 文件检查 =========="

# 检查数据集
if [ -d "${DATASET_PATH}" ]; then
    echo "✅ 数据集目录存在"
    SOURCE_COUNT=$(find "${DATASET_PATH}/C3VD_ply_source" -name "*.ply" 2>/dev/null | wc -l)
    TARGET_COUNT=$(find "${DATASET_PATH}/visible_point_cloud_ply_depth" -name "*.ply" 2>/dev/null | wc -l)
    echo "📊 源点云文件数量: ${SOURCE_COUNT}"
    echo "📊 目标点云文件数量: ${TARGET_COUNT}"
else
    echo "❌ 错误: 数据集目录不存在: ${DATASET_PATH}"
    exit 1
fi

# 检查类别文件
if [ -f "${CATEGORY_FILE}" ]; then
    echo "✅ 类别文件存在"
    CATEGORY_COUNT=$(wc -l < "${CATEGORY_FILE}")
    echo "📊 类别数量: ${CATEGORY_COUNT}"
else
    echo "❌ 错误: 类别文件不存在: ${CATEGORY_FILE}"
    exit 1
fi

# 检查扰动目录
if [ -d "${PERTURBATION_DIR}" ]; then
    echo "✅ 扰动目录存在"
    PERT_COUNT=$(find "${PERTURBATION_DIR}" -name "pert_*.csv" | wc -l)
    echo "📊 扰动文件数量: ${PERT_COUNT}"
    if [ ${PERT_COUNT} -eq 0 ]; then
        echo "⚠️  警告: 扰动目录中没有pert_*.csv文件"
    else
        echo "📋 扰动文件列表:"
        find "${PERTURBATION_DIR}" -name "pert_*.csv" | sort | head -5
        if [ ${PERT_COUNT} -gt 5 ]; then
            echo "   ... (共${PERT_COUNT}个扰动文件)"
        fi
    fi
else
    echo "❌ 错误: 扰动目录不存在: ${PERTURBATION_DIR}"
    exit 1
fi

# 检查GT姿态文件
if [ -f "${GT_POSES_FILE}" ]; then
    echo "✅ GT姿态文件存在: ${GT_POSES_FILE}"
    GT_POSES_SIZE=$(du -h "${GT_POSES_FILE}" | cut -f1)
    GT_POSES_LINES=$(wc -l < "${GT_POSES_FILE}")
    echo "📊 GT姿态文件大小: ${GT_POSES_SIZE}"
    echo "📊 GT姿态条目数量: ${GT_POSES_LINES}"
else
    echo "❌ 错误: GT姿态文件不存在: ${GT_POSES_FILE}"
    exit 1
fi

# GPU内存检查
echo ""
echo "========== GPU状态检查 =========="
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️  GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️  无法获取GPU信息"
fi

echo ""
echo "========== 开始新Mamba3D配准模型测试 =========="
echo "🚀 即将开始两轮测试..."
echo "⏱️  预计测试时间: ~60-120分钟（依据扰动文件数量和样本数量）"
if [ "${USE_VOXELIZATION}" = "true" ]; then
    echo "🔧 启用体素化预处理，可能会稍微增加处理时间"
fi
echo ""

# 构建可视化参数
VISUALIZE_PARAMS=""
if [ -n "${VISUALIZE_PERT}" ]; then
    VISUALIZE_PARAMS="${VISUALIZE_PERT} --visualize-samples ${VISUALIZE_SAMPLES}"
fi

# =============================================================================
# 第一轮测试：处理gt文件夹中的扰动文件（0-90度）
# =============================================================================
echo "========== 第一轮测试：角度扰动文件 =========="
echo "🎯 测试目标: 处理 gt 文件夹中的 10 个扰动文件（0-90度）"
echo "📂 扰动目录: ${PERTURBATION_DIR}"
echo "📁 结果存储: 各角度子目录（angle_000 到 angle_090）"
echo ""

# 第一轮测试的输出前缀
TEST_OUTPUT_PREFIX_ROUND1="${TEST_RESULTS_DIR}/results"

# 运行第一轮测试
echo "🚀 开始第一轮测试..."
${PY3} test_pointlk.py \
  -o ${TEST_OUTPUT_PREFIX_ROUND1} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l ${TEST_LOG} \
  --dataset-type c3vd \
  --num-points ${NUM_POINTS} \
  --max-iter ${MAX_ITER} \
  --delta ${DELTA} \
  --device ${DEVICE} \
  --max-samples ${MAX_SAMPLES_ROUND1} \
  --pair-mode ${PAIR_MODE} \
  ${REFERENCE_PARAM} \
  --perturbation-dir ${PERTURBATION_DIR} \
  --model-type mamba3d \
  --dim-k ${DIM_K} \
  --num-mamba-blocks ${NUM_MAMBA_BLOCKS} \
  --d-state ${D_STATE} \
  --expand ${EXPAND} \
  --symfn ${SYMFN} \
  --pretrained ${MAMBA3D_MODEL} \
  ${VOXELIZATION_PARAMS} \
  ${VISUALIZE_PARAMS}

# 检查第一轮测试结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 第一轮测试（角度扰动）完成!"

    # 统计第一轮结果
    ANGLE_DIRS=$(find "${TEST_RESULTS_DIR}" -type d -name "angle_*" | wc -l)
    echo "📊 生成的角度目录数: ${ANGLE_DIRS}"
    if [ ${ANGLE_DIRS} -gt 0 ]; then
        echo "📋 角度目录列表:"
        find "${TEST_RESULTS_DIR}" -type d -name "angle_*" | sort | head -5
        if [ ${ANGLE_DIRS} -gt 5 ]; then
            echo "   ... (共${ANGLE_DIRS}个角度目录)"
        fi
    fi
else
    echo ""
    echo "❌ 第一轮测试（角度扰动）失败!"
    echo "请检查错误日志: ${TEST_LOG}"

    # 显示最后几行错误信息
    if [ -f "${TEST_LOG}" ]; then
        echo ""
        echo "📋 最新错误信息:"
        tail -10 "${TEST_LOG}"
    fi

    exit 1
fi

# =============================================================================
# 第二轮测试：处理单独的gt_poses.csv文件
# =============================================================================
echo ""
echo "========== 第二轮测试：GT姿态文件 =========="
echo "🎯 测试目标: 处理GT姿态文件"
echo "📄 GT文件: ${GT_POSES_FILE}"
echo "📁 结果存储: gt 子目录"
echo ""

# 第二轮测试的输出前缀（指向gt子目录）
TEST_OUTPUT_PREFIX_ROUND2="${TEST_RESULTS_DIR}/gt/results"
TEST_LOG_ROUND2="${TEST_RESULTS_DIR}/gt/test_log_gt_${DATE_TAG}.log"

echo "🚀 开始第二轮测试..."
echo "📄 直接使用GT姿态文件: ${GT_POSES_FILE}"
echo "🎯 GT_POSES模式将自动激活（每个扰动随机选择一个测试样本）"

# 运行第二轮测试 - 直接使用GT姿态文件
${PY3} test_pointlk.py \
  -o ${TEST_OUTPUT_PREFIX_ROUND2} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l ${TEST_LOG_ROUND2} \
  --dataset-type c3vd \
  --num-points ${NUM_POINTS} \
  --max-iter ${MAX_ITER} \
  --delta ${DELTA} \
  --device ${DEVICE} \
  --max-samples ${MAX_SAMPLES_ROUND2} \
  --pair-mode ${PAIR_MODE} \
  ${REFERENCE_PARAM} \
  --perturbation-file ${GT_POSES_FILE} \
  --model-type mamba3d \
  --dim-k ${DIM_K} \
  --num-mamba-blocks ${NUM_MAMBA_BLOCKS} \
  --d-state ${D_STATE} \
  --expand ${EXPAND} \
  --symfn ${SYMFN} \
  --pretrained ${MAMBA3D_MODEL} \
  ${VOXELIZATION_PARAMS} \
  ${VISUALIZE_PARAMS}

# 检查第二轮测试结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 第二轮测试（GT姿态）完成!"
    
    # 统计第二轮结果
    GT_RESULT_FILES=$(find "${TEST_RESULTS_DIR}/gt" -name "*.log" -type f | wc -l)
    echo "📊 生成的GT结果文件数: ${GT_RESULT_FILES}"
    if [ ${GT_RESULT_FILES} -gt 0 ]; then
        echo "📋 GT结果文件列表:"
        find "${TEST_RESULTS_DIR}/gt" -name "*.log" -type f | sort | head -5
    fi
else
    echo ""
    echo "❌ 第二轮测试（GT姿态）失败!"
    echo "请检查错误日志: ${TEST_LOG_ROUND2}"
    
    # 显示最后几行错误信息
    if [ -f "${TEST_LOG_ROUND2}" ]; then
        echo ""
        echo "📋 最新错误信息:"
        tail -10 "${TEST_LOG_ROUND2}"
    fi
    
    exit 1
fi

echo ""
echo "🎯 GT_POSES模式测试说明:"
echo "   - 直接使用了gt_poses.csv文件"
echo "   - 系统自动检测到文件名包含'gt_poses'，启用随机选择模式"
echo "   - 每个扰动随机选择一个测试样本进行测试"
echo "   - 总测试次数等于扰动数量（而不是数据集大小）"

# =============================================================================
# 最终结果汇总
# =============================================================================
echo ""
echo "🎉 所有测试完成!"
echo "📁 测试结果保存到: ${TEST_RESULTS_DIR}"
echo "📋 主测试日志: ${TEST_LOG}"
echo "📋 GT测试日志: ${TEST_LOG_ROUND2}"

# 显示生成的结果文件汇总
echo ""
echo "📊 最终测试结果汇总:"

# 统计角度目录（第一轮测试被注释掉，角度目录数为0）
ANGLE_DIRS=$(find "${TEST_RESULTS_DIR}" -type d -name "angle_*" 2>/dev/null | wc -l)
echo "   - 角度测试目录数: ${ANGLE_DIRS}"

# 统计GT目录结果
GT_RESULT_FILES=$(find "${TEST_RESULTS_DIR}/gt" -name "*.log" -type f 2>/dev/null | wc -l)
echo "   - GT测试结果文件数: ${GT_RESULT_FILES}"

# 统计总结果文件
TOTAL_RESULT_FILES=$(find "${TEST_RESULTS_DIR}" -name "*.log" -type f | wc -l)
echo "   - 总结果文件数量: ${TOTAL_RESULT_FILES}"

echo ""
echo "📂 结果目录结构:"
echo "   ${TEST_RESULTS_DIR}/"
echo "   ├── angle_*/ (第一轮：角度扰动测试)"
echo "   ├── gt/       (第二轮：GT姿态测试)"
echo "   ├── test_log_${DATE_TAG}.log (主测试日志)"
echo "   └── gt/test_log_gt_${DATE_TAG}.log (GT测试日志)"

# 保存测试配置信息
CONFIG_FILE="${TEST_RESULTS_DIR}/new_mamba_test_${DATE_TAG}_config.txt"
echo "🧠 新Mamba3D配准模型双轮测试配置" > ${CONFIG_FILE}
echo "=====================================" >> ${CONFIG_FILE}
echo "测试完成时间: $(date)" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "🔧 测试轮次:" >> ${CONFIG_FILE}
echo "第一轮: 角度扰动文件测试（gt 文件夹中的 pert_*.csv）" >> ${CONFIG_FILE}
echo "第二轮: GT姿态文件测试（gt_poses.csv）" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "🔧 模型配置:" >> ${CONFIG_FILE}
echo "模型类型: 新Mamba3D配准模型" >> ${CONFIG_FILE}
echo "特征维度: ${DIM_K}" >> ${CONFIG_FILE}
echo "Mamba块数量: ${NUM_MAMBA_BLOCKS}" >> ${CONFIG_FILE}
echo "状态空间维度: ${D_STATE}" >> ${CONFIG_FILE}
echo "扩展因子: ${EXPAND}" >> ${CONFIG_FILE}
echo "聚合函数: ${SYMFN}" >> ${CONFIG_FILE}
echo "LK最大迭代: ${MAX_ITER}" >> ${CONFIG_FILE}
echo "LK步长: ${DELTA}" >> ${CONFIG_FILE}
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
echo "📊 测试配置:" >> ${CONFIG_FILE}
echo "数据集路径: ${DATASET_PATH}" >> ${CONFIG_FILE}
echo "类别文件: ${CATEGORY_FILE}" >> ${CONFIG_FILE}
echo "配对模式: ${PAIR_MODE}" >> ${CONFIG_FILE}
if [ "${PAIR_MODE}" = "scene_reference" ] && [ -n "${REFERENCE_NAME}" ]; then
  echo "参考点云名称: ${REFERENCE_NAME}" >> ${CONFIG_FILE}
fi
echo "点云数量: ${NUM_POINTS}" >> ${CONFIG_FILE}
echo "第一轮最大测试样本: ${MAX_SAMPLES_ROUND1}" >> ${CONFIG_FILE}
echo "第二轮最大测试样本: ${MAX_SAMPLES_ROUND2:-'无限制'}" >> ${CONFIG_FILE}
echo "设备: ${DEVICE}" >> ${CONFIG_FILE}
echo "扰动目录（第一轮）: ${PERTURBATION_DIR}" >> ${CONFIG_FILE}
echo "GT姿态文件（第二轮）: ${GT_POSES_FILE}" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "📁 模型文件:" >> ${CONFIG_FILE}
echo "配准模型: ${MAMBA3D_MODEL}" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "📁 输出文件:" >> ${CONFIG_FILE}
echo "第一轮测试结果: ${TEST_OUTPUT_PREFIX_ROUND1}" >> ${CONFIG_FILE}
echo "第二轮测试结果: ${TEST_OUTPUT_PREFIX_ROUND2}" >> ${CONFIG_FILE}
echo "主测试日志: ${TEST_LOG}" >> ${CONFIG_FILE}
echo "GT测试日志: ${TEST_LOG_ROUND2}" >> ${CONFIG_FILE}

echo ""
echo "💾 测试配置信息已保存到: ${CONFIG_FILE}"

echo ""
echo "🎯 测试完成总结:"
echo "📂 结果目录: ${TEST_RESULTS_DIR}"
echo "📄 配置文件: ${CONFIG_FILE}"
echo "🎯 配准模型: ${MAMBA3D_MODEL}"
echo "📋 第一轮（角度）日志: ${TEST_LOG}"
echo "📋 第二轮（GT）日志: ${TEST_LOG_ROUND2}"
if [ "${USE_VOXELIZATION}" = "true" ]; then
    echo "🔧 使用了体素化预处理技术"
else
    echo "🔧 未使用体素化"
fi
echo "⏰ 完成时间: $(date)"

echo ""
echo "🎉🎉🎉 新Mamba3D配准模型双轮测试全部完成! 🎉🎉🎉" 
echo "📊 第一轮：角度扰动测试（存储在angle_*目录中）"
echo "📊 第二轮：GT姿态测试（存储在gt目录中）" 