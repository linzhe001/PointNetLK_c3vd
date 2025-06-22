#!/usr/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G            
#$ -l h_rt=36000  # 1小时测试时间
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -pe gpu 1
#$ -N ljiang_test_mamba3d_modelnet
#$ -o /SAN/medic/MRpcr/logs/f_test_mamba3d_modelnet_output.log
#$ -e /SAN/medic/MRpcr/logs/f_test_mamba3d_modelnet_error.log
#$ -wd /SAN/medic/MRpcr

cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
# 设置工作目录
echo "当前工作目录: $(pwd)"

# 激活Conda环境
echo "激活Conda环境..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

# 创建结果和日志目录
mkdir -p /SAN/medic/MRpcr/results/mamba3d_modelnet/test_results
mkdir -p /SAN/medic/MRpcr/results/mamba3d_modelnet/test_results/gt

# Python命令
PY3="nice -n 10 python"

# 设置变量
DATASET_PATH="/SAN/medic/MRpcr/ModelNet40"
CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/modelnet40_half2.txt"
NUM_POINTS=1024
DEVICE="cuda:0"
DATE_TAG=$(date +"%m%d")

# Mamba3D模型配置（与训练保持一致）
DIM_K=1024                    # 特征维度
NUM_MAMBA_BLOCKS=1            # Mamba块数量
D_STATE=8                     # 状态空间维度
EXPAND=2                      # 扩展因子
SYMFN="max"                   # 聚合函数：max, avg, 或 selective
MAX_ITER=20                   # LK最大迭代次数
DELTA=1.0e-2                  # LK步长

# 最大测试样本数
MAX_SAMPLES_ROUND1=1000  # 第一轮测试：角度扰动文件的最大样本数
MAX_SAMPLES_ROUND2=0 # 第二轮测试：精度测试文件的最大样本数

# 可视化设置（可选）
VISUALIZE_PERT="" # 如果需要可视化，设置为 "--visualize-pert pert_010.csv pert_020.csv"
VISUALIZE_SAMPLES=3

# 模型路径（需要根据实际训练结果调整）
MAMBA_MODEL_PREFIX="/SAN/medic/MRpcr/results/mamba3d_modelnet/mamba3d_pointlk_${DATE_TAG}"
MAMBA_MODEL="${MAMBA_MODEL_PREFIX}_model_best.pth"
# 注意：Mamba3D模型只使用配准模型权重，不需要单独的分类器权重

# 如果找不到今天的模型，尝试查找最新模型
if [ ! -f "${MAMBA_MODEL}" ]; then
    echo "⚠️  未找到今天的模型文件: ${MAMBA_MODEL}"
    echo "🔍 搜索最新的Mamba3D模型..."
    
    # 搜索最新的Mamba3D pointlk模型
    LATEST_MODEL=$(find /SAN/medic/MRpcr/results/mamba3d_modelnet/ -name "mamba3d_pointlk_*_model_best.pth" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -n "${LATEST_MODEL}" ] && [ -f "${LATEST_MODEL}" ]; then
        MAMBA_MODEL="${LATEST_MODEL}"
        # 提取模型前缀
        MAMBA_MODEL_PREFIX=$(echo "${MAMBA_MODEL}" | sed 's/_model_best\.pth$//')
        echo "✅ 找到最新模型: ${MAMBA_MODEL}"
    else
        echo "❌ 错误：未找到Mamba3D模型文件！"
        echo "请确保已运行train_mamba3d_modelnet.sh并成功训练模型"
        exit 1
    fi
fi

# 扰动文件配置
PERTURBATION_DIR="/SAN/medic/MRpcr/PointNetLK_c3vd/gt"
GT_POSES_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/gt_poses.csv"

# 测试结果输出目录
TEST_RESULTS_DIR="/SAN/medic/MRpcr/results/mamba3d_modelnet/test_results"
TEST_LOG="${TEST_RESULTS_DIR}/test_log_${DATE_TAG}.log"

# 打印配置信息
echo "========== Mamba3D配准模型测试配置 =========="
echo "🧠 模型类型: Mamba3D配准模型"
echo "📂 数据集路径: ${DATASET_PATH}"
echo "📄 类别文件: ${CATEGORY_FILE}"
echo "🎲 点云数量: ${NUM_POINTS}"
echo "🖥️  设备: ${DEVICE}"
echo "📊 第一轮最大测试样本: ${MAX_SAMPLES_ROUND1}"
echo "📊 第二轮最大测试样本: ${MAX_SAMPLES_ROUND2}"
echo ""
echo "🔧 Mamba3D参数:"
echo "   - 特征维度: ${DIM_K}"
echo "   - Mamba块数量: ${NUM_MAMBA_BLOCKS}"
echo "   - 状态空间维度: ${D_STATE}"
echo "   - 扩展因子: ${EXPAND}"
echo "   - 聚合函数: ${SYMFN}"
echo "   - LK最大迭代次数: ${MAX_ITER}"
echo "   - LK步长: ${DELTA}"
echo ""
echo "📁 模型文件:"
echo "   - 配准模型: ${MAMBA_MODEL}"
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
    CATEGORY_COUNT=$(find "${DATASET_PATH}" -maxdepth 1 -type d | wc -l)
    echo "📊 可用类别数: $((CATEGORY_COUNT - 1))"  # 减去父目录
else
    echo "❌ 错误: 数据集目录不存在: ${DATASET_PATH}"
    exit 1
fi

# 检查类别文件
if [ -f "${CATEGORY_FILE}" ]; then
    echo "✅ 类别文件存在"
    CATEGORY_COUNT=$(wc -l < "${CATEGORY_FILE}")
    echo "📊 测试类别数量: ${CATEGORY_COUNT}"
else
    echo "❌ 错误: 类别文件不存在: ${CATEGORY_FILE}"
    exit 1
fi

# 检查模型文件
if [ -f "${MAMBA_MODEL}" ]; then
    echo "✅ Mamba3D配准模型存在: ${MAMBA_MODEL}"
    MODEL_SIZE=$(du -h "${MAMBA_MODEL}" | cut -f1)
    echo "📊 模型文件大小: ${MODEL_SIZE}"
else
    echo "❌ 错误: Mamba3D配准模型不存在: ${MAMBA_MODEL}"
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
    GT_FILE_SIZE=$(du -h "${GT_POSES_FILE}" | cut -f1)
    GT_FILE_LINES=$(wc -l < "${GT_POSES_FILE}")
    echo "📊 GT姿态文件大小: ${GT_FILE_SIZE}"
    echo "📊 GT姿态条目数量: ${GT_FILE_LINES}"
else
    echo "❌ 错误: GT姿态文件不存在: ${GT_POSES_FILE}"
    echo "请确保已准备好GT姿态文件"
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
echo "========== 开始Mamba3D配准模型测试 =========="
echo "🚀 即将开始两轮测试..."
echo "⏱️  预计测试时间: ~30-60分钟（依据扰动文件数量和样本数量）"
echo ""

# 构建可视化参数
VISUALIZE_PARAMS=""
if [ -n "${VISUALIZE_PERT}" ]; then
    VISUALIZE_PARAMS="${VISUALIZE_PERT} --visualize-samples ${VISUALIZE_SAMPLES}"
fi

# =============================================================================
# 第一轮测试：处理gt文件夹中的扰动文件
# =============================================================================
# echo "========== 第一轮测试：角度扰动文件 =========="
# echo "🎯 测试目标: 处理 gt 文件夹中的扰动文件"
# echo "📂 扰动目录: ${PERTURBATION_DIR}"
# echo "📁 结果存储: 各角度子目录"
# echo ""

# # 第一轮测试的输出前缀
# TEST_OUTPUT_PREFIX_ROUND1="${TEST_RESULTS_DIR}/results"

# # 运行第一轮测试
# echo "🚀 开始第一轮测试..."
# ${PY3} test_pointlk.py \
#   -o ${TEST_OUTPUT_PREFIX_ROUND1} \
#   -i ${DATASET_PATH} \
#   -c ${CATEGORY_FILE} \
#   -l ${TEST_LOG} \
#   --dataset-type modelnet \
#   --num-points ${NUM_POINTS} \
#   --max-iter ${MAX_ITER} \
#   --delta ${DELTA} \
#   --device ${DEVICE} \
#   --max-samples ${MAX_SAMPLES_ROUND1} \
#   --perturbation-dir ${PERTURBATION_DIR} \
#   --model-type mamba3d \
#   --dim-k ${DIM_K} \
#   --num-mamba-blocks ${NUM_MAMBA_BLOCKS} \
#   --d-state ${D_STATE} \
#   --expand ${EXPAND} \
#   --symfn ${SYMFN} \
#   --pretrained ${MAMBA_MODEL} \
#   ${VISUALIZE_PARAMS}

# # 检查第一轮测试结果
# if [ $? -eq 0 ]; then
#     echo ""
#     echo "✅ 第一轮测试（角度扰动）完成!"
    
#     # 统计第一轮结果
#     ANGLE_DIRS=$(find "${TEST_RESULTS_DIR}" -type d -name "angle_*" | wc -l)
#     echo "📊 生成的角度目录数: ${ANGLE_DIRS}"
#     if [ ${ANGLE_DIRS} -gt 0 ]; then
#         echo "📋 角度目录列表:"
#         find "${TEST_RESULTS_DIR}" -type d -name "angle_*" | sort | head -5
#         if [ ${ANGLE_DIRS} -gt 5 ]; then
#             echo "   ... (共${ANGLE_DIRS}个角度目录)"
#         fi
#     fi
# else
#     echo ""
#     echo "❌ 第一轮测试（角度扰动）失败!"
#     echo "请检查错误日志: ${TEST_LOG}"
    
#     # 显示最后几行错误信息
#     if [ -f "${TEST_LOG}" ]; then
#         echo ""
#         echo "📋 最新错误信息:"
#         tail -10 "${TEST_LOG}"
#     fi
    
#     exit 1
# fi

# =============================================================================
# 第二轮测试：处理GT姿态文件
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

# 构建第二轮测试的MAX_SAMPLES参数
MAX_SAMPLES_PARAM_ROUND2=""
if [ ${MAX_SAMPLES_ROUND2} -gt 0 ]; then
    MAX_SAMPLES_PARAM_ROUND2="--max-samples ${MAX_SAMPLES_ROUND2}"
fi

# 运行第二轮测试 - 直接使用GT姿态文件
${PY3} test_pointlk.py \
  -o ${TEST_OUTPUT_PREFIX_ROUND2} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l ${TEST_LOG_ROUND2} \
  --dataset-type modelnet \
  --num-points ${NUM_POINTS} \
  --max-iter ${MAX_ITER} \
  --delta ${DELTA} \
  --device ${DEVICE} \
  ${MAX_SAMPLES_PARAM_ROUND2} \
  --perturbation-file ${GT_POSES_FILE} \
  --model-type mamba3d \
  --dim-k ${DIM_K} \
  --num-mamba-blocks ${NUM_MAMBA_BLOCKS} \
  --d-state ${D_STATE} \
  --expand ${EXPAND} \
  --symfn ${SYMFN} \
  --pretrained ${MAMBA_MODEL} \
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
echo "   - 角度测试目录数: ${ANGLE_DIRS}（第一轮测试已注释掉）"

# 统计GT目录结果
GT_RESULT_FILES=$(find "${TEST_RESULTS_DIR}/gt" -name "*.log" -type f 2>/dev/null | wc -l)
echo "   - GT测试结果文件数: ${GT_RESULT_FILES}"

# 统计总结果文件
TOTAL_RESULT_FILES=$(find "${TEST_RESULTS_DIR}" -name "*.log" -type f | wc -l)
echo "   - 总结果文件数量: ${TOTAL_RESULT_FILES}"

echo ""
echo "📂 结果目录结构:"
echo "   ${TEST_RESULTS_DIR}/"
echo "   ├── angle_*/ (第一轮：角度扰动测试 - 已注释掉)"
echo "   ├── gt/       (第二轮：GT姿态测试)"
echo "   ├── test_log_${DATE_TAG}.log (主测试日志 - 已注释掉)"
echo "   └── gt/test_log_gt_${DATE_TAG}.log (GT测试日志)"

# 保存测试配置信息
CONFIG_FILE="${TEST_RESULTS_DIR}/mamba3d_modelnet_test_${DATE_TAG}_config.txt"
echo "🧠 Mamba3D配准模型双轮测试配置" > ${CONFIG_FILE}
echo "=====================================" >> ${CONFIG_FILE}
echo "测试完成时间: $(date)" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "🔧 测试轮次:" >> ${CONFIG_FILE}
echo "第一轮: 角度扰动文件测试（gt 文件夹中的 pert_*.csv）" >> ${CONFIG_FILE}
echo "第二轮: GT姿态文件测试（gt_poses.csv）" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "🔧 模型配置:" >> ${CONFIG_FILE}
echo "模型类型: Mamba3D配准模型" >> ${CONFIG_FILE}
echo "特征维度: ${DIM_K}" >> ${CONFIG_FILE}
echo "Mamba块数量: ${NUM_MAMBA_BLOCKS}" >> ${CONFIG_FILE}
echo "状态空间维度: ${D_STATE}" >> ${CONFIG_FILE}
echo "扩展因子: ${EXPAND}" >> ${CONFIG_FILE}
echo "聚合函数: ${SYMFN}" >> ${CONFIG_FILE}
echo "LK最大迭代次数: ${MAX_ITER}" >> ${CONFIG_FILE}
echo "LK步长: ${DELTA}" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "📊 测试配置:" >> ${CONFIG_FILE}
echo "数据集路径: ${DATASET_PATH}" >> ${CONFIG_FILE}
echo "类别文件: ${CATEGORY_FILE}" >> ${CONFIG_FILE}
echo "点云数量: ${NUM_POINTS}" >> ${CONFIG_FILE}
echo "第一轮最大测试样本: ${MAX_SAMPLES_ROUND1}" >> ${CONFIG_FILE}
echo "第二轮最大测试样本: ${MAX_SAMPLES_ROUND2}" >> ${CONFIG_FILE}
echo "设备: ${DEVICE}" >> ${CONFIG_FILE}
echo "扰动目录（第一轮）: ${PERTURBATION_DIR}" >> ${CONFIG_FILE}
echo "GT姿态文件（第二轮）: ${GT_POSES_FILE}" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "📁 模型文件:" >> ${CONFIG_FILE}
echo "配准模型: ${MAMBA_MODEL}" >> ${CONFIG_FILE}
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
echo "🎯 配准模型: ${MAMBA_MODEL}"
echo "📋 第一轮（角度）日志: ${TEST_LOG}"
echo "📋 第二轮（GT）日志: ${TEST_LOG_ROUND2}"
echo "⏰ 完成时间: $(date)"

echo ""
echo "🎉🎉🎉 Mamba3D配准模型双轮测试全部完成! 🎉🎉🎉" 
echo "📊 第一轮：角度扰动测试（存储在angle_*目录中）"
echo "📊 第二轮：GT姿态测试（存储在gt目录中）"