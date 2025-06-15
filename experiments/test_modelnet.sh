#!/usr/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G            
#$ -l h_rt=3600  # 1 hour test time
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -pe gpu 1
#$ -N ljiang_test_pointnet_modelnet
#$ -o /SAN/medic/MRpcr/logs/f_test_pointnet_modelnet_output.log
#$ -e /SAN/medic/MRpcr/logs/f_test_pointnet_modelnet_error.log
#$ -wd /SAN/medic/MRpcr

cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
# Set working directory
echo "Current working directory: $(pwd)"

# Activate Conda environment
echo "Activating Conda environment..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

# Create results and log directories
mkdir -p /SAN/medic/MRpcr/results/pointnet_modelnet/test_results

# Python command
PY3="nice -n 10 python"

# Set variables
DATASET_PATH="/SAN/medic/MRpcr/ModelNet40"
CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/modelnet40_half2.txt"
NUM_POINTS=1024
DEVICE="cuda:0"
DATE_TAG=$(date +"%m%d")

# PointNet model configuration (keep consistent with training)
DIM_K=1024                    # Feature dimension
SYMFN="max"                   # Aggregation function: max or avg
MAX_ITER=20                   # LK maximum iterations
DELTA=1.0e-2                  # LK step size

# Maximum test samples
MAX_SAMPLES=1000

# Visualization settings (optional)
VISUALIZE_PERT="" # If visualization needed, set to "--visualize-pert pert_010.csv pert_020.csv"
VISUALIZE_SAMPLES=3

# Model path - using specified trained model weights
POINTNET_MODEL="/SAN/medic/MRpcr/results/modelnet/modelnet_pointlk_0603_model_best.pth"
CLASSIFIER_MODEL="/SAN/medic/MRpcr/results/modelnet/modelnet_classifier_0603_model_best.pth"

# Print model information
echo "🎯 Using specified model weights:"
echo "   - Registration model: ${POINTNET_MODEL}"
echo "   - Classifier model: ${CLASSIFIER_MODEL}"

# Perturbation file directory
PERTURBATION_DIR="/SAN/medic/MRpcr/PointNetLK_c3vd/gt"

# Test results output directory
TEST_RESULTS_DIR="/SAN/medic/MRpcr/results/pointnet_modelnet/test_results"
# Base name for output files - results will be stored in subdirectories based on angle
TEST_OUTPUT_PREFIX="${TEST_RESULTS_DIR}/results"
TEST_LOG="${TEST_RESULTS_DIR}/test_log_${DATE_TAG}.log"

# Print configuration information
echo "========== PointNet Registration Model Test Configuration =========="
echo "🧠 Model type: PointNet registration model"
echo "📂 Dataset path: ${DATASET_PATH}"
echo "📄 Category file: ${CATEGORY_FILE}"
echo "🎲 Number of points: ${NUM_POINTS}"
echo "🖥️  Device: ${DEVICE}"
echo "📊 Maximum test samples: ${MAX_SAMPLES}"
echo ""
echo "🔧 PointNet parameters:"
echo "   - Feature dimension: ${DIM_K}"
echo "   - Aggregation function: ${SYMFN}"
echo "   - LK max iterations: ${MAX_ITER}"
echo "   - LK step size: ${DELTA}"
echo ""
echo "📁 Model files:"
echo "   - Registration model: ${POINTNET_MODEL}"
echo "   - Classifier model: ${CLASSIFIER_MODEL}"
echo "   - Perturbation directory: ${PERTURBATION_DIR}"
echo ""
echo "📁 Output paths:"
echo "   - Test result prefix: ${TEST_OUTPUT_PREFIX}"
echo "   - Test log: ${TEST_LOG}"

# Check necessary files
echo ""
echo "========== File Check =========="

# Check dataset
if [ -d "${DATASET_PATH}" ]; then
    echo "✅ Dataset directory exists"
    CATEGORY_COUNT=$(find "${DATASET_PATH}" -maxdepth 1 -type d | wc -l)
    echo "📊 Available categories: $((CATEGORY_COUNT - 1))"  # Subtract 1 for parent directory
else
    echo "❌ Error: Dataset directory does not exist: ${DATASET_PATH}"
    exit 1
fi

# Check category file
if [ -f "${CATEGORY_FILE}" ]; then
    echo "✅ Category file exists"
    CATEGORY_COUNT=$(wc -l < "${CATEGORY_FILE}")
    echo "📊 Test category count: ${CATEGORY_COUNT}"
else
    echo "❌ Error: Category file does not exist: ${CATEGORY_FILE}"
    exit 1
fi

# Check model files
if [ -f "${POINTNET_MODEL}" ]; then
    echo "✅ PointNet registration model exists: ${POINTNET_MODEL}"
    MODEL_SIZE=$(du -h "${POINTNET_MODEL}" | cut -f1)
    echo "📊 Model file size: ${MODEL_SIZE}"
else
    echo "❌ Error: PointNet registration model does not exist: ${POINTNET_MODEL}"
    echo "Please check if the specified model file exists"
    exit 1
fi

if [ -f "${CLASSIFIER_MODEL}" ]; then
    echo "✅ Classifier model exists: ${CLASSIFIER_MODEL}"
    CLASSIFIER_SIZE=$(du -h "${CLASSIFIER_MODEL}" | cut -f1)
    echo "📊 Classifier file size: ${CLASSIFIER_SIZE}"
else
    echo "❌ Error: Classifier model does not exist: ${CLASSIFIER_MODEL}"
    echo "Please check if the specified classifier model file exists"
    exit 1
fi

# Check perturbation directory
if [ -d "${PERTURBATION_DIR}" ]; then
    echo "✅ 扰动目录存在"
    PERT_COUNT=$(find "${PERTURBATION_DIR}" -name "*.csv" | wc -l)
    echo "📊 扰动文件数量: ${PERT_COUNT}"
    if [ ${PERT_COUNT} -eq 0 ]; then
        echo "⚠️  警告: 扰动目录中没有.csv文件"
        echo "将生成大角度扰动文件以进行有效测试..."
        
        # 生成大角度扰动文件 - 修复幅度设置
        mkdir -p "${PERTURBATION_DIR}"
        
        # 使用更大的扰动幅度 (弧度制) - 从中等到大角度
        PERTURBATION_MAGNITUDES=(0.52 0.79 1.05 1.31 1.57 1.83 2.09 2.36 2.62 2.88)
        # 对应角度: 30° 45° 60° 75° 90° 105° 120° 135° 150° 165°
        
        echo "🎯 生成扰动文件，角度范围: 30°-165°"
        
        for mag in "${PERTURBATION_MAGNITUDES[@]}"; do
            # 计算对应的角度(度)
            angle_deg=$(python3 -c "import math; print(f'{math.degrees($mag):.0f}')")
            PERT_FILE="${PERTURBATION_DIR}/pert_${angle_deg}.csv"
            echo "   正在生成扰动文件: ${PERT_FILE} (幅度: ${mag} 弧度 ≈ ${angle_deg}°)"
            
            ${PY3} generate_perturbations.py \
                -o ${PERT_FILE} \
                -i ${DATASET_PATH} \
                -c ${CATEGORY_FILE} \
                --mag ${mag} \
                --dataset-type modelnet
            
            if [ $? -ne 0 ]; then
                echo "   ❌ 生成扰动文件失败: ${PERT_FILE}"
                exit 1
            fi
        done
        
        echo "✅ 成功生成大角度扰动文件"
        PERT_COUNT=$(find "${PERTURBATION_DIR}" -name "*.csv" | wc -l)
        echo "📊 新扰动文件数量: ${PERT_COUNT}"
    else
        echo "📋 现有扰动文件列表:"
        find "${PERTURBATION_DIR}" -name "*.csv" | head -10
        if [ ${PERT_COUNT} -gt 10 ]; then
            echo "   ... (共${PERT_COUNT}个扰动文件)"
        fi
        
        # 检查现有扰动文件的角度范围
        echo "🔍 正在分析现有扰动文件的角度分布..."
        python3 -c "
import pandas as pd
import numpy as np
import os
import glob

pert_dir = '${PERTURBATION_DIR}'
pert_files = glob.glob(os.path.join(pert_dir, '*.csv'))

print('扰动文件角度分析:')
for pert_file in sorted(pert_files)[:5]:  # 只分析前5个文件
    try:
        data = pd.read_csv(pert_file, header=None)
        if len(data.columns) >= 6:
            # 前3列是旋转参数
            rot_params = data.iloc[:, :3]
            max_rotation = rot_params.abs().max().max()
            max_angle_deg = max_rotation * 180 / np.pi
            filename = os.path.basename(pert_file)
            print(f'  {filename}: 最大旋转角度 {max_angle_deg:.1f}°')
        else:
            print(f'  {os.path.basename(pert_file)}: 格式错误')
    except Exception as e:
        print(f'  {os.path.basename(pert_file)}: 读取失败 - {str(e)}')
"
    fi
else
    echo "❌ 错误: 扰动目录不存在: ${PERTURBATION_DIR}"
    echo "正在创建扰动目录并生成大角度扰动文件..."
    mkdir -p "${PERTURBATION_DIR}"
    
    # 使用大角度扰动
    PERTURBATION_MAGNITUDES=(0.52 0.79 1.05 1.31 1.57 1.83 2.09 2.36 2.62 2.88)
    
    for mag in "${PERTURBATION_MAGNITUDES[@]}"; do
        angle_deg=$(python3 -c "import math; print(f'{math.degrees($mag):.0f}')")
        PERT_FILE="${PERTURBATION_DIR}/pert_${angle_deg}.csv"
        echo "   正在生成扰动文件: ${PERT_FILE} (${angle_deg}°)"
        
        ${PY3} generate_perturbations.py \
            -o ${PERT_FILE} \
            -i ${DATASET_PATH} \
            -c ${CATEGORY_FILE} \
            --mag ${mag} \
            --dataset-type modelnet
        
        if [ $? -ne 0 ]; then
            echo "   ❌ 生成扰动文件失败: ${PERT_FILE}"
            exit 1
        fi
    done
    
    echo "✅ 成功创建扰动目录并生成大角度扰动文件"
fi

# GPU memory check
echo ""
echo "========== GPU Status Check =========="
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️  GPU information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️  Unable to get GPU information"
fi

echo ""
echo "========== Starting PointNet Registration Model Test =========="
echo "🚀 About to start testing..."
echo "⏱️  Estimated test time: ~30-60 minutes (depends on number of perturbation files and samples)"
echo ""

# Build visualization parameters
VISUALIZE_PARAMS=""
if [ -n "${VISUALIZE_PERT}" ]; then
    VISUALIZE_PARAMS="${VISUALIZE_PERT} --visualize-samples ${VISUALIZE_SAMPLES}"
fi

# Run test
${PY3} test_pointlk.py \
  -o ${TEST_OUTPUT_PREFIX} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l ${TEST_LOG} \
  --dataset-type modelnet \
  --num-points ${NUM_POINTS} \
  --max-iter ${MAX_ITER} \
  --delta ${DELTA} \
  --device ${DEVICE} \
  --max-samples ${MAX_SAMPLES} \
  --perturbation-dir ${PERTURBATION_DIR} \
  --model-type pointnet \
  --dim-k ${DIM_K} \
  --symfn ${SYMFN} \
  --pretrained ${POINTNET_MODEL} \
  --transfer-from ${CLASSIFIER_MODEL} \
  ${VISUALIZE_PARAMS}

# Check test results
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 PointNet registration model test completed!"
    echo "📁 Test results saved to: ${TEST_RESULTS_DIR}"
    echo "📋 Test log: ${TEST_LOG}"
    
    # Display generated result files - modified to show angle directories and log files
    echo ""
    echo "📊 Generated test result directories:"
    find "${TEST_RESULTS_DIR}" -type d -name "angle_*" | sort | xargs ls -ld 2>/dev/null
    
    echo ""
    echo "📊 Sample result files:"
    # Find and display some result files from angle directories
    find "${TEST_RESULTS_DIR}/angle_"* -name "*.log" -type f 2>/dev/null | sort | head -10 | xargs ls -lh 2>/dev/null
    
    # Statistics
    echo ""
    echo "📈 Test statistics:"
    # Count all log files (both in angle directories and main directory)
    RESULT_FILES=$(find "${TEST_RESULTS_DIR}" -name "*.log" -type f | wc -l)
    echo "   - Total result file count: ${RESULT_FILES}"
    
    # Count angle directories
    ANGLE_DIRS=$(find "${TEST_RESULTS_DIR}" -type d -name "angle_*" | wc -l)
    echo "   - Angle directories: ${ANGLE_DIRS}"
    
    if [ ${RESULT_FILES} -gt 0 ]; then
        # Display statistics from a sample result file
        SAMPLE_FILE=$(find "${TEST_RESULTS_DIR}" -name "results_*.log" -type f | head -1)
        if [ -f "${SAMPLE_FILE}" ]; then
            echo "   - Sample result file: ${SAMPLE_FILE}"
            echo "   - Result file statistics preview:"
            # Display statistics section (final statistical results)
            grep "# Average.*error:" "${SAMPLE_FILE}" | head -3
        fi
    fi
    
else
    echo ""
    echo "❌ PointNet registration model test failed!"
    echo "Please check error log: ${TEST_LOG}"
    
    # Display last few lines of error information
    if [ -f "${TEST_LOG}" ]; then
        echo ""
        echo "📋 Latest error information:"
        tail -10 "${TEST_LOG}"
    fi
    
    exit 1
fi

# Save test configuration information
CONFIG_FILE="${TEST_RESULTS_DIR}/pointnet_test_${DATE_TAG}_config.txt"
echo "🧠 PointNet Registration Model Test Configuration" > ${CONFIG_FILE}
echo "=====================================" >> ${CONFIG_FILE}
echo "Test completion time: $(date)" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "🔧 Model configuration:" >> ${CONFIG_FILE}
echo "Model type: PointNet registration model" >> ${CONFIG_FILE}
echo "Feature dimension: ${DIM_K}" >> ${CONFIG_FILE}
echo "Aggregation function: ${SYMFN}" >> ${CONFIG_FILE}
echo "LK max iterations: ${MAX_ITER}" >> ${CONFIG_FILE}
echo "LK step size: ${DELTA}" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "📊 Test configuration:" >> ${CONFIG_FILE}
echo "Dataset path: ${DATASET_PATH}" >> ${CONFIG_FILE}
echo "Category file: ${CATEGORY_FILE}" >> ${CONFIG_FILE}
echo "Number of points: ${NUM_POINTS}" >> ${CONFIG_FILE}
echo "Maximum test samples: ${MAX_SAMPLES}" >> ${CONFIG_FILE}
echo "Device: ${DEVICE}" >> ${CONFIG_FILE}
echo "Perturbation directory: ${PERTURBATION_DIR}" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "📁 Model files:" >> ${CONFIG_FILE}
echo "Registration model: ${POINTNET_MODEL}" >> ${CONFIG_FILE}
echo "Classifier model: ${CLASSIFIER_MODEL}" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "📁 Output files:" >> ${CONFIG_FILE}
echo "Test result prefix: ${TEST_OUTPUT_PREFIX}" >> ${CONFIG_FILE}
echo "Test log: ${TEST_LOG}" >> ${CONFIG_FILE}

echo ""
echo "💾 Test configuration information saved to: ${CONFIG_FILE}"

echo ""
echo "🎯 Test completion summary:"
echo "📂 Result directory: ${TEST_RESULTS_DIR}"
echo "📄 Configuration file: ${CONFIG_FILE}"
echo "🎯 Registration model: ${POINTNET_MODEL}"
echo "🎯 Classifier model: ${CLASSIFIER_MODEL}"
echo "📋 Test log: ${TEST_LOG}"
echo "⏰ Completion time: $(date)"

echo ""
echo "🎉🎉🎉 PointNet registration model test all completed! 🎉🎉🎉"
