#!/usr/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G            
#$ -l h_rt=3600  # 1 hour test time
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -pe gpu 1
#$ -N ljiang_test_attention_c3vd_ref
#$ -o /SAN/medic/MRpcr/logs/f_test_attention_c3vd_ref_output.log
#$ -e /SAN/medic/MRpcr/logs/f_test_attention_c3vd_ref_error.log
#$ -wd /SAN/medic/MRpcr

cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
# Set working directory
echo "Current working directory: $(pwd)"

# Activate Conda environment
echo "Activating Conda environment..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

# Create results and log directories
mkdir -p /SAN/medic/MRpcr/results/attention_c3vd/test_results_ref

# Python command
PY3="nice -n 10 python"

# Set variables
DATASET_PATH="/SAN/medic/MRpcr/C3VD_datasets"
CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/c3vd.txt"
NUM_POINTS=1024
DEVICE="cuda:0"
DATE_TAG=$(date +"%m%d")

# AttentionNet model configuration (keep consistent with training)
DIM_K=1024                    # 特征维度
NUM_ATTENTION_BLOCKS=2        # 修改：从3改为2，与训练脚本保持一致
NUM_HEADS=4                   # 修改：从8改为4，与训练脚本保持一致
SYMFN="max"                   # 聚合函数: max, avg, or attention
MAX_ITER=20                   # LK最大迭代次数
DELTA=1.0e-2                  # LK步长

# C3VD pairing mode settings - 使用scene_reference模式
PAIR_MODE="scene_reference"  # 使用场景参考配对模式
REFERENCE_NAME="coverage_mesh.ply"  # 明确指定参考点云文件名
REFERENCE_PARAM="--reference-name ${REFERENCE_NAME}"

# Joint normalization settings
USE_JOINT_NORM="--use-joint-normalization"

# Maximum test samples
MAX_SAMPLES=2000

# Visualization settings (optional)
VISUALIZE_PERT="" # 如需可视化，设置为"--visualize-pert pert_010.csv pert_020.csv"
VISUALIZE_SAMPLES=3

# Model path (needs to be adjusted according to actual training results)
ATTENTION_MODEL_PREFIX="/SAN/medic/MRpcr/results/attention_c3vd/attention_pointlk_${DATE_TAG}"
ATTENTION_MODEL="${ATTENTION_MODEL_PREFIX}_model_best.pth"
# Note: During testing, only use complete registration model weights, no separate feature weights needed

# If today's model not found, try to find the latest model
if [ ! -f "${ATTENTION_MODEL}" ]; then
    echo "⚠️  Today's model file not found: ${ATTENTION_MODEL}"
    echo "🔍 Searching for latest attention model..."
    
    # Search for latest attention pointlk model
    LATEST_MODEL=$(find /SAN/medic/MRpcr/results/attention_c3vd/ -name "attention_pointlk_*_model_best.pth" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -n "${LATEST_MODEL}" ] && [ -f "${LATEST_MODEL}" ]; then
        ATTENTION_MODEL="${LATEST_MODEL}"
        # Extract model prefix
        ATTENTION_MODEL_PREFIX=$(echo "${ATTENTION_MODEL}" | sed 's/_model_best\.pth$//')
        echo "✅ Found latest model: ${ATTENTION_MODEL}"
    else
        echo "❌ Error: No attention model files found!"
        echo "Please ensure train_attention_c3vd.sh has been run and successfully trained the model"
        exit 1
    fi
fi

# Perturbation file directory
PERTURBATION_DIR="/SAN/medic/MRpcr/PointNetLK_c3vd/gt"

# 修改输出目录为新目录
# Test results output directory
TEST_RESULTS_DIR="/SAN/medic/MRpcr/results/attention_c3vd/test_results_ref"
# Base name for output files - results will be stored in subdirectories based on angle
TEST_OUTPUT_PREFIX="${TEST_RESULTS_DIR}/results"
TEST_LOG="${TEST_RESULTS_DIR}/test_log_ref_${DATE_TAG}.log"

# Print configuration information
echo "========== AttentionNet Registration Model Test Configuration (Reference Mode) =========="
echo "🧠 Model type: AttentionNet registration model"
echo "📂 Dataset path: ${DATASET_PATH}"
echo "📄 Category file: ${CATEGORY_FILE}"
echo "🔗 Pairing mode: ${PAIR_MODE} (使用完整参考点云)"
echo "🎯 Reference point cloud: ${REFERENCE_NAME:-'Auto-select'}"
echo "🎲 Number of points: ${NUM_POINTS}"
echo "🖥️  Device: ${DEVICE}"
echo "📊 Maximum test samples: ${MAX_SAMPLES}"
echo ""
echo "🔧 AttentionNet parameters:"
echo "   - Feature dimension: ${DIM_K}"
echo "   - Number of Attention blocks: ${NUM_ATTENTION_BLOCKS}"
echo "   - Multi-head attention heads: ${NUM_HEADS}"
echo "   - Aggregation function: ${SYMFN}"
echo "   - LK max iterations: ${MAX_ITER}"
echo "   - LK step size: ${DELTA}"
echo ""
echo "📁 Model files:"
echo "   - Registration model: ${ATTENTION_MODEL}"
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
    SOURCE_COUNT=$(find "${DATASET_PATH}/C3VD_ply_source" -name "*.ply" 2>/dev/null | wc -l)
    TARGET_COUNT=$(find "${DATASET_PATH}/C3VD_ref" -name "${REFERENCE_NAME}" 2>/dev/null | wc -l)
    echo "📊 Source point cloud file count: ${SOURCE_COUNT}"
    echo "📊 Reference point cloud file count: ${TARGET_COUNT}"
else
    echo "❌ Error: Dataset directory does not exist: ${DATASET_PATH}"
    exit 1
fi

# Check category file
if [ -f "${CATEGORY_FILE}" ]; then
    echo "✅ Category file exists"
    CATEGORY_COUNT=$(wc -l < "${CATEGORY_FILE}")
    echo "📊 Category count: ${CATEGORY_COUNT}"
else
    echo "❌ Error: Category file does not exist: ${CATEGORY_FILE}"
    exit 1
fi

# Check model file
if [ -f "${ATTENTION_MODEL}" ]; then
    echo "✅ AttentionNet registration model exists: ${ATTENTION_MODEL}"
    MODEL_SIZE=$(du -h "${ATTENTION_MODEL}" | cut -f1)
    echo "📊 Model file size: ${MODEL_SIZE}"
else
    echo "❌ Error: AttentionNet registration model does not exist: ${ATTENTION_MODEL}"
    exit 1
fi

# Check perturbation directory
if [ -d "${PERTURBATION_DIR}" ]; then
    echo "✅ Perturbation directory exists"
    PERT_COUNT=$(find "${PERTURBATION_DIR}" -name "*.csv" | wc -l)
    echo "📊 Perturbation file count: ${PERT_COUNT}"
    if [ ${PERT_COUNT} -eq 0 ]; then
        echo "⚠️  Warning: No .csv files in perturbation directory"
        echo "Will try to generate default perturbation files..."
        # Code to generate perturbation files can be added here
    else
        echo "📋 Perturbation file list:"
        find "${PERTURBATION_DIR}" -name "*.csv" | head -5
        if [ ${PERT_COUNT} -gt 5 ]; then
            echo "   ... (total ${PERT_COUNT} perturbation files)"
        fi
    fi
else
    echo "❌ Error: Perturbation directory does not exist: ${PERTURBATION_DIR}"
    exit 1
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
echo "========== Starting AttentionNet Registration Model Test (Reference Mode) =========="
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
  --dataset-type c3vd \
  --num-points ${NUM_POINTS} \
  --max-iter ${MAX_ITER} \
  --delta ${DELTA} \
  --device ${DEVICE} \
  --max-samples ${MAX_SAMPLES} \
  --pair-mode ${PAIR_MODE} \
  ${REFERENCE_PARAM} \
  ${USE_JOINT_NORM} \
  --perturbation-dir ${PERTURBATION_DIR} \
  --model-type attention \
  --dim-k ${DIM_K} \
  --num-attention-blocks ${NUM_ATTENTION_BLOCKS} \
  --num-heads ${NUM_HEADS} \
  --symfn ${SYMFN} \
  --pretrained ${ATTENTION_MODEL} \
  ${VISUALIZE_PARAMS}

# Check test results
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 AttentionNet registration model test (Reference Mode) completed!"
    echo "📁 Test results saved to: ${TEST_RESULTS_DIR}"
    echo "📋 Test log: ${TEST_LOG}"
    
    # Display generated result files - modified to show angle directories and log files
    echo ""
    echo "📊 Generated test result directories:"
    find "${TEST_RESULTS_DIR}" -type d -name "angle_*" | sort | xargs ls -ld
    
    echo ""
    echo "📊 Sample result files:"
    # Find and display some result files from angle directories
    find "${TEST_RESULTS_DIR}/angle_"* -name "*.log" -type f | sort | head -10 | xargs ls -lh
    
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
    echo "❌ AttentionNet registration model test failed!"
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
CONFIG_FILE="${TEST_RESULTS_DIR}/attention_test_ref_${DATE_TAG}_config.txt"
echo "🧠 AttentionNet Registration Model Test Configuration (Reference Mode)" > ${CONFIG_FILE}
echo "=====================================" >> ${CONFIG_FILE}
echo "Test completion time: $(date)" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "🔧 Model configuration:" >> ${CONFIG_FILE}
echo "Model type: AttentionNet registration model" >> ${CONFIG_FILE}
echo "Feature dimension: ${DIM_K}" >> ${CONFIG_FILE}
echo "Number of Attention blocks: ${NUM_ATTENTION_BLOCKS}" >> ${CONFIG_FILE}
echo "Multi-head attention heads: ${NUM_HEADS}" >> ${CONFIG_FILE}
echo "Aggregation function: ${SYMFN}" >> ${CONFIG_FILE}
echo "LK max iterations: ${MAX_ITER}" >> ${CONFIG_FILE}
echo "LK step size: ${DELTA}" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "📊 Test configuration:" >> ${CONFIG_FILE}
echo "Dataset path: ${DATASET_PATH}" >> ${CONFIG_FILE}
echo "Category file: ${CATEGORY_FILE}" >> ${CONFIG_FILE}
echo "Pairing mode: ${PAIR_MODE} (使用完整参考点云)" >> ${CONFIG_FILE}
if [ "${PAIR_MODE}" = "scene_reference" ] && [ -n "${REFERENCE_NAME}" ]; then
  echo "Reference point cloud name: ${REFERENCE_NAME}" >> ${CONFIG_FILE}
else
  echo "Reference point cloud name: Auto-select (first point cloud in scene)" >> ${CONFIG_FILE}
fi
echo "Number of points: ${NUM_POINTS}" >> ${CONFIG_FILE}
echo "Maximum test samples: ${MAX_SAMPLES}" >> ${CONFIG_FILE}
echo "Device: ${DEVICE}" >> ${CONFIG_FILE}
echo "Perturbation directory: ${PERTURBATION_DIR}" >> ${CONFIG_FILE}
echo "Joint normalization: Yes" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "📁 Model files:" >> ${CONFIG_FILE}
echo "Registration model: ${ATTENTION_MODEL}" >> ${CONFIG_FILE}
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
echo "🎯 Registration model: ${ATTENTION_MODEL}"
echo "📋 Test log: ${TEST_LOG}"
echo "⏰ Completion time: $(date)"

echo ""
echo "🎉🎉🎉 AttentionNet registration model test (Reference Mode) all completed! 🎉🎉🎉" 