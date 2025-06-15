#!/usr/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G            
#$ -l h_rt=3600  # 1 hour test time
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -pe gpu 1
#$ -N ljiang_test_cformer_c3vd
#$ -o /SAN/medic/MRpcr/logs/f_test_cformer_c3vd_output.log
#$ -e /SAN/medic/MRpcr/logs/f_test_cformer_c3vd_error.log
#$ -wd /SAN/medic/MRpcr

cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
# Set working directory
echo "Current working directory: $(pwd)"

# Activate Conda environment
echo "Activating Conda environment..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

# Create results and log directories
mkdir -p /SAN/medic/MRpcr/results/cformer_c3vd/test_results

# Python command
PY3="nice -n 10 python"

# Set variables
DATASET_PATH="/SAN/medic/MRpcr/C3VD_datasets"
CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/c3vd.txt"
NUM_POINTS=1024
DEVICE="cuda:0"
DATE_TAG=$(date +"%m%d")

# CFormer model configuration (keep consistent with training)
DIM_K=1024                    # Feature dimension
NUM_PROXY_POINTS=8            # Number of proxy points
NUM_BLOCKS=2                  # Number of CFormer blocks
SYMFN="max"                   # Aggregation function: max, avg, or cd_pool
MAX_ITER=20                   # LK maximum iterations
DELTA=1.0e-2                  # LK step size

# C3VD pairing mode settings (keep consistent with training)
PAIR_MODE="one_to_one"  # Use point-to-point pairing mode
REFERENCE_NAME=""  # Clear reference point cloud name
REFERENCE_PARAM=""
if [ "${PAIR_MODE}" = "scene_reference" ] && [ -n "${REFERENCE_NAME}" ]; then
  REFERENCE_PARAM="--reference-name ${REFERENCE_NAME}"
fi

# Joint normalization settings
USE_JOINT_NORM="--use-joint-normalization"

# Maximum test samples
MAX_SAMPLES=2000

# Visualization settings (optional)
VISUALIZE_PERT="" # If visualization needed, set to "--visualize-pert pert_010.csv pert_020.csv"
VISUALIZE_SAMPLES=3

# Model path (needs to be adjusted according to actual training results)
CFORMER_MODEL_PREFIX="/SAN/medic/MRpcr/results/cformer_c3vd/cformer_pointlk_${DATE_TAG}"
CFORMER_MODEL="${CFORMER_MODEL_PREFIX}_model_best.pth"
# Note: During testing, only use complete registration model weights, no separate feature weights needed

# If today's model not found, try to find the latest model
if [ ! -f "${CFORMER_MODEL}" ]; then
    echo "⚠️  Today's model file not found: ${CFORMER_MODEL}"
    echo "🔍 Searching for latest cformer model..."
    
    # Search for latest cformer pointlk model
    LATEST_MODEL=$(find /SAN/medic/MRpcr/results/cformer_c3vd/ -name "cformer_pointlk_*_model_best.pth" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -n "${LATEST_MODEL}" ] && [ -f "${LATEST_MODEL}" ]; then
        CFORMER_MODEL="${LATEST_MODEL}"
        # Extract model prefix
        CFORMER_MODEL_PREFIX=$(echo "${CFORMER_MODEL}" | sed 's/_model_best\.pth$//')
        echo "✅ Found latest model: ${CFORMER_MODEL}"
    else
        echo "❌ Error: No cformer model files found!"
        echo "Please ensure train_cformer_c3vd.sh has been run and successfully trained the model"
        exit 1
    fi
fi

# Perturbation file directory
PERTURBATION_DIR="/SAN/medic/MRpcr/PointNetLK_c3vd/gt"

# Test results output directory
TEST_RESULTS_DIR="/SAN/medic/MRpcr/results/cformer_c3vd/test_results"
# Base name for output files - results will be stored in subdirectories based on angle
TEST_OUTPUT_PREFIX="${TEST_RESULTS_DIR}/results"
TEST_LOG="${TEST_RESULTS_DIR}/test_log_${DATE_TAG}.log"

# Print configuration information
echo "========== CFormer Registration Model Test Configuration =========="
echo "🧠 Model type: CFormer registration model"
echo "📂 Dataset path: ${DATASET_PATH}"
echo "📄 Category file: ${CATEGORY_FILE}"
echo "🔗 Pairing mode: ${PAIR_MODE}"
echo "🎯 Reference point cloud: ${REFERENCE_NAME:-'Auto-select'}"
echo "🎲 Number of points: ${NUM_POINTS}"
echo "🖥️  Device: ${DEVICE}"
echo "📊 Maximum test samples: ${MAX_SAMPLES}"
echo ""
echo "🔧 CFormer parameters:"
echo "   - Feature dimension: ${DIM_K}"
echo "   - Number of proxy points: ${NUM_PROXY_POINTS}"
echo "   - Number of CFormer blocks: ${NUM_BLOCKS}"
echo "   - Aggregation function: ${SYMFN}"
echo "   - LK max iterations: ${MAX_ITER}"
echo "   - LK step size: ${DELTA}"
echo ""
echo "📁 Model files:"
echo "   - Registration model: ${CFORMER_MODEL}"
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
    TARGET_COUNT=$(find "${DATASET_PATH}/visible_point_cloud_ply_depth" -name "*.ply" 2>/dev/null | wc -l)
    echo "📊 Source point cloud file count: ${SOURCE_COUNT}"
    echo "📊 Target point cloud file count: ${TARGET_COUNT}"
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
if [ -f "${CFORMER_MODEL}" ]; then
    echo "✅ CFormer registration model exists: ${CFORMER_MODEL}"
    MODEL_SIZE=$(du -h "${CFORMER_MODEL}" | cut -f1)
    echo "📊 Model file size: ${MODEL_SIZE}"
else
    echo "❌ Error: CFormer registration model does not exist: ${CFORMER_MODEL}"
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
echo "========== Starting CFormer Registration Model Test =========="
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
  --model-type cformer \
  --dim-k ${DIM_K} \
  --num-proxy-points ${NUM_PROXY_POINTS} \
  --num-blocks ${NUM_BLOCKS} \
  --symfn ${SYMFN} \
  --pretrained ${CFORMER_MODEL} \
  ${VISUALIZE_PARAMS}

# Check test results
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 CFormer registration model test completed!"
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
    echo "❌ CFormer registration model test failed!"
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
CONFIG_FILE="${TEST_RESULTS_DIR}/cformer_test_${DATE_TAG}_config.txt"
echo "🧠 CFormer Registration Model Test Configuration" > ${CONFIG_FILE}
echo "=====================================" >> ${CONFIG_FILE}
echo "Test completion time: $(date)" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "🔧 Model configuration:" >> ${CONFIG_FILE}
echo "Model type: CFormer registration model" >> ${CONFIG_FILE}
echo "Feature dimension: ${DIM_K}" >> ${CONFIG_FILE}
echo "Number of proxy points: ${NUM_PROXY_POINTS}" >> ${CONFIG_FILE}
echo "Number of CFormer blocks: ${NUM_BLOCKS}" >> ${CONFIG_FILE}
echo "Aggregation function: ${SYMFN}" >> ${CONFIG_FILE}
echo "LK max iterations: ${MAX_ITER}" >> ${CONFIG_FILE}
echo "LK step size: ${DELTA}" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "📊 Test configuration:" >> ${CONFIG_FILE}
echo "Dataset path: ${DATASET_PATH}" >> ${CONFIG_FILE}
echo "Category file: ${CATEGORY_FILE}" >> ${CONFIG_FILE}
echo "Pairing mode: ${PAIR_MODE}" >> ${CONFIG_FILE}
if [ "${PAIR_MODE}" = "scene_reference" ] && [ -n "${REFERENCE_NAME}" ]; then
  echo "Reference point cloud name: ${REFERENCE_NAME}" >> ${CONFIG_FILE}
fi
echo "Number of points: ${NUM_POINTS}" >> ${CONFIG_FILE}
echo "Maximum test samples: ${MAX_SAMPLES}" >> ${CONFIG_FILE}
echo "Device: ${DEVICE}" >> ${CONFIG_FILE}
echo "Perturbation directory: ${PERTURBATION_DIR}" >> ${CONFIG_FILE}
echo "Joint normalization: Yes" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "📁 Model files:" >> ${CONFIG_FILE}
echo "Registration model: ${CFORMER_MODEL}" >> ${CONFIG_FILE}
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
echo "🎯 Registration model: ${CFORMER_MODEL}"
echo "📋 Test log: ${TEST_LOG}"
echo "⏰ Completion time: $(date)"

echo ""
echo "🎉🎉🎉 CFormer registration model test all completed! 🎉🎉🎉" 