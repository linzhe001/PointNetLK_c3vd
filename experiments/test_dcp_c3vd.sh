#!/usr/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G            
#$ -l h_rt=36000  # 1小时测试时间
#$ -l gpu=true

#$ -pe gpu 1
#$ -N test_dcp_c3vd
#$ -o /SAN/medic/MRpcr/logs/test_dcp_c3vd_output.log
#$ -e /SAN/medic/MRpcr/logs/test_dcp_c3vd_error.log
#$ -wd /SAN/medic/MRpcr

cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
# 设置工作目录
echo "当前工作目录: $(pwd)"

# 激活Conda环境
echo "激活Conda环境..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

# Python命令
PY3="nice -n 10 python"

# --- 配置 ---
DATASET_PATH="/SAN/medic/MRpcr/C3VD_datasets"
PERTURBATION_DIR="/SAN/medic/MRpcr/PointNetLK_c3vd/gt"
# 根据用户反馈修正 gt_poses.csv 的路径
# Correct the path for gt_poses.csv based on user feedback
GT_POSES_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/gt_poses.csv"
TEST_RESULTS_DIR="/SAN/medic/MRpcr/results/dcp_c3vd/test_results"
MODEL_DIR="/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/checkpoints/dcp_c3vd"
DCP_MODEL="${MODEL_DIR}/best.pth"

# 样本数量 (与 pointlk 一致)
MAX_SAMPLES_ROUND1=1000
MAX_SAMPLES_ROUND2=0 # 0 表示无限制

# DCP 模型配置 (应与训练时保持一致)
EMB_DIMS=512
EMB_NN=dgcnn
POINTER=transformer
HEAD=mlp
# 在训练DCP时通常不使用cycle loss，这里设为false
CYCLE=false 

# 其他测试参数
NUM_POINTS=1024
DEVICE="cuda:0"
PAIR_MODE="one_to_one"
DATE_TAG=$(date +"%Y%m%d_%H%M%S")

# --- 准备工作 ---
# 为第二轮测试创建gt子目录
mkdir -p "${TEST_RESULTS_DIR}/gt"
echo "测试结果将保存到: ${TEST_RESULTS_DIR}"

# --- 通用参数 ---
COMMON_PARAMS=(
  "--model-path" "${DCP_MODEL}"
  "-i" "${DATASET_PATH}"
  "--dataset-type" "c3vd"
  "--pair-mode" "${PAIR_MODE}"
  "--num-points" "${NUM_POINTS}"
  "--device" "${DEVICE}"
  "--emb-dims" "${EMB_DIMS}"
  "--emb-nn" "${EMB_NN}"
  "--pointer" "${POINTER}"
  "--head" "${HEAD}"
)
if [ "$CYCLE" = true ]; then
    COMMON_PARAMS+=(--cycle)
fi

echo "========== DCP C3VD 双轮测试 =========="
echo "模型: ${DCP_MODEL}"
echo "数据集: ${DATASET_PATH}"
echo "扰动目录: ${PERTURBATION_DIR}"
echo "GT姿态文件: ${GT_POSES_FILE}"
echo "======================================="

# # --- 第一轮测试：处理gt文件夹中的所有扰动文件 ---
# echo "========== 第一轮测试：角度扰动文件 =========="
# # test_dcp.py会根据-o参数的父目录来创建angle_xxx子目录
# TEST_OUTPUT_PREFIX_ROUND1="${TEST_RESULTS_DIR}/results_round1.csv"
# TEST_LOG_ROUND1="${TEST_RESULTS_DIR}/test_log_round1_${DATE_TAG}.log"

# echo "🚀 开始第一轮测试... (最大样本数: ${MAX_SAMPLES_ROUND1})"
# ${PY3} test_dcp.py \
#   "${COMMON_PARAMS[@]}" \
#   -o "${TEST_OUTPUT_PREFIX_ROUND1}" \
#   -l "${TEST_LOG_ROUND1}" \
#   --max-samples "${MAX_SAMPLES_ROUND1}" \
#   --perturbation-dir "${PERTURBATION_DIR}"

# if [ $? -eq 0 ]; then
#     echo "✅ 第一轮测试（角度扰动）完成!"
# else
#     echo "❌ 第一轮测试（角度扰动）失败!"
#     echo "请检查日志: ${TEST_LOG_ROUND1}"
#     exit 1
# fi

# --- 第二轮测试：处理单独的gt_poses.csv文件 ---
echo ""
echo "========== 第二轮测试：GT姿态文件 =========="
# 将第二轮的结果直接输出到gt子目录
TEST_OUTPUT_PREFIX_ROUND2="${TEST_RESULTS_DIR}/gt/results_round2.csv"
TEST_LOG_ROUND2="${TEST_RESULTS_DIR}/gt/test_log_round2_${DATE_TAG}.log"

echo "🚀 开始第二轮测试... (最大样本数: ${MAX_SAMPLES_ROUND2:-'无限制'})"
${PY3} test_dcp.py \
  "${COMMON_PARAMS[@]}" \
  -o "${TEST_OUTPUT_PREFIX_ROUND2}" \
  -l "${TEST_LOG_ROUND2}" \
  --max-samples "${MAX_SAMPLES_ROUND2}" \
  --perturbation-file "${GT_POSES_FILE}"

if [ $? -eq 0 ]; then
    echo "✅ 第二轮测试（GT姿态）完成!"
else
    echo "❌ 第二轮测试（GT姿态）失败!"
    echo "请检查日志: ${TEST_LOG_ROUND2}"
    exit 1
fi

echo ""
echo "🎉🎉🎉 DCP C3VD 双轮测试全部完成! 🎉🎉🎉"
echo "📂 结果目录: ${TEST_RESULTS_DIR}" 