#!/usr/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G            
#$ -l h_rt=36000  # 1小时测试时间
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -pe gpu 1
#$ -N ljiang_test_attention_modelnet
#$ -o /SAN/medic/MRpcr/logs/f_test_attention_modelnet_output.log
#$ -e /SAN/medic/MRpcr/logs/f_test_attention_modelnet_error.log
#$ -wd /SAN/medic/MRpcr

cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
# 设置工作目录
echo "当前工作目录: $(pwd)"

# 激活Conda环境
echo "激活Conda环境..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

# 创建结果和日志目录
mkdir -p /SAN/medic/MRpcr/results/attention_modelnet/test_results
mkdir -p /SAN/medic/MRpcr/results/attention_modelnet/test_results/gt

# Python命令
PY3="nice -n 10 python"

# 设置变量
DATASET_PATH="/SAN/medic/MRpcr/ModelNet40"
CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/modelnet40_half2.txt"
NUM_POINTS=1024
DEVICE="cuda:0"
DATE_TAG=$(date +"%m%d")

# Attention 模型配置（与训练保持一致）
DIM_K=1024                   
NUM_ATTENTION_BLOCKS=2       
NUM_HEADS=4                  
SYMFN="max"                
MAX_ITER=20                  
DELTA=1.0e-2                 

# 最大测试样本数
MAX_SAMPLES_ROUND1=1000      
MAX_SAMPLES_ROUND2=0         

# 可视化设置（可选）
VISUALIZE_PERT=""         
VISUALIZE_SAMPLES=3          

# 模型路径（自动查找最新）
ATT_MODEL_PREFIX="/SAN/medic/MRpcr/results/attention_modelnet/attention_pointlk_${DATE_TAG}"
ATT_MODEL="${ATT_MODEL_PREFIX}_model_best.pth"
if [ ! -f "${ATT_MODEL}" ]; then
    echo "⚠️ 未找到今日Attention模型: ${ATT_MODEL}"
    LATEST_MODEL=$(find /SAN/medic/MRpcr/results/attention_modelnet/ -name "attention_pointlk_*_model_best.pth" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -f2- -d' ')
    if [ -n "${LATEST_MODEL}" ] && [ -f "${LATEST_MODEL}" ]; then
        ATT_MODEL="${LATEST_MODEL}"
        echo "✅ 找到最新Attention模型: ${ATT_MODEL}"
    else
        echo "❌ 错误: 未找到Attention模型文件!"
        exit 1
    fi
else
    echo "✅ 使用指定Attention模型: ${ATT_MODEL}"
fi

# 扰动文件配置
PERTURBATION_DIR="/SAN/medic/MRpcr/PointNetLK_c3vd/gt"
GT_POSES_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/gt_poses.csv"

# 测试结果输出目录
TEST_RESULTS_DIR="/SAN/medic/MRpcr/results/attention_modelnet/test_results"
TEST_LOG="${TEST_RESULTS_DIR}/test_log_${DATE_TAG}.log"

# 第一轮测试：角度扰动文件
echo "========== 第一轮测试：角度扰动文件 =========="
${PY3} test_pointlk.py \
  -o ${TEST_RESULTS_DIR}/results \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l ${TEST_LOG} \
  --dataset-type modelnet \
  --num-points ${NUM_POINTS} \
  --max-iter ${MAX_ITER} \
  --delta ${DELTA} \
  --device ${DEVICE} \
  --max-samples ${MAX_SAMPLES_ROUND1} \
  --perturbation-dir ${PERTURBATION_DIR} \
  --no-voxelization \
  --model-type attention \
  --dim-k ${DIM_K} \
  --num-attention-blocks ${NUM_ATTENTION_BLOCKS} \
  --num-heads ${NUM_HEADS} \
  --symfn ${SYMFN} \
  --pretrained ${ATT_MODEL} \
  ${VISUALIZE_PERT}
if [ $? -ne 0 ]; then
    echo "❌ 第一轮测试失败"
    exit 1
else
    echo "✅ 第一轮测试完成"
fi

# 第二轮测试：GT姿态文件
echo "========== 第二轮测试：GT姿态文件 =========="
ROUND2_DIR="${TEST_RESULTS_DIR}/gt"
mkdir -p ${ROUND2_DIR}
TEST_LOG2="${ROUND2_DIR}/test_log_gt_${DATE_TAG}.log"
${PY3} test_pointlk.py \
  -o ${ROUND2_DIR} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l ${TEST_LOG2} \
  --dataset-type modelnet \
  --num-points ${NUM_POINTS} \
  --max-iter ${MAX_ITER} \
  --delta ${DELTA} \
  --device ${DEVICE} \
  --perturbation-file ${GT_POSES_FILE} \
  --model-type attention \
  --dim-k ${DIM_K} \
  --num-attention-blocks ${NUM_ATTENTION_BLOCKS} \
  --num-heads ${NUM_HEADS} \
  --symfn ${SYMFN} \
  --pretrained ${ATT_MODEL} \
  ${VISUALIZE_PERT}
if [ $? -ne 0 ]; then
    echo "❌ 第二轮测试失败"
    exit 1
else
    echo "✅ 第二轮测试完成"
fi

echo "🎉 Attention 模型测试全部完成！" 