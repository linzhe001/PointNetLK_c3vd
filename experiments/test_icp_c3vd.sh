#!/usr/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=36000  # 1小时测试时间
#$ -l gpu=true

#$ -pe gpu 1
#$ -N ljiang_test_icp_c3vd
#$ -o /SAN/medic/MRpcr/logs/f_test_icp_c3vd_output.log
#$ -e /SAN/medic/MRpcr/logs/f_test_icp_c3vd_error.log
#$ -wd /SAN/medic/MRpcr

# 进入实验脚本所在目录
cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments

echo "当前工作目录: $(pwd)"

# 激活Conda环境
echo "激活Conda环境..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

# 创建结果目录
mkdir -p /SAN/medic/MRpcr/results/icp_c3vd/test_results
mkdir -p /SAN/medic/MRpcr/results/icp_c3vd/test_results/gt

# Python命令
PY3="nice -n 10 python"

# 配置参数
DATASET_PATH="/SAN/medic/MRpcr/C3VD_datasets"
CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/c3vd.txt"
PERTURBATION_DIR="/SAN/medic/MRpcr/PointNetLK_c3vd/gt"
OUTFILE="/SAN/medic/MRpcr/results/icp_c3vd/test_results/icp_results.csv"
NUM_POINTS=1024             # 点云点数
DEVICE="cpu"              # 使用CPU进行ICP
MAX_ITER=20                 # ICP最大迭代次数
MAX_SAMPLES=1000            # 最大测试样本数
PAIR_MODE="one_to_one"    # 配对模式
WORKERS=4                   # 数据加载线程数

# 打印配置信息
echo "========== ICP 测试配置 =========="
echo "数据集路径: ${DATASET_PATH}"
echo "类别文件: ${CATEGORY_FILE}"
echo "扰动文件夹: ${PERTURBATION_DIR}"
echo "输出文件: ${OUTFILE}"
echo "点云数量: ${NUM_POINTS}"
echo "最大迭代次数: ${MAX_ITER}"
echo "最大测试样本数: ${MAX_SAMPLES}"
echo "配对模式: ${PAIR_MODE}"
echo "使用设备: ${DEVICE}"
echo "数据加载线程: ${WORKERS}"
echo "================================"

# 运行ICP测试
${PY3} test_icp.py \
  -o ${OUTFILE} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  --dataset-type c3vd \
  --perturbation-dir ${PERTURBATION_DIR} \
  --num-points ${NUM_POINTS} \
  --max-iter ${MAX_ITER} \
  --max-samples ${MAX_SAMPLES} \
  --pair-mode ${PAIR_MODE} \
  -j ${WORKERS} \
  --device ${DEVICE}

# =============================================================================
# 第二轮测试：GT姿态文件
# =============================================================================
GT_POSES_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/gt_poses.csv"
OUTFILE_GT="/SAN/medic/MRpcr/results/icp_c3vd/test_results/gt/icp_results_gt.csv"
echo "========== 第二轮测试：GT姿态文件 =========="
${PY3} test_icp.py \
  -o ${OUTFILE_GT} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  --dataset-type c3vd \
  --perturbations ${GT_POSES_FILE} \
  --num-points ${NUM_POINTS} \
  --max-iter ${MAX_ITER} \
  --max-samples ${MAX_SAMPLES} \
  --pair-mode ${PAIR_MODE} \
  -j ${WORKERS} \
  --device ${DEVICE} 