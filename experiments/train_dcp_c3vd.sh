#!/usr/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=174000  # 最大运行时长4小时
#$ -l gpu=true

#$ -pe gpu 1
#$ -N train_dcp_c3vd
#$ -o /SAN/medic/MRpcr/logs/train_dcp_c3vd_output.log
#$ -e /SAN/medic/MRpcr/logs/train_dcp_c3vd_error.log
#$ -wd /SAN/medic/MRpcr
cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
echo "Current working directory: $(pwd)"

# Activate Conda environment
echo "Activating Conda environment..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

# Set Python command to use Conda environment's python
PYTHON="nice -n 10 python"
# 用 DCP 在 C3VD 数据集上训练

# 数据集路径
DATASET_PATH=/SAN/medic/MRpcr/C3VD_datasets
# 模型保存目录
SAVE_DIR=checkpoints/dcp_c3vd
# 超参数设置
BATCH_SIZE=8
EPOCHS=200
LR=0.001
EMB_DIMS=512
EMB_NN=dgcnn
POINTER=transformer
HEAD=mlp
NUM_POINTS=1024
PAIR_MODE=one_to_one

# 运行训练脚本
${PYTHON} train_dcp.py \
  --dataset-type c3vd \
  --dataset-path ${DATASET_PATH} \
  --pair-mode ${PAIR_MODE} \
  --num-points ${NUM_POINTS} \
  --use-voxelization \
  --emb-dims ${EMB_DIMS} \
  --emb-nn ${EMB_NN} \
  --pointer ${POINTER} \
  --head ${HEAD} \
  --batch-size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --save-dir ${SAVE_DIR} \
  "$@" 