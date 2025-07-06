#!/usr/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G
#$ -l h_rt=174000  # 最大运行时长4小时
#$ -l gpu=true

#$ -pe gpu 1
#$ -N train_dcp_modelnet
#$ -o /SAN/medic/MRpcr/logs/train_dcp_modelnet_output.log
#$ -e /SAN/medic/MRpcr/logs/train_dcp_modelnet_error.log
#$ -wd /SAN/medic/MRpcr
cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
echo "Current working directory: $(pwd)"

# Activate Conda environment
echo "Activating Conda environment..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

# 设置 ModelNet40 HDF5 数据集路径环境变量
export MODELNET40_HDF5_PATH=${DATASET_PATH}

# Set Python command to use Conda environment's python
PYTHON="nice -n 10 python"
# 用 DCP 在 ModelNet40 数据集上训练

# 数据集路径
DATASET_PATH=/SAN/medic/MRpcr/modelnet40_ply_hdf5_2048
# 类别列表文件
CATEGORY_FILE=sampledata/modelnet40_half1.txt
# 模型保存目录
SAVE_DIR=checkpoints/dcp_modelnet
# 超参数设置
BATCH_SIZE=16
EPOCHS=100
LR=0.0005
EMB_DIMS=512
EMB_NN=dgcnn
POINTER=transformer
HEAD=mlp
NUM_POINTS=1024

# 运行训练脚本
${PYTHON} train_dcp.py \
  --dataset-type modelnet \
  --dataset-path ${DATASET_PATH} \
  --categoryfile ${CATEGORY_FILE} \
  --num-points ${NUM_POINTS} \
  --emb-dims ${EMB_DIMS} \
  --emb-nn ${EMB_NN} \
  --pointer ${POINTER} \
  --head ${HEAD} \
  --batch-size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --save-dir ${SAVE_DIR} \
  "$@" 