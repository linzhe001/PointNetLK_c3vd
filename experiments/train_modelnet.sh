#! /usr/bin/bash
#$ -l tmem=64G
#$ -l h_vmem=64G            
#$ -l h_rt=720000  
#$ -l gpu=true

#$ -pe gpu 1
#$ -N ljiang_train_models_modelnet
#$ -o /SAN/medic/MRpcr/logs/c_train_models_modelnet_output.log
#$ -e /SAN/medic/MRpcr/logs/c_train_models_modelnet_error.log
#$ -wd /SAN/medic/MRpcr

  
# 设置工作目录为项目根目录
cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
echo "当前工作目录: $(pwd)"

# 激活Conda环境
echo "正在激活Conda环境..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 创建结果和日志目录
mkdir -p /SAN/medic/MRpcr/results/modelnet

# Python命令
PY3="nice -n 10 python"

# 设置变量
DATASET_PATH="/SAN/medic/MRpcr/ModelNet40"
CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/modelnet40_half1.txt"
NUM_POINTS=1024
DEVICE="cuda:0"
BATCH_SIZE=32
DATE_TAG=$(date +"%m%d")
MAG=0.8 # PointLK的变换幅度

# 打印配置信息
echo "========== 训练配置 =========="
echo "数据集路径: ${DATASET_PATH}"
echo "类别文件: ${CATEGORY_FILE}"
echo "点云数量: ${NUM_POINTS}"
echo "批次大小: ${BATCH_SIZE}"
echo "设备: ${DEVICE}"

# 检查数据集路径
echo "========== 数据集路径检查 =========="
if [ -d "${DATASET_PATH}" ]; then
    echo "数据集目录存在"
    echo "可用类别: $(ls ${DATASET_PATH} | wc -l)"
else
    echo "警告: 数据集目录不存在!"
fi

# 阶段1: 训练分类器
echo "========== 训练分类器 =========="
${PY3} train_classifier.py \
  -o /SAN/medic/MRpcr/results/modelnet/modelnet_classifier_${DATE_TAG} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l /SAN/medic/MRpcr/results/modelnet/modelnet_classifier_${DATE_TAG}.log \
  --dataset-type modelnet \
  --num-points ${NUM_POINTS} \
  --device ${DEVICE} \
  --epochs 250 \
  --batch-size ${BATCH_SIZE} \
  --workers 4 \
  --drop-last \
  --verbose \
  --base-lr 0.001 \
  --warmup-epochs 5 \
  --cosine-annealing

# 检查上一条命令是否成功执行
if [ $? -ne 0 ]; then
    echo "分类器训练失败，退出脚本"
    exit 1
fi

# 阶段2: 训练PointLK
echo "========== 训练PointLK =========="
${PY3} train_pointlk.py \
  -o /SAN/medic/MRpcr/results/modelnet/modelnet_pointlk_${DATE_TAG} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l /SAN/medic/MRpcr/results/modelnet/modelnet_pointlk_${DATE_TAG}.log \
  --dataset-type modelnet \
  --transfer-from /SAN/medic/MRpcr/results/modelnet/modelnet_classifier_${DATE_TAG}_feat_best.pth \
  --num-points ${NUM_POINTS} \
  --mag ${MAG} \
  --pointnet tune \
  --epochs 150 \
  --batch-size ${BATCH_SIZE} \
  --workers 4 \
  --device ${DEVICE} \
  --drop-last \
  --verbose

echo "训练完成!"

# 保存配置信息
CONFIG_FILE="/SAN/medic/MRpcr/results/modelnet/modelnet_training_${DATE_TAG}_config.txt"
echo "训练配置信息" > ${CONFIG_FILE}
echo "=============================" >> ${CONFIG_FILE}
echo "日期: $(date)" >> ${CONFIG_FILE}
echo "数据集路径: ${DATASET_PATH}" >> ${CONFIG_FILE}
echo "类别文件: ${CATEGORY_FILE}" >> ${CONFIG_FILE}
echo "点云数量: ${NUM_POINTS}" >> ${CONFIG_FILE}
echo "批次大小: ${BATCH_SIZE}" >> ${CONFIG_FILE}
echo "设备: ${DEVICE}" >> ${CONFIG_FILE}
echo "变换幅度: ${MAG}" >> ${CONFIG_FILE}
echo "总训练时间: $(date)" >> ${CONFIG_FILE}