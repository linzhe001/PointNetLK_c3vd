#! /usr/bin/bash
#$ -l tmem=64G
#$ -l h_vmem=64G            
#$ -l h_rt=720000  
#$ -l gpu=true
#$ -pe gpu 1
#$ -N ljiang_train_models_c3vd
#$ -o /SAN/medic/MRpcr/logs/c_train_models_c3vd_output.log
#$ -e /SAN/medic/MRpcr/logs/c_train_models_c3vd_error.log
#$ -wd /SAN/medic/MRpcr

  
# Set working directory to project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
echo "Current working directory: $(pwd)"

# Activate Conda environment
echo "Activating Conda environment..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Create result and log directories
mkdir -p /SAN/medic/MRpcr/results/c3vd

# Python command
PY3="nice -n 10 python"

# Set variables
DATASET_PATH="/SAN/medic/MRpcr/C3VD_datasets"
CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK/experiments/sampledata/C3VD.txt"
NUM_POINTS=1024
DEVICE="cuda:0"
BATCH_SIZE=16
DATE_TAG=$(date +"%m%d")

# Classifier uses random split, PointLK uses scene split
CLASSIFIER_SPLIT=""  # Classifier uses random split
POINTLK_SPLIT="--scene-split"  # PointLK uses scene split

# Quick dataset check
echo "========== Quick Dataset Check =========="
${PY3} -c "
import os, glob
dataset_path = '${DATASET_PATH}'
source_root = os.path.join(dataset_path, 'C3VD_ply_rot_scale_trans')
target_root = os.path.join(dataset_path, 'C3VD_ref')

# Count scenes and point cloud pairs
scenes = [os.path.basename(d) for d in glob.glob(os.path.join(source_root, '*')) if os.path.isdir(d)]
pairs_count = 0

for scene in scenes:
    source_files = glob.glob(os.path.join(source_root, scene, '????_adjusted.ply'))
    target_file = os.path.join(target_root, scene, 'coverage_mesh.ply')
    if os.path.exists(target_file):
        pairs_count += len(source_files)
        print(f'Scene {scene}: {len(source_files)} point cloud pairs')
    else:
        print(f'Scene {scene}: target point cloud not found {target_file}')

train_size = int(pairs_count * 0.8)
test_size = pairs_count - train_size

print(f'Total: {pairs_count} point cloud pairs, Training set ~{train_size}, Test set ~{test_size}')
print(f'Batch size: {${BATCH_SIZE}}')

if train_size % ${BATCH_SIZE} != 0 or test_size % ${BATCH_SIZE} != 0:
    print(f'Warning: Training or test set size is not divisible by batch size')
    print(f'Last batch: Training={train_size%${BATCH_SIZE}}, Test={test_size%${BATCH_SIZE}}')
"

# Stage 1: Train classifier (using random split)
echo "========== Training Classifier =========="
${PY3} train_classifier.py \
  -o /SAN/medic/MRpcr/results/c3vd/c3vd_classifier_${DATE_TAG} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l /SAN/medic/MRpcr/results/c3vd/c3vd_classifier_${DATE_TAG}.log \
  --dataset-type c3vd \
  --num-points ${NUM_POINTS} \
  --device ${DEVICE} \
  --epochs 50 \
  --batch-size ${BATCH_SIZE} \
  --workers 2 \
  --drop-last \
  ${CLASSIFIER_SPLIT} \
  --verbose 

# Check if previous command succeeded
if [ $? -ne 0 ]; then
    echo "Classifier training failed, exiting script"
    exit 1
fi

# Stage 2: Train PointLK (using scene split)
echo "========== Training PointLK =========="
${PY3} train_pointlk.py \
  -o /SAN/medic/MRpcr/results/c3vd/c3vd_pointlk_${DATE_TAG} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l /SAN/medic/MRpcr/results/c3vd/c3vd_pointlk_${DATE_TAG}.log \
  --dataset-type c3vd \
  --transfer-from /SAN/medic/MRpcr/results/c3vd/c3vd_classifier_${DATE_TAG}_feat_best.pth \
  --num-points ${NUM_POINTS} \
  --mag 0.8 \
  --pointnet tune \
  --epochs 300 \
  --batch-size ${BATCH_SIZE} \
  --workers 4 \
  --device ${DEVICE} \
  --drop-last \
  ${POINTLK_SPLIT} \
  --verbose

echo "Training complete!"

# Save configuration information
echo "Dataset path: ${DATASET_PATH}" > /SAN/medic/MRpcr/results/c3vd/c3vd_training_${DATE_TAG}_config.txt
echo "Category file: ${CATEGORY_FILE}" >> /SAN/medic/MRpcr/results/c3vd/c3vd_training_${DATE_TAG}_config.txt
echo "Point count: ${NUM_POINTS}" >> /SAN/medic/MRpcr/results/c3vd/c3vd_training_${DATE_TAG}_config.txt
echo "Batch size: ${BATCH_SIZE}" >> /SAN/medic/MRpcr/results/c3vd/c3vd_training_${DATE_TAG}_config.txt
echo "Device: ${DEVICE}" >> /SAN/medic/MRpcr/results/c3vd/c3vd_training_${DATE_TAG}_config.txt
echo "Training date: $(date)" >> /SAN/medic/MRpcr/results/c3vd/c3vd_training_${DATE_TAG}_config.txt

# Disabled after training
# sudo swapoff /swapfile
# sudo rm /swapfile
