#! /usr/bin/bash
#$ -l tmem=64G
#$ -l h_vmem=64G            
#$ -l h_rt=720000  
#$ -l gpu=true
#$ -pe gpu 1
#$ -N ljiang_test_icp_c3vd
#$ -o /SAN/medic/MRpcr/logs/test_icp_c3vd_output.log
#$ -e /SAN/medic/MRpcr/logs/test_icp_c3vd_error.log
#$ -wd /SAN/medic/MRpcr

# 激活正确的conda环境 - 改为与PointLK相同的环境
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

# 在conda activate pointlk后添加以下行
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

# 标记
TAG="icp_ppt"

# 配对模式设置 - 固定使用one_to_one模式
PAIR_MODE="one_to_one"

# 打印配对模式信息
echo "========== 配对模式配置 =========="
echo "使用配对模式: ${PAIR_MODE}"
echo "配对方式: 每个源点云匹配对应帧号的目标点云"
echo "目标点云目录: visible_point_cloud_ply_depth"

# 修改为与PointLK相同的输出目录
OUTDIR=/SAN/medic/MRpcr/result/c3vd_${TAG}
mkdir -p ${OUTDIR}

# Python命令
PY3="nice -n 10 python"

# 修改为与PointLK相同的数据集路径和文件引用方式
DATASET_PATH="/SAN/medic/MRpcr/C3VD_datasets"
CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/c3vd.txt"
NUM_POINTS=1024
MAX_SAMPLES=200
DEVICE="cuda:0"  # 与PointLK保持一致
PERDIR="/SAN/medic/MRpcr/result/gt"  # 与PointLK保持一致

# categories for testing
CMN="-i ${DATASET_PATH} -c ${CATEGORY_FILE} --format wt --dataset-type c3vd --num-points ${NUM_POINTS} --max-samples ${MAX_SAMPLES} --device ${DEVICE} --pair-mode ${PAIR_MODE}"

# 测试ICP算法
echo "========== 开始测试ICP算法 =========="
echo "使用配对模式: ${PAIR_MODE}"
echo "最大样本数: ${MAX_SAMPLES}"
echo "每个点云点数: ${NUM_POINTS}"
echo "设备: ${DEVICE}"
echo "数据集路径: ${DATASET_PATH}"
echo "结果输出到: ${OUTDIR}"

# 修改为与PointLK相同的脚本路径引用风格
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_icp.py ${CMN} -o ${OUTDIR}/result_010_${TAG}.csv -p ${PERDIR}/pert_010.csv -l ${OUTDIR}/log_010.log
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_icp.py ${CMN} -o ${OUTDIR}/result_000_${TAG}.csv -p ${PERDIR}/pert_000.csv -l ${OUTDIR}/log_000.log
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_icp.py ${CMN} -o ${OUTDIR}/result_090_${TAG}.csv -p ${PERDIR}/pert_090.csv -l ${OUTDIR}/log_090.log
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_icp.py ${CMN} -o ${OUTDIR}/result_060_${TAG}.csv -p ${PERDIR}/pert_060.csv -l ${OUTDIR}/log_060.log
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_icp.py ${CMN} -o ${OUTDIR}/result_120_${TAG}.csv -p ${PERDIR}/pert_120.csv -l ${OUTDIR}/log_120.log
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_icp.py ${CMN} -o ${OUTDIR}/result_030_${TAG}.csv -p ${PERDIR}/pert_030.csv -l ${OUTDIR}/log_030.log
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_icp.py ${CMN} -o ${OUTDIR}/result_180_${TAG}.csv -p ${PERDIR}/pert_180.csv -l ${OUTDIR}/log_180.log
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_icp.py ${CMN} -o ${OUTDIR}/result_020_${TAG}.csv -p ${PERDIR}/pert_020.csv -l ${OUTDIR}/log_020.log
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_icp.py ${CMN} -o ${OUTDIR}/result_040_${TAG}.csv -p ${PERDIR}/pert_040.csv -l ${OUTDIR}/log_040.log
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_icp.py ${CMN} -o ${OUTDIR}/result_050_${TAG}.csv -p ${PERDIR}/pert_050.csv -l ${OUTDIR}/log_050.log
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_icp.py ${CMN} -o ${OUTDIR}/result_070_${TAG}.csv -p ${PERDIR}/pert_070.csv -l ${OUTDIR}/log_070.log
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_icp.py ${CMN} -o ${OUTDIR}/result_080_${TAG}.csv -p ${PERDIR}/pert_080.csv -l ${OUTDIR}/log_080.log
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_icp.py ${CMN} -o ${OUTDIR}/result_100_${TAG}.csv -p ${PERDIR}/pert_100.csv -l ${OUTDIR}/log_100.log
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_icp.py ${CMN} -o ${OUTDIR}/result_110_${TAG}.csv -p ${PERDIR}/pert_110.csv -l ${OUTDIR}/log_110.log
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_icp.py ${CMN} -o ${OUTDIR}/result_130_${TAG}.csv -p ${PERDIR}/pert_130.csv -l ${OUTDIR}/log_130.log
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_icp.py ${CMN} -o ${OUTDIR}/result_140_${TAG}.csv -p ${PERDIR}/pert_140.csv -l ${OUTDIR}/log_140.log
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_icp.py ${CMN} -o ${OUTDIR}/result_150_${TAG}.csv -p ${PERDIR}/pert_150.csv -l ${OUTDIR}/log_150.log
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_icp.py ${CMN} -o ${OUTDIR}/result_160_${TAG}.csv -p ${PERDIR}/pert_160.csv -l ${OUTDIR}/log_160.log
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_icp.py ${CMN} -o ${OUTDIR}/result_170_${TAG}.csv -p ${PERDIR}/pert_170.csv -l ${OUTDIR}/log_170.log

echo "ICP测试完成!"
echo "结果保存在: ${OUTDIR}"

#EOF


