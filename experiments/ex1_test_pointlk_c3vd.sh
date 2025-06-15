#! /usr/bin/bash
#$ -l tmem=64G
#$ -l h_vmem=64G            
#$ -l h_rt=720000  
#$ -l gpu=true
#$ -pe gpu 1
#$ -N ljiang_test_pointlk_c3vd_2
#$ -o /SAN/medic/MRpcr/logs/f_test_pointlk_c3vd_output.log
#$ -e /SAN/medic/MRpcr/logs/f_test_pointlk_c3vd_error.log
#$ -wd /SAN/medic/MRpcr

source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

# 可以选择使用 "zero_shot_c3vd" 或 "nonconverged_c3vd" 作为标记
# 修改这个变量来切换标记类型
# TAG="zero_shot_c3vd"
TAG="f_test_pointlk_c3vd"

# 配对模式设置 - 固定使用one_to_one模式
PAIR_MODE="one_to_one"

# 联合归一化设置
USE_JOINT_NORMALIZATION="--use-joint-normalization"  # 启用联合归一化

# 打印配对模式信息
echo "========== 配对模式配置 =========="
echo "使用配对模式: ${PAIR_MODE}"
echo "配对方式: 每个源点云匹配对应帧号的目标点云"
echo "目标点云目录: visible_point_cloud_ply_depth"

# for output
OUTDIR=/SAN/medic/MRpcr/result/c3vd_${TAG}
mkdir -p ${OUTDIR}

# Python3 command
PY3="nice -n 10 python"

# categories for testing and the trained model
MODEL=/SAN/medic/MRpcr/PointNetLK_c3vd/results/pointnet_c3vd_approx_0529/pointlk_model_best.pth
CMN="-i /SAN/medic/MRpcr/C3VD_datasets -c /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/c3vd.txt --format wt --pretrained ${MODEL} --dataset-type c3vd --num-points 1024 --max-samples 1000 --device cuda:0 --pair-mode ${PAIR_MODE} ${USE_JOINT_NORMALIZATION}"

# 扰动文件夹路径 - 更新为新路径
PERDIR=/SAN/medic/MRpcr/PointNetLK_c3vd/gt

# 单个扰动文件路径
SINGLE_PERT_FILE=/SAN/medic/MRpcr/PointNetLK_c3vd/gt_poses.csv

# 需要可视化的扰动文件列表
# 修改此变量可添加或移除要可视化的扰动文件
VIS_PERTS="pert_060.csv"
# 每个扰动文件可视化的样本数量
VIS_SAMPLES=2

# 添加可视化参数到命令中
echo "========== 可视化配置 =========="
echo "将为以下扰动文件进行可视化: ${VIS_PERTS}"
echo "每个扰动文件将可视化 ${VIS_SAMPLES} 个随机样本"
echo "可视化结果将保存在: ${OUTDIR}/visualize 目录下"
echo "每个可视化样本会创建以下文件:"
echo "  - {场景名}_{序号}_source.ply：源点云"
echo "  - {场景名}_{序号}_target.ply：目标点云"
echo "  - visualization_log.txt：包含配准详情的日志文件"

# 设置可视化参数
VIS_PARAMS="--visualize-pert ${VIS_PERTS} --visualize-samples ${VIS_SAMPLES}"

# 使用单个扰动文件和扰动文件夹同时测试
echo "========== 开始测试 =========="
echo "扰动文件夹: ${PERDIR}"
echo "单个扰动文件: ${SINGLE_PERT_FILE}"

# 首先使用单个扰动文件进行测试
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_pointlk.py ${CMN} -o ${OUTDIR}/result_single_${TAG}.csv -p ${SINGLE_PERT_FILE} -l ${OUTDIR}/log_single.log ${VIS_PARAMS}

# 然后使用扰动文件夹中的文件进行测试
${PY3} /SAN/medic/MRpcr/PointNetLK_c3vd/experiments/test_pointlk.py ${CMN} -o ${OUTDIR}/result_folder_${TAG}.csv --perturbation-dir ${PERDIR} -l ${OUTDIR}/log_folder.log ${VIS_PARAMS}

echo "========== 测试和可视化已完成 =========="
echo "测试结果保存在: ${OUTDIR}"
echo "可视化结果保存在: ${OUTDIR}/visualize"

#EOF
