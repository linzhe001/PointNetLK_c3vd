#!/bin/bash
#$ -l tmem=16G
#$ -l h_vmem=16G            
#$ -l h_rt=7200  # 2小时测试时间，单对测试比较快
#$ -l gpu=true
#$ -pe gpu 1
#$ -N ljiang_test_single_pair_cformer_depth_mesh
#$ -o /SAN/medic/MRpcr/logs/f_test_single_pair_cformer_depth_mesh_output.log
#$ -e /SAN/medic/MRpcr/logs/f_test_single_pair_cformer_depth_mesh_error.log
#$ -wd /SAN/medic/MRpcr

# CFormer单对点云配准测试脚本 - 多体素化参数测试
# CFormer Single pair point cloud registration test script - Multiple voxelization parameters test

echo "========== CFormer单对点云配准测试 - 多体素化参数 CFormer Single Pair Point Cloud Registration Test - Multiple Voxelization Parameters =========="

# 设置环境
cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

# 基础配置
PYTHON="python"
DEVICE="cuda:0"
NUM_POINTS=1024
MODEL_TYPE="cformer"

# 定义多组体素化配置
declare -A VOXEL_CONFIGS
VOXEL_CONFIGS["no_voxel"]="false,0,0,0,0,0"
VOXEL_CONFIGS["small_voxel"]="true,2,16,50,10000,0.05"
VOXEL_CONFIGS["medium_voxel"]="true,4,32,100,20000,0.1"
VOXEL_CONFIGS["large_voxel"]="true,8,64,200,30000,0.1"
VOXEL_CONFIGS["dense_voxel"]="true,1,64,300,50000,0.05"
VOXEL_CONFIGS["sparse_voxel"]="true,10,16,50,5000,0.2"

# 示例点云文件路径
SOURCE_CLOUD="/SAN/medic/MRpcr/pit_dataset/depth/depth.ply"
TARGET_CLOUD="/SAN/medic/MRpcr/pit_dataset/MRI/Mesh.ply"

# 检查文件是否存在
if [ ! -f "${SOURCE_CLOUD}" ]; then
    echo "错误: 源点云文件不存在: ${SOURCE_CLOUD}"
    exit 1
fi

if [ ! -f "${TARGET_CLOUD}" ]; then
    echo "错误: 目标点云文件不存在: ${TARGET_CLOUD}"
    exit 1
fi

# CFormer模型文件路径
MODEL_PATH="/SAN/medic/MRpcr/results/cformer_c3vd/cformer_pointlk_resume_0620_model_best.pth"

# 检查模型文件是否存在
if [ ! -f "${MODEL_PATH}" ]; then
    echo "错误: CFormer模型文件不存在: ${MODEL_PATH}"
    echo "请确保模型文件路径正确或训练相应的模型"
    exit 1
fi

# 输出目录
OUTPUT_DIR="/SAN/medic/MRpcr/test_results_output/single_pair_cformer_test_depth_mesh"
mkdir -p ${OUTPUT_DIR}

# 简化的扰动列表 - 专注于关键角度测试
PERTURBATIONS=(
    "0.0,0.0,0.0,0.1,0.1,0.1"         # 0度旋转 + 固定平移
    "0.2618,0.0,0.0,0.1,0.1,0.1"      # 15度X轴旋转 + 固定平移
    "0.5236,0.0,0.0,0.1,0.1,0.1"      # 30度X轴旋转 + 固定平移
    "0.7854,0.0,0.0,0.1,0.1,0.1"      # 45度X轴旋转 + 固定平移
)

# CFormer模型参数（与训练保持一致）
DIM_K=1024
NUM_PROXY_POINTS=8
NUM_BLOCKS=2
SYMFN="max"

echo "========== CFormer配置信息 CFormer Configuration =========="
echo "源点云: ${SOURCE_CLOUD}"
echo "目标点云: ${TARGET_CLOUD}"
echo "模型类型: ${MODEL_TYPE}"
echo "模型路径: ${MODEL_PATH}"
echo "设备: ${DEVICE}"
echo "点云数量: ${NUM_POINTS}"
echo "输出目录: ${OUTPUT_DIR}"
echo "体素化配置数量: ${#VOXEL_CONFIGS[@]}"
echo "扰动数量: ${#PERTURBATIONS[@]}"
echo ""

# 测试每种体素化配置
for voxel_name in "${!VOXEL_CONFIGS[@]}"; do
    echo ""
    echo "========== 测试体素化配置: ${voxel_name} Testing Voxelization Config: ${voxel_name} =========="
    
    # 解析体素化配置
    IFS=',' read -r use_voxel voxel_size voxel_grid_size max_voxel_points max_voxels min_voxel_points_ratio <<< "${VOXEL_CONFIGS[$voxel_name]}"
    
    # 构建体素化参数
    if [ "${use_voxel}" = "true" ]; then
        VOXELIZATION_PARAMS="--use-voxelization --voxel-size ${voxel_size} --voxel-grid-size ${voxel_grid_size} --max-voxel-points ${max_voxel_points} --max-voxels ${max_voxels} --min-voxel-points-ratio ${min_voxel_points_ratio}"
        echo "体素化参数: 大小=${voxel_size}, 网格=${voxel_grid_size}, 最大点数=${max_voxel_points}, 最大体素=${max_voxels}, 最小比例=${min_voxel_points_ratio}"
    else
        VOXELIZATION_PARAMS="--no-voxelization"
        echo "体素化: 禁用"
    fi
    
    # 测试每个扰动
    for i in "${!PERTURBATIONS[@]}"; do
        PERTURBATION="${PERTURBATIONS[$i]}"
        OUTPUT_FILE="${OUTPUT_DIR}/cformer_${voxel_name}_result_${i}_$(echo ${PERTURBATION} | tr ',' '_').csv"
        
        echo ""
        echo "  测试 $((i+1))/${#PERTURBATIONS[@]} (${voxel_name}): 扰动=${PERTURBATION}"
        
        # 运行CFormer单对点云配准测试
        ${PYTHON} test_pointlk.py \
            --single-pair-mode \
            --source-cloud "${SOURCE_CLOUD}" \
            --target-cloud "${TARGET_CLOUD}" \
            --single-perturbation "${PERTURBATION}" \
            --enhanced-output \
            -o "${OUTPUT_FILE}" \
            -i "/SAN/medic/MRpcr/C3VD_datasets" \
            -c "/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/c3vd.txt" \
            --model-type "${MODEL_TYPE}" \
            --pretrained "${MODEL_PATH}" \
            --device "${DEVICE}" \
            --num-points ${NUM_POINTS} \
            ${VOXELIZATION_PARAMS} \
            --max-iter 20 \
            --delta 1.0e-2 \
            --dataset-type c3vd \
            --dim-k ${DIM_K} \
            --num-proxy-points ${NUM_PROXY_POINTS} \
            --num-blocks ${NUM_BLOCKS} \
            --symfn ${SYMFN}
        
        if [ $? -eq 0 ]; then
            echo "  ✅ 测试 $((i+1)) (${voxel_name}) 完成，结果保存到: ${OUTPUT_FILE}"
            
            # 在结果文件中添加体素化配置信息
            sed -i "1i# 体素化配置: ${voxel_name} - ${VOXEL_CONFIGS[$voxel_name]}" "${OUTPUT_FILE}"
        else
            echo "  ❌ 测试 $((i+1)) (${voxel_name}) 失败"
        fi
    done
done

echo ""
echo "========== CFormer多体素化参数测试完成 CFormer Multiple Voxelization Parameters Tests Completed =========="
echo "所有CFormer结果保存在: ${OUTPUT_DIR}"
echo "查看结果文件:"
ls -la ${OUTPUT_DIR}/*.csv 2>/dev/null || echo "没有生成结果文件"

# 生成测试总结
SUMMARY_FILE="${OUTPUT_DIR}/cformer_voxel_test_summary_$(date +%Y%m%d_%H%M%S).txt"
echo "CFormer多体素化参数测试总结" > ${SUMMARY_FILE}
echo "==============================" >> ${SUMMARY_FILE}
echo "完成时间: $(date)" >> ${SUMMARY_FILE}
echo "模型类型: ${MODEL_TYPE}" >> ${SUMMARY_FILE}
echo "模型路径: ${MODEL_PATH}" >> ${SUMMARY_FILE}
echo "源点云: ${SOURCE_CLOUD}" >> ${SUMMARY_FILE}
echo "目标点云: ${TARGET_CLOUD}" >> ${SUMMARY_FILE}
echo "体素化配置数量: ${#VOXEL_CONFIGS[@]}" >> ${SUMMARY_FILE}
echo "扰动数量: ${#PERTURBATIONS[@]}" >> ${SUMMARY_FILE}
echo "设备: ${DEVICE}" >> ${SUMMARY_FILE}
echo "" >> ${SUMMARY_FILE}
echo "体素化配置列表:" >> ${SUMMARY_FILE}
for voxel_name in "${!VOXEL_CONFIGS[@]}"; do
    echo "${voxel_name}: ${VOXEL_CONFIGS[$voxel_name]}" >> ${SUMMARY_FILE}
done
echo "" >> ${SUMMARY_FILE}
echo "扰动列表:" >> ${SUMMARY_FILE}
for i in "${!PERTURBATIONS[@]}"; do
    echo "$((i+1)). ${PERTURBATIONS[$i]}" >> ${SUMMARY_FILE}
done

echo ""
echo "📋 测试总结已保存到: ${SUMMARY_FILE}"
echo "🎉 CFormer多体素化参数测试全部完成!" 