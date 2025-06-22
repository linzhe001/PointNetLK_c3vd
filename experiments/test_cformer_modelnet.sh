#!/usr/bin/bash
#$ -l tmem=32G
#$ -l h_vmem=32G            
#$ -l h_rt=3600  # 1小时测试时间
#$ -l gpu=true
#$ -l gpu_type=a6000
#$ -pe gpu 1
#$ -N ljiang_test_cformer_modelnet
#$ -o /SAN/medic/MRpcr/logs/f_test_cformer_modelnet_output.log
#$ -e /SAN/medic/MRpcr/logs/f_test_cformer_modelnet_error.log
#$ -wd /SAN/medic/MRpcr

cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
# 设置工作目录
echo "当前工作目录: $(pwd)"

# 激活Conda环境
echo "激活Conda环境..."
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

# 创建结果和日志目录
mkdir -p /SAN/medic/MRpcr/results/cformer_modelnet/test_results

# Python命令
PY3="nice -n 10 python"

# 设置变量
DATASET_PATH="/SAN/medic/MRpcr/ModelNet40"
CATEGORY_FILE="/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/modelnet40_half2.txt"
NUM_POINTS=1024
DEVICE="cuda:0"
DATE_TAG=$(date +"%m%d")

# CFormer模型配置 (与训练保持一致)
DIM_K=1024                    # 特征维度
NUM_PROXY_POINTS=8            # 代理点数量
NUM_BLOCKS=2                  # CFormer块数量
SYMFN="max"                   # 聚合函数: max, avg, 或 cd_pool
MAX_ITER=20                   # LK最大迭代次数
DELTA=1.0e-2                  # LK步长

# 最大测试样本数
MAX_SAMPLES=2000

# 可视化设置（可选）
VISUALIZE_PERT="" # 如果需要可视化，设置为 "--visualize-pert pert_010.csv pert_020.csv"
VISUALIZE_SAMPLES=3

# 模型路径 (需要根据实际训练结果调整)
CFORMER_MODEL_PREFIX="/SAN/medic/MRpcr/results/cformer_modelnet/cformer_pointlk_${DATE_TAG}"
CFORMER_MODEL="${CFORMER_MODEL_PREFIX}_model_best.pth"
# 注意：测试时只使用完整的配准模型权重，不需要单独的特征权重

# 如果今天的模型未找到，尝试寻找最新模型
if [ ! -f "${CFORMER_MODEL}" ]; then
    echo "⚠️  未找到今天的模型文件: ${CFORMER_MODEL}"
    echo "🔍 搜索最新的cformer模型..."
    
    # 搜索最新的cformer pointlk模型
    LATEST_MODEL=$(find /SAN/medic/MRpcr/results/cformer_modelnet/ -name "cformer_pointlk_*_model_best.pth" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [ -n "${LATEST_MODEL}" ] && [ -f "${LATEST_MODEL}" ]; then
        CFORMER_MODEL="${LATEST_MODEL}"
        # 提取模型前缀
        CFORMER_MODEL_PREFIX=$(echo "${CFORMER_MODEL}" | sed 's/_model_best\.pth$//')
        echo "✅ 找到最新模型: ${CFORMER_MODEL}"
    else
        echo "❌ 错误: 未找到任何cformer模型文件!"
        echo "请确保已运行train_cformer_modelnet.sh并成功训练了模型"
        exit 1
    fi
fi

# 扰动文件目录
PERTURBATION_DIR="/SAN/medic/MRpcr/PointNetLK_c3vd/gt"

# 检查扰动目录
if [ ! -d "${PERTURBATION_DIR}" ] || [ $(find "${PERTURBATION_DIR}" -name "*.csv" | wc -l) -eq 0 ]; then
    echo "❌ 错误: 扰动目录不存在或为空: ${PERTURBATION_DIR}"
    echo "请确保扰动文件已正确生成"
    exit 1
fi

# 测试结果输出目录
TEST_RESULTS_DIR="/SAN/medic/MRpcr/results/cformer_modelnet/test_results"
# 输出文件基本名称 - 结果将按角度存储在子目录中
TEST_OUTPUT_PREFIX="${TEST_RESULTS_DIR}/results"
TEST_LOG="${TEST_RESULTS_DIR}/test_log_${DATE_TAG}.log"

# 打印配置信息
echo "========== CFormer配准模型测试配置 =========="
echo "🧠 模型类型: CFormer配准模型"
echo "📂 数据集路径: ${DATASET_PATH}"
echo "📄 类别文件: ${CATEGORY_FILE}"
echo "🎲 点数: ${NUM_POINTS}"
echo "🖥️  设备: ${DEVICE}"
echo "📊 最大测试样本数: ${MAX_SAMPLES}"
echo ""
echo "🔧 CFormer参数:"
echo "   - 特征维度: ${DIM_K}"
echo "   - 代理点数量: ${NUM_PROXY_POINTS}"
echo "   - CFormer块数量: ${NUM_BLOCKS}"
echo "   - 聚合函数: ${SYMFN}"
echo "   - LK最大迭代: ${MAX_ITER}"
echo "   - LK步长: ${DELTA}"
echo ""
echo "📁 模型文件:"
echo "   - 配准模型: ${CFORMER_MODEL}"
echo "   - 扰动目录: ${PERTURBATION_DIR}"
echo ""
echo "📁 输出路径:"
echo "   - 测试结果前缀: ${TEST_OUTPUT_PREFIX}"
echo "   - 测试日志: ${TEST_LOG}"

# 检查必要文件
echo ""
echo "========== 文件检查 =========="

# 检查数据集
if [ -d "${DATASET_PATH}" ]; then
    echo "✅ 数据集目录存在"
    TRAIN_COUNT=$(find "${DATASET_PATH}/train" -name "*.off" 2>/dev/null | wc -l)
    TEST_COUNT=$(find "${DATASET_PATH}/test" -name "*.off" 2>/dev/null | wc -l)
    echo "📊 训练集文件数量: ${TRAIN_COUNT}"
    echo "📊 测试集文件数量: ${TEST_COUNT}"
else
    echo "❌ 错误: 数据集目录不存在: ${DATASET_PATH}"
    exit 1
fi

# 检查类别文件
if [ -f "${CATEGORY_FILE}" ]; then
    echo "✅ 类别文件存在"
    CATEGORY_COUNT=$(wc -l < "${CATEGORY_FILE}")
    echo "📊 类别数量: ${CATEGORY_COUNT}"
else
    echo "❌ 错误: 类别文件不存在: ${CATEGORY_FILE}"
    exit 1
fi

# 检查模型文件
if [ -f "${CFORMER_MODEL}" ]; then
    echo "✅ CFormer配准模型存在: ${CFORMER_MODEL}"
    MODEL_SIZE=$(du -h "${CFORMER_MODEL}" | cut -f1)
    echo "📊 模型文件大小: ${MODEL_SIZE}"
else
    echo "❌ 错误: CFormer配准模型不存在: ${CFORMER_MODEL}"
    exit 1
fi

# 检查扰动目录
if [ -d "${PERTURBATION_DIR}" ]; then
    echo "✅ 扰动目录存在"
    PERT_COUNT=$(find "${PERTURBATION_DIR}" -name "*.csv" | wc -l)
    echo "📊 扰动文件数量: ${PERT_COUNT}"
    if [ ${PERT_COUNT} -eq 0 ]; then
        echo "⚠️  警告: 扰动目录中没有.csv文件"
        echo "将尝试生成默认扰动文件..."
        # 可以在这里添加生成扰动文件的代码
    else
        echo "📋 扰动文件列表:"
        find "${PERTURBATION_DIR}" -name "*.csv" | head -5
        if [ ${PERT_COUNT} -gt 5 ]; then
            echo "   ... (共${PERT_COUNT}个扰动文件)"
        fi
    fi
else
    echo "❌ 错误: 扰动目录不存在: ${PERTURBATION_DIR}"
    exit 1
fi

# GPU内存检查
echo ""
echo "========== GPU状态检查 =========="
if command -v nvidia-smi &> /dev/null; then
    echo "🖥️  GPU信息:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️  无法获取GPU信息"
fi

echo ""
echo "========== 开始CFormer配准模型测试 =========="
echo "🚀 即将开始测试..."
echo "⏱️  预计测试时间: ~30-60分钟 (取决于扰动文件数量和样本数)"
echo ""

# 构建可视化参数
VISUALIZE_PARAMS=""
if [ -n "${VISUALIZE_PERT}" ]; then
    VISUALIZE_PARAMS="${VISUALIZE_PERT} --visualize-samples ${VISUALIZE_SAMPLES}"
fi

# 运行测试
${PY3} test_pointlk.py \
  -o ${TEST_OUTPUT_PREFIX} \
  -i ${DATASET_PATH} \
  -c ${CATEGORY_FILE} \
  -l ${TEST_LOG} \
  --dataset-type modelnet \
  --num-points ${NUM_POINTS} \
  --max-iter ${MAX_ITER} \
  --delta ${DELTA} \
  --device ${DEVICE} \
  --max-samples ${MAX_SAMPLES} \
  --perturbation-dir ${PERTURBATION_DIR} \
  --model-type cformer \
  --dim-k ${DIM_K} \
  --num-proxy-points ${NUM_PROXY_POINTS} \
  --num-blocks ${NUM_BLOCKS} \
  --symfn ${SYMFN} \
  --pretrained ${CFORMER_MODEL} \
  ${VISUALIZE_PARAMS}

# 检查测试结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 CFormer配准模型测试完成!"
    echo "📁 测试结果保存到: ${TEST_RESULTS_DIR}"
    echo "📋 测试日志: ${TEST_LOG}"
    
    # 显示生成的结果文件 - 修改为显示角度目录和日志文件
    echo ""
    echo "📊 生成的测试结果目录:"
    find "${TEST_RESULTS_DIR}" -type d -name "angle_*" | sort | xargs ls -ld
    
    echo ""
    echo "📊 样本结果文件:"
    # 查找并显示角度目录中的一些结果文件
    find "${TEST_RESULTS_DIR}/angle_"* -name "*.log" -type f | sort | head -10 | xargs ls -lh
    
    # 统计信息
    echo ""
    echo "📈 测试统计:"
    # 计算所有日志文件（包括角度目录和主目录中的）
    RESULT_FILES=$(find "${TEST_RESULTS_DIR}" -name "*.log" -type f | wc -l)
    echo "   - 总结果文件数: ${RESULT_FILES}"
    
    # 计算角度目录数
    ANGLE_DIRS=$(find "${TEST_RESULTS_DIR}" -type d -name "angle_*" | wc -l)
    echo "   - 角度目录数: ${ANGLE_DIRS}"
    
    if [ ${RESULT_FILES} -gt 0 ]; then
        # 显示样本结果文件的统计信息
        SAMPLE_FILE=$(find "${TEST_RESULTS_DIR}" -name "results_*.log" -type f | head -1)
        if [ -f "${SAMPLE_FILE}" ]; then
            echo "   - 样本结果文件: ${SAMPLE_FILE}"
            echo "   - 结果文件统计预览:"
            # 显示统计部分（最终统计结果）
            grep "# Average.*error:" "${SAMPLE_FILE}" | head -3
        fi
    fi
    
else
    echo ""
    echo "❌ CFormer配准模型测试失败!"
    echo "请检查错误日志: ${TEST_LOG}"
    
    # 显示最后几行错误信息
    if [ -f "${TEST_LOG}" ]; then
        echo ""
        echo "📋 最新错误信息:"
        tail -10 "${TEST_LOG}"
    fi
    
    exit 1
fi

# 保存测试配置信息
CONFIG_FILE="${TEST_RESULTS_DIR}/cformer_test_${DATE_TAG}_config.txt"
echo "🧠 CFormer配准模型测试配置" > ${CONFIG_FILE}
echo "=====================================" >> ${CONFIG_FILE}
echo "测试完成时间: $(date)" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "🔧 模型配置:" >> ${CONFIG_FILE}
echo "模型类型: CFormer配准模型" >> ${CONFIG_FILE}
echo "特征维度: ${DIM_K}" >> ${CONFIG_FILE}
echo "代理点数量: ${NUM_PROXY_POINTS}" >> ${CONFIG_FILE}
echo "CFormer块数量: ${NUM_BLOCKS}" >> ${CONFIG_FILE}
echo "聚合函数: ${SYMFN}" >> ${CONFIG_FILE}
echo "LK最大迭代: ${MAX_ITER}" >> ${CONFIG_FILE}
echo "LK步长: ${DELTA}" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "📊 测试配置:" >> ${CONFIG_FILE}
echo "数据集路径: ${DATASET_PATH}" >> ${CONFIG_FILE}
echo "类别文件: ${CATEGORY_FILE}" >> ${CONFIG_FILE}
echo "数据集类型: modelnet" >> ${CONFIG_FILE}
echo "点数: ${NUM_POINTS}" >> ${CONFIG_FILE}
echo "最大测试样本数: ${MAX_SAMPLES}" >> ${CONFIG_FILE}
echo "设备: ${DEVICE}" >> ${CONFIG_FILE}
echo "扰动目录: ${PERTURBATION_DIR}" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "📁 模型文件:" >> ${CONFIG_FILE}
echo "配准模型: ${CFORMER_MODEL}" >> ${CONFIG_FILE}
echo "" >> ${CONFIG_FILE}
echo "📁 输出文件:" >> ${CONFIG_FILE}
echo "测试结果前缀: ${TEST_OUTPUT_PREFIX}" >> ${CONFIG_FILE}
echo "测试日志: ${TEST_LOG}" >> ${CONFIG_FILE}

echo ""
echo "💾 测试配置信息已保存到: ${CONFIG_FILE}"

echo ""
echo "🎯 测试完成总结:"
echo "📂 结果目录: ${TEST_RESULTS_DIR}"
echo "📄 配置文件: ${CONFIG_FILE}"
echo "🎯 配准模型: ${CFORMER_MODEL}"
echo "📋 测试日志: ${TEST_LOG}"
echo "⏰ 完成时间: $(date)"

echo ""
echo "🎉🎉🎉 CFormer配准模型测试全部完成! 🎉🎉🎉" 