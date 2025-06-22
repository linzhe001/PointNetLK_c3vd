# AttentionNet配准模型测试指南

本文档说明如何使用修改后的测试脚本来测试通过`train_attention_c3vd.sh`训练出来的AttentionNet配准模型。

## 文件概述

### 修改的文件
- **`test_pointlk.py`**: 原始测试脚本，已添加对AttentionNet模型的支持
  - 新增`--model-type`参数，支持选择`pointnet`或`attention`
  - 新增`--num-attention-blocks`和`--num-heads`参数
  - 修改Action类以支持创建AttentionNet模型

### 新增的测试脚本
- **`test_attention_c3vd.sh`**: 集群环境测试脚本（适用于SGE集群）
- **`test_attention_local.sh`**: 本地环境测试脚本（适用于本地或其他环境）

## 使用方法

### 1. 集群环境测试（推荐用于SGE集群）

```bash
# 直接提交作业到集群
qsub test_attention_c3vd.sh

# 或者先检查配置再提交
./test_attention_c3vd.sh
```

**特点：**
- 自动检测GPU可用性
- 自动查找最新训练的模型
- 完整的文件检查和错误处理
- 详细的配置信息输出
- 支持扰动文件夹批量测试

### 2. 本地环境测试

```bash
# 方法1: 直接运行
./test_attention_local.sh

# 方法2: 指定模型路径
export ATTENTION_MODEL_PATH="/path/to/your/attention_model.pth"
./test_attention_local.sh
```

**特点：**
- 自动检测conda环境
- 灵活的路径配置
- 减少的样本数量（快速测试）
- 适用于不同的计算环境

### 3. 手动运行Python脚本

```bash
python test_pointlk.py \
  -o results/attention_test.csv \
  -i /path/to/C3VD_datasets \
  -c sampledata/c3vd.txt \
  -l test.log \
  --dataset-type c3vd \
  --num-points 1024 \
  --max-iter 20 \
  --delta 1.0e-2 \
  --device cuda:0 \
  --max-samples 1000 \
  --pair-mode one_to_one \
  --use-joint-normalization \
  --perturbation-dir /path/to/perturbations \
  --model-type attention \
  --dim-k 1024 \
  --num-attention-blocks 3 \
  --num-heads 8 \
  --symfn max \
  --pretrained /path/to/attention_model.pth
```

## 配置参数说明

### 必需参数
- `-o`: 输出CSV文件路径
- `-i`: C3VD数据集根目录
- `-c`: 类别文件路径
- `--model-type attention`: 指定使用AttentionNet模型
- `--pretrained`: 预训练的AttentionNet模型路径

### AttentionNet特定参数
- `--dim-k`: 特征维度（默认1024，需与训练时一致）
- `--num-attention-blocks`: 注意力块数量（默认3）
- `--num-heads`: 多头注意力头数（默认8）
- `--symfn`: 聚合函数（max/avg，默认max）

### 测试配置参数
- `--max-samples`: 最大测试样本数（减少此值以加快测试）
- `--device`: 计算设备（cuda:0或cpu）
- `--pair-mode`: 点云配对模式（one_to_one或scene_reference）
- `--perturbation-dir`: 扰动文件目录（包含多个.csv扰动文件）

## 预期输出

### 测试结果文件
```
results/attention_c3vd/test_results/
├── attention_test_MMDD_HHMM.csv           # 主结果文件
├── attention_test_MMDD_HHMM.log           # 测试日志
├── attention_test_MMDD_HHMM_config.txt    # 配置信息
└── angle_XX/                              # 各角度扰动的详细结果
    ├── results_pert_XX.csv
    └── log_pert_XX.csv.log
```

### 结果文件格式
CSV文件包含以下列：
- `h_w1, h_w2, h_w3`: 预测的旋转扭曲向量
- `h_v1, h_v2, h_v3`: 预测的平移向量
- `g_w1, g_w2, g_w3`: 真实的旋转扭曲向量
- `g_v1, g_v2, g_v3`: 真实的平移向量

## 故障排除

### 常见问题

1. **找不到模型文件**
   ```bash
   # 检查模型目录
   ls results/attention_c3vd/attention_pointlk_*_model_best.pth
   
   # 手动指定模型路径
   export ATTENTION_MODEL_PATH="/absolute/path/to/model.pth"
   ```

2. **CUDA内存不足**
   ```bash
   # 方法1: 减少样本数量
   # 在脚本中修改 MAX_SAMPLES=200
   
   # 方法2: 使用CPU
   # 在脚本中修改 DEVICE="cpu"
   ```

3. **缺少扰动文件**
   ```bash
   # 检查扰动目录
   ls perturbations/*.csv
   
   # 如果没有，需要先生成扰动文件
   python generate_perturbations.py
   ```

4. **数据集路径错误**
   ```bash
   # 检查数据集结构
   ls C3VD_datasets/
   # 应该包含：C3VD_ply_source/ 和 visible_point_cloud_ply_depth/
   ```

### 验证测试结果

1. **检查日志文件**查看是否有错误或警告
2. **检查CSV文件**确认包含有效的数值数据
3. **比较不同扰动角度**的配准精度
4. **计算配准成功率**和平均误差

## 模型比较

您可以使用相同的扰动文件测试不同的模型：
- 原始PointNet配准模型（`--model-type pointnet`）
- AttentionNet配准模型（`--model-type attention`）

这样可以直接比较两种架构的配准性能。

## 注意事项

1. **参数一致性**：测试时的模型参数必须与训练时完全一致
2. **数据预处理**：确保测试数据的预处理方式与训练时相同
3. **扰动范围**：测试扰动的范围应该与训练时的扰动范围相匹配
4. **GPU内存**：AttentionNet模型可能需要更多GPU内存
5. **计算时间**：由于attention机制，推理时间可能比原始PointNet稍长 