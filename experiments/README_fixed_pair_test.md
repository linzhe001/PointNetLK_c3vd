# 固定点云对配准测试指南

## 脚本说明

### 1. `test_fixed_pair_mamba3d.sh` - Bash测试脚本
使用现有的`test_pointlk.py`进行固定点云对测试，配置简单。

### 2. `test_fixed_pair.py` - Python测试脚本  
专门的固定点云对测试脚本，功能完整，输出详细。

## 使用方法

### Bash脚本使用 (推荐)
```bash
# 编辑配置
nano test_fixed_pair_mamba3d.sh

# 修改关键配置:
SOURCE_CLOUD="${DATASET_PATH}/C3VD_ply_source/cecum_t1_a/0200_depth_pcd.ply"
TARGET_CLOUD="${DATASET_PATH}/C3VD_ply_source/cecum_t1_a/0210_depth_pcd.ply"
FIXED_PERTURBATION="0.1,0.15,0.05,2.0,1.5,0.8"

# 运行测试
./test_fixed_pair_mamba3d.sh
```

### Python脚本使用
```bash
python test_fixed_pair.py \
  --source-cloud /path/to/source.ply \
  --target-cloud /path/to/target.ply \
  --model-path /path/to/model.pth \
  --perturbation "0.1,0.15,0.05,2.0,1.5,0.8" \
  --model-type mamba3d \
  --device cuda:0
```

## 参数说明

### 扰动值格式: "rx,ry,rz,tx,ty,tz"
- rx,ry,rz: 旋转角度(弧度)
- tx,ty,tz: 平移距离(毫米)

### 扰动建议
- 小扰动: "0.05,0.03,0.02,1.0,0.5,0.3"
- 中等扰动: "0.15,0.10,0.08,3.0,2.0,1.5"  
- 大扰动: "0.25,0.20,0.15,5.0,4.0,3.0"

## 输出结果
- CSV文件: 定量配准误差结果
- 详细文件: 完整测试信息和变换矩阵
- 可选点云文件: 处理过程中的点云数据

## 评估标准
- 旋转误差 < 1°: 优秀
- 平移误差 < 1mm: 优秀
- 配准时间 < 1秒: 良好 
# Fixed Point Cloud Pair Registration Test Guide

本文档介绍如何使用固定点云对和固定扰动进行配准测试。

## 文件说明

### 1. `test_fixed_pair_mamba3d.sh`
一个完整的bash测试脚本，使用现有的`test_pointlk.py`进行固定点云对测试。

### 2. `test_fixed_pair.py`
一个专门的Python脚本，直接进行固定点云对的配准测试，更加简洁和直观。

## 使用方法

### 方法1：使用bash脚本 (推荐)

```bash
# 编辑脚本配置
nano test_fixed_pair_mamba3d.sh

# 在脚本中修改以下配置:
# 1. 源点云和目标点云路径
SOURCE_CLOUD="${DATASET_PATH}/C3VD_ply_source/cecum_t1_a/0200_depth_pcd.ply"
TARGET_CLOUD="${DATASET_PATH}/C3VD_ply_source/cecum_t1_a/0210_depth_pcd.ply"

# 2. 固定扰动值 (旋转角度弧度, 平移距离毫米)
FIXED_PERTURBATION="0.1,0.15,0.05,2.0,1.5,0.8"

# 运行测试
./test_fixed_pair_mamba3d.sh
```

### 方法2：使用Python脚本

```bash
# 基本用法
python test_fixed_pair.py \
  --source-cloud /path/to/source.ply \
  --target-cloud /path/to/target.ply \
  --model-path /path/to/model.pth \
  --perturbation "0.1,0.15,0.05,2.0,1.5,0.8"

# 完整用法示例
python test_fixed_pair.py \
  --source-cloud /SAN/medic/MRpcr/C3VD_datasets/C3VD_ply_source/cecum_t1_a/0200_depth_pcd.ply \
  --target-cloud /SAN/medic/MRpcr/C3VD_datasets/C3VD_ply_source/cecum_t1_a/0210_depth_pcd.ply \
  --model-path /SAN/medic/MRpcr/results/mamba3d_c3vd/mamba3d_pointlk_1201_model_best.pth \
  --perturbation "0.1,0.15,0.05,2.0,1.5,0.8" \
  --model-type mamba3d \
  --num-points 1024 \
  --max-iter 20 \
  --device cuda:0 \
  --output-dir ./results/my_fixed_test \
  --save-clouds \
  --verbose
```

## 参数说明

### 必需参数
- `--source-cloud`: 源点云文件路径(.ply格式)
- `--target-cloud`: 目标点云文件路径(.ply格式)  
- `--model-path`: 训练好的模型文件路径(.pth格式)
- `--perturbation`: 扰动值，格式为"rx,ry,rz,tx,ty,tz"
  - rx,ry,rz: 旋转角度(弧度)
  - tx,ty,tz: 平移距离(毫米)

### 可选参数
- `--model-type`: 模型类型 (默认: mamba3d)
- `--num-points`: 点云点数 (默认: 1024)
- `--max-iter`: LK最大迭代次数 (默认: 20)
- `--device`: 计算设备 (默认: cuda:0)
- `--output-dir`: 输出目录 (默认: ./results/fixed_pair_test)
- `--save-clouds`: 保存处理后的点云文件
- `--verbose`: 详细输出

## 扰动值设置建议

### 小扰动测试 (精度测试)
```bash
PERTURBATION="0.05,0.03,0.02,1.0,0.5,0.3"  # 小角度小平移
```

### 中等扰动测试 (鲁棒性测试)
```bash
PERTURBATION="0.15,0.10,0.08,3.0,2.0,1.5"  # 中等角度和平移
```

### 大扰动测试 (极限测试)
```bash
PERTURBATION="0.25,0.20,0.15,5.0,4.0,3.0"  # 大角度大平移
```

### 单轴测试
```bash
# 仅X轴旋转
PERTURBATION="0.1,0.0,0.0,0.0,0.0,0.0"

# 仅Z轴平移
PERTURBATION="0.0,0.0,0.0,0.0,0.0,2.0"
```

## 点云对选择建议

### 1. 同一场景内的连续帧
```bash
# 时间相近的帧，变化较小
SOURCE_CLOUD="cecum_t1_a/0200_depth_pcd.ply"
TARGET_CLOUD="cecum_t1_a/0205_depth_pcd.ply"
```

### 2. 同一场景内的间隔帧
```bash
# 时间间隔较大的帧，变化明显
SOURCE_CLOUD="cecum_t1_a/0200_depth_pcd.ply"  
TARGET_CLOUD="cecum_t1_a/0220_depth_pcd.ply"
```

### 3. 不同场景间的测试
```bash
# 不同场景，测试模型泛化能力
SOURCE_CLOUD="cecum_t1_a/0200_depth_pcd.ply"
TARGET_CLOUD="cecum_t2_a/0200_depth_pcd.ply"
```

## 输出结果

### CSV结果文件
包含配准误差的定量结果：
- `source_cloud`: 源点云文件名
- `target_cloud`: 目标点云文件名  
- `perturbation`: 使用的扰动值
- `rotation_error_deg`: 旋转误差(度)
- `translation_error_mm`: 平移误差(毫米)
- `registration_time_sec`: 配准耗时(秒)

### 详细结果文件
包含完整的测试信息：
- 测试配置参数
- 扰动矩阵
- 预测变换矩阵
- 详细误差分析

### 点云文件(可选)
如果启用`--save-clouds`选项，会保存：
- `source_original.txt`: 原始源点云
- `source_perturbed.txt`: 应用扰动后的源点云
- `target.txt`: 目标点云

## 批量测试示例

### 创建批量测试脚本
```bash
#!/bin/bash
# batch_fixed_test.sh

# 定义测试组合
SOURCES=("cecum_t1_a/0200_depth_pcd.ply" "sigmoid_t1_a/0150_depth_pcd.ply")
TARGETS=("cecum_t1_a/0210_depth_pcd.ply" "sigmoid_t1_a/0160_depth_pcd.ply")
PERTURBATIONS=("0.05,0.03,0.02,1.0,0.5,0.3" "0.15,0.10,0.08,3.0,2.0,1.5")

MODEL_PATH="/path/to/your/model.pth"

# 循环测试
for i in ${!SOURCES[@]}; do
    for j in ${!PERTURBATIONS[@]}; do
        echo "测试组合 $((i+1))-$((j+1))"
        python test_fixed_pair.py \
          --source-cloud "/SAN/medic/MRpcr/C3VD_datasets/C3VD_ply_source/${SOURCES[$i]}" \
          --target-cloud "/SAN/medic/MRpcr/C3VD_datasets/C3VD_ply_source/${TARGETS[$i]}" \
          --model-path "${MODEL_PATH}" \
          --perturbation "${PERTURBATIONS[$j]}" \
          --output-csv "results/batch_test_${i}_${j}.csv"
    done
done
```

## 故障排除

### 常见错误

1. **模型文件不存在**
   ```
   ❌ 错误: 模型文件不存在: /path/to/model.pth
   ```
   解决：检查模型路径是否正确

2. **点云文件不存在**
   ```
   ❌ 错误: 点云文件不存在: /path/to/cloud.ply
   ```
   解决：检查点云文件路径是否正确

3. **GPU内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   解决：减小`--num-points`或使用CPU(`--device cpu`)

4. **扰动值格式错误**
   ```
   ❌ 扰动值必须是6个数字，当前提供了X个
   ```
   解决：确保扰动值格式为"rx,ry,rz,tx,ty,tz"

### 性能优化建议

1. **使用GPU加速**: `--device cuda:0`
2. **合理设置点云数量**: 1024是平衡精度和速度的好选择
3. **调整最大迭代次数**: 根据精度需求调整`--max-iter`
4. **禁用不必要的输出**: 不需要时不要使用`--save-clouds`和`--verbose`

## 结果分析

### 评估指标
- **旋转误差 < 1°**: 优秀
- **旋转误差 1-5°**: 良好  
- **旋转误差 > 5°**: 需要改进

- **平移误差 < 1mm**: 优秀
- **平移误差 1-3mm**: 良好
- **平移误差 > 3mm**: 需要改进

### 影响因素
1. **点云质量**: 点云密度、噪声水平
2. **扰动大小**: 扰动越大，配准越困难
3. **场景复杂度**: 纹理丰富度、几何特征
4. **模型训练质量**: 训练数据分布、模型收敛程度 
 