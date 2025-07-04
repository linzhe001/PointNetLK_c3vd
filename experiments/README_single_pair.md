# 单对点云配准功能说明 Single Pair Point Cloud Registration Documentation

## 概述 Overview

这个功能允许你使用训练好的PointNet-LK模型对单对点云进行配准测试，支持自定义扰动和详细的结果输出。数据处理方法与C3VD数据集保持完全一致，包括体素化和智能降采样等操作。

This feature allows you to perform registration testing on a single pair of point clouds using trained PointNet-LK models, with support for custom perturbations and detailed result output. The data processing methods are exactly the same as the C3VD dataset, including voxelization and intelligent downsampling operations.

## 功能特点 Features

- ✅ **单对点云输入**: 支持直接输入源点云和目标点云文件
- ✅ **自定义扰动**: 支持6维扰动向量 (rx,ry,rz,tx,ty,tz)
- ✅ **C3VD兼容处理**: 使用与C3VD完全相同的数据处理管道
- ✅ **体素化支持**: 智能体素化和降采样，提高配准质量
- ✅ **多模型支持**: 支持PointNet, Attention, Mamba3D, Fast Attention, CFormer等模型
- ✅ **详细输出**: 输出传入的扰动、预测的变换、配准误差等详细信息
- ✅ **批量测试**: 通过脚本支持多个扰动的批量测试

## 使用方法 Usage

### 方法1: 直接命令行调用 Direct Command Line

```bash
cd /SAN/medic/MRpcr/PointNetLK_c3vd/experiments
source /SAN/medic/MRpcr/miniconda3/etc/profile.d/conda.sh
conda activate pointlk

python test_pointlk.py \
    --single-pair-mode \
    --source-cloud "/path/to/source.ply" \
    --target-cloud "/path/to/target.ply" \
    --single-perturbation "0.05,0.02,0.0,0.005,0.01,0.0" \
    --enhanced-output \
    --outfile "/path/to/output.csv" \
    --dataset-path "/SAN/medic/MRpcr/C3VD_datasets" \
    --categoryfile "/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/c3vd.txt" \
    --model-type pointnet \
    --pretrained "/path/to/model.pth" \
    --device cuda:0 \
    --num-points 1024 \
    --use-voxelization \
    --voxel-size 4 \
    --voxel-grid-size 32 \
    --max-voxel-points 100 \
    --max-voxels 20000 \
    --min-voxel-points-ratio 0.1
```

### 方法2: 使用提供的测试脚本 Using Provided Test Script

```bash
# 编辑脚本中的配置
vim test_single_pair.sh

# 修改以下变量：
# - SOURCE_CLOUD: 源点云文件路径
# - TARGET_CLOUD: 目标点云文件路径  
# - MODEL_TYPE: 模型类型 (pointnet, attention, cformer等)
# - MODEL_PATH: 模型文件路径

# 运行测试
./test_single_pair.sh
```

## 参数说明 Parameters

### 必需参数 Required Parameters

| 参数 Parameter | 说明 Description |
|----------------|------------------|
| `--single-pair-mode` | 启用单对点云模式 |
| `--source-cloud` | 源点云文件路径 (.ply格式) |
| `--target-cloud` | 目标点云文件路径 (.ply格式) |
| `--single-perturbation` | 扰动值，格式: "rx,ry,rz,tx,ty,tz" |
| `--outfile` | 输出文件路径 |
| `--dataset-path` | 数据集路径 (用于兼容性) |
| `--categoryfile` | 类别文件路径 (用于兼容性) |
| `--pretrained` | 预训练模型文件路径 |

### 可选参数 Optional Parameters

| 参数 Parameter | 默认值 Default | 说明 Description |
|----------------|---------------|------------------|
| `--enhanced-output` | False | 输出详细信息到文件 |
| `--model-type` | pointnet | 模型类型 (pointnet/attention/cformer等) |
| `--device` | cpu | 计算设备 (cpu/cuda:0) |
| `--num-points` | 1024 | 点云点数 |
| `--use-voxelization` | True | 启用体素化处理 |
| `--voxel-size` | 4 | 体素大小 |
| `--voxel-grid-size` | 32 | 体素网格尺寸 |
| `--max-voxel-points` | 100 | 每个体素最大点数 |
| `--max-voxels` | 20000 | 最大体素数量 |
| `--min-voxel-points-ratio` | 0.1 | 最小体素点数比例 |

## 扰动格式 Perturbation Format

扰动值是一个6维向量，格式为: `"rx,ry,rz,tx,ty,tz"`

- `rx, ry, rz`: 绕X、Y、Z轴的旋转 (弧度)
- `tx, ty, tz`: 沿X、Y、Z轴的平移

示例 Examples:
- `"0.1,0.0,0.0,0.0,0.0,0.0"` - 仅绕X轴旋转0.1弧度
- `"0.0,0.0,0.0,0.1,0.0,0.0"` - 仅沿X轴平移0.1单位
- `"0.05,0.02,0.0,0.005,0.01,0.0"` - 复合变换

## 输出格式 Output Format

### 控制台输出 Console Output

```
========== 配准结果 Registration Results ==========
传入的扰动 Input Perturbation:
  向量形式: [0.050000, 0.020000, 0.000000, 0.005000, 0.010000, 0.000000]
  旋转部分 (rx,ry,rz): [0.050000, 0.020000, 0.000000]
  平移部分 (tx,ty,tz): [0.005000, 0.010000, 0.000000]

预测的变换 Predicted Transformation:
  变换矩阵:
    [ 0.998436,  0.019893, -0.052036,  0.004721]
    [-0.019998,  0.999800, -0.000996,  0.009823]
    [ 0.052010,  0.001996,  0.998646, -0.000123]
    [ 0.000000,  0.000000,  0.000000,  1.000000]
  扭转向量形式: [0.049876, 0.019994, 0.000123, 0.004721, 0.009823, -0.000123]
  旋转部分: [0.049876, 0.019994, 0.000123]
  平移部分: [0.004721, 0.009823, -0.000123]

配准误差 Registration Error:
  旋转误差: 0.000124 弧度 = 0.007089 度
  平移误差: 0.000279
```

### 文件输出 File Output (--enhanced-output)

```csv
# 单对点云配准结果 Single Pair Point Cloud Registration Results
# 源点云: /path/to/source.ply
# 目标点云: /path/to/target.ply
# 模型类型: pointnet
# 处理时间: 2024-01-01 12:00:00

# 传入的扰动 Input Perturbation
input_perturbation_vector,0.050000,0.020000,0.000000,0.005000,0.010000,0.000000
input_rotation_part,0.050000,0.020000,0.000000
input_translation_part,0.005000,0.010000,0.000000

# 预测的变换 Predicted Transformation
predicted_twist_vector,0.049876,0.019994,0.000123,0.004721,0.009823,-0.000123
predicted_rotation_part,0.049876,0.019994,0.000123
predicted_translation_part,0.004721,0.009823,-0.000123

# 预测变换矩阵 Predicted Transformation Matrix
transformation_matrix_row_0,0.998436,0.019893,-0.052036,0.004721
transformation_matrix_row_1,-0.019998,0.999800,-0.000996,0.009823
transformation_matrix_row_2,0.052010,0.001996,0.998646,-0.000123
transformation_matrix_row_3,0.000000,0.000000,0.000000,1.000000

# 配准误差 Registration Error
rotation_error_rad,0.000124
rotation_error_deg,0.007089
translation_error,0.000279
```

## 数据处理流程 Data Processing Pipeline

1. **点云读取**: 读取PLY格式的源点云和目标点云
2. **数据清理**: 移除无效点（NaN、Inf等）
3. **体素化处理**: 
   - 计算点云重叠区域
   - 体素化表示
   - 智能交集采样
4. **归一化**: 对每个点云单独进行归一化
5. **扰动应用**: 对源点云应用指定的刚性变换
6. **配准预测**: 使用模型预测变换矩阵
7. **误差计算**: 计算旋转和平移误差

## 模型支持 Model Support

| 模型类型 Model Type | 参数配置 Parameters |
|-------------------|-------------------|
| PointNet | `--model-type pointnet` |
| Attention | `--model-type attention --num-attention-blocks 3 --num-heads 8` |
| Mamba3D | `--model-type mamba3d --num-mamba-blocks 3 --d-state 16` |
| Fast Attention | `--model-type fast_attention --num-fast-attention-blocks 2` |
| CFormer | `--model-type cformer --num-proxy-points 8 --num-blocks 2` |

## 故障排除 Troubleshooting

### 常见错误 Common Errors

1. **文件不存在**: 检查点云文件路径是否正确
2. **模型文件不存在**: 检查预训练模型路径是否正确
3. **GPU内存不足**: 减少`--num-points`或使用CPU (`--device cpu`)
4. **点云格式错误**: 确保使用PLY格式且包含有效的xyz坐标

### 性能优化 Performance Optimization

1. **体素化配置**: 调整`--voxel-size`和`--voxel-grid-size`以平衡速度和精度
2. **点云数量**: 减少`--num-points`可以提高速度
3. **GPU使用**: 使用`--device cuda:0`可以显著提高速度

## 示例 Examples

### 示例1: 基础测试 Basic Test

```bash
python test_pointlk.py \
    --single-pair-mode \
    --source-cloud "/SAN/medic/MRpcr/C3VD_datasets/C3VD_ply_source/cecum_t1_a/0000_depth_pcd.ply" \
    --target-cloud "/SAN/medic/MRpcr/C3VD_datasets/visible_point_cloud_ply_depth/cecum_t1_a/frame_0000_visible.ply" \
    --single-perturbation "0.1,0.0,0.0,0.0,0.0,0.0" \
    --outfile "test_result.csv" \
    --dataset-path "/SAN/medic/MRpcr/C3VD_datasets" \
    --categoryfile "/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/c3vd.txt" \
    --pretrained "/SAN/medic/MRpcr/results/pointnet_c3vd/pointnet_pointlk_resume_0620_model_best.pth" \
    --device cuda:0
```

### 示例2: 使用CFormer模型 Using CFormer Model

```bash
python test_pointlk.py \
    --single-pair-mode \
    --source-cloud "source.ply" \
    --target-cloud "target.ply" \
    --single-perturbation "0.05,0.02,0.0,0.005,0.01,0.0" \
    --enhanced-output \
    --outfile "cformer_result.csv" \
    --dataset-path "/SAN/medic/MRpcr/C3VD_datasets" \
    --categoryfile "/SAN/medic/MRpcr/PointNetLK_c3vd/experiments/sampledata/c3vd.txt" \
    --model-type cformer \
    --num-proxy-points 8 \
    --num-blocks 2 \
    --pretrained "/SAN/medic/MRpcr/results/cformer_c3vd/cformer_pointlk_resume_0620_model_best.pth" \
    --device cuda:0
```

## 兼容性说明 Compatibility Notes

- 此功能完全兼容现有的批量测试脚本
- 数据处理管道与C3VD训练保持一致
- 支持所有现有的模型类型和参数配置
- 不影响原有的测试功能

## 更新历史 Update History

- **v1.0**: 初始版本，支持单对点云输入和自定义扰动
- 添加了体素化处理和智能采样
- 支持详细的结果输出和误差计算
- 提供了便捷的测试脚本 