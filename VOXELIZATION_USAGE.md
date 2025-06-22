# 体素化功能使用说明

## 概述

已成功将 `PointNetLK_com` 项目中的体素化方法移植到 `PointNetLK_c3vd` 项目中，用于替换原有的简单重采样方法。体素化方法能够更好地保持点云的空间结构和细节信息。

## 主要改进

### 1. 数据预处理方法升级
- **原方法**: 简单的随机重采样
- **新方法**: 基于体素化的智能采样，保持点云空间结构

### 2. 新增功能
- 体素网格划分
- 重叠区域检测
- 智能点云采样
- 后备随机采样（当体素化失败时）

## 使用方法

### 训练时启用体素化

在训练脚本 `train_pointlk.py` 中，现在支持以下新参数：

```bash
# 启用体素化（默认）
python experiments/train_pointlk.py \
    --use-voxelization \
    --voxel-size 0.05 \
    --voxel-grid-size 32 \
    --max-voxel-points 100 \
    --max-voxels 20000 \
    --min-voxel-points-ratio 0.1 \
    [其他参数...]

# 禁用体素化，使用简单重采样
python experiments/train_pointlk.py \
    --no-voxelization \
    [其他参数...]
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--use-voxelization` | True | 启用体素化预处理 |
| `--no-voxelization` | - | 禁用体素化，使用简单重采样 |
| `--voxel-size` | 0.05 | 体素大小 |
| `--voxel-grid-size` | 32 | 体素网格尺寸 (32x32x32) |
| `--max-voxel-points` | 100 | 每个体素最大点数 |
| `--max-voxels` | 20000 | 最大体素数量 |
| `--min-voxel-points-ratio` | 0.1 | 最小体素点数比例阈值 |

## 核心类和函数

### 1. VoxelizationConfig 类
```python
voxel_config = VoxelizationConfig(
    voxel_size=0.05,
    voxel_grid_size=32,
    max_voxel_points=100,
    max_voxels=20000,
    min_voxel_points_ratio=0.1
)
```

### 2. C3VDset4tracking 类更新
- 新增 `use_voxelization` 参数
- 新增 `voxel_config` 参数
- 自动回退机制：体素化失败时使用重采样

### 3. C3VDset4tracking_test 类更新
- 继承父类的体素化功能
- 保持测试功能完整性

## 体素化处理流程

1. **数据清理**: 移除 NaN/Inf 值
2. **重叠区域计算**: 找到两个点云的重叠边界框
3. **体素化**: 将点云划分到体素网格中
4. **智能采样**: 从体素中采样点，优先保留重叠区域
5. **后备处理**: 如果体素化失败，自动回退到随机采样

## 优势

1. **空间结构保持**: 体素化能更好地保持点云的空间分布
2. **重叠区域优化**: 重点采样两个点云的重叠区域，提高配准效果
3. **鲁棒性**: 具备后备机制，确保在各种情况下都能正常工作
4. **兼容性**: 完全兼容原有的数据加载和训练流程

## 向后兼容

- 所有原有的训练脚本和参数仍然有效
- 默认启用体素化，但可以通过 `--no-voxelization` 禁用
- 保持与原有数据集和模型的完全兼容

## 示例使用

### 基本训练（启用体素化）
```bash
python experiments/train_pointlk.py \
    -o model_output \
    -i /path/to/C3VD_datasets \
    -c /path/to/categories.txt \
    --dataset-type c3vd \
    --num-points 1024 \
    --epochs 100
```

### 自定义体素化参数
```bash
python experiments/train_pointlk.py \
    -o model_output \
    -i /path/to/C3VD_datasets \
    -c /path/to/categories.txt \
    --dataset-type c3vd \
    --voxel-size 0.02 \
    --voxel-grid-size 64 \
    --max-voxel-points 50
```

## 注意事项

1. 体素化会增加一些计算开销，但能提供更好的数据质量
2. 对于点云密度较低的数据，建议减小 `voxel-size` 参数
3. 如果遇到内存问题，可以减少 `max-voxels` 或 `max-voxel-points` 参数
4. 体素化失败时会自动回退到重采样，不会中断训练过程

## 总结

体素化功能的成功移植为 `PointNetLK_c3vd` 项目提供了更先进的数据预处理能力，预期能够提升点云配准的性能和稳定性。 