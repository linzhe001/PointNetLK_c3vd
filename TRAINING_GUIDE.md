# PointNetLK_c3vd 训练指南

## 概述

这个指南说明了如何使用体素化功能和从已有权重恢复训练的PointNetLK_c3vd项目。

## 1. 体素化功能概述

### 1.1 变换逻辑检查结果

✅ **变换逻辑完全保持不变**：
- 联合归一化：计算两个点云的联合边界框进行归一化
- 刚性变换应用：`transformed_source = self.rigid_transform(source_normalized)` 
- 变换矩阵获取：`igt = self.rigid_transform.igt`
- 体素化只在**数据预处理阶段**（归一化之前）添加智能采样

### 1.2 体素化配置参数

在 `train_cformer_c3vd.sh` 中新增的体素化参数：

```bash
# 体素化配置参数
USE_VOXELIZATION=true           # 是否启用体素化（true/false）
VOXEL_SIZE=0.05                 # 体素大小
VOXEL_GRID_SIZE=32              # 体素网格尺寸
MAX_VOXEL_POINTS=100            # 每个体素最大点数
MAX_VOXELS=20000                # 最大体素数量
MIN_VOXEL_POINTS_RATIO=0.1      # 最小体素点数比例
```

### 1.3 体素化命令行参数

在 `train_pointlk.py` 中可使用的体素化参数：

```bash
--use-voxelization              # 启用体素化预处理方法 (默认: True)
--no-voxelization              # 禁用体素化，使用简单重采样方法
--voxel-size 0.05              # 体素大小 (默认: 0.05)
--voxel-grid-size 32           # 体素网格尺寸 (默认: 32)
--max-voxel-points 100         # 每个体素最大点数 (默认: 100)
--max-voxels 20000             # 最大体素数量 (默认: 20000)
--min-voxel-points-ratio 0.1   # 最小体素点数比例阈值 (默认: 0.1)
```

## 2. 从已有权重恢复训练

### 2.1 支持的恢复方式

`train_pointlk.py` 支持两种方式加载已有权重：

#### 方式1：从checkpoint恢复完整训练状态
```bash
python train_pointlk.py [其他参数] --resume /path/to/checkpoint.pth
```
- 恢复模型权重、优化器状态、训练轮次、最佳损失等完整状态
- 适合意外中断后继续训练

#### 方式2：使用预训练模型权重
```bash
python train_pointlk.py [其他参数] --pretrained /path/to/model.pth
```
- 仅加载模型权重，其他训练状态从头开始
- 适合使用预训练分类器权重初始化PointLK训练

### 2.2 权重文件格式

支持两种checkpoint格式：

1. **完整checkpoint格式**（包含训练状态）：
```python
{
    'model': model_state_dict,
    'optimizer': optimizer_state_dict,
    'epoch': current_epoch,
    'min_loss': best_loss,
    'best_epoch': best_epoch
}
```

2. **简单模型权重格式**：
```python
model_state_dict  # 仅包含模型权重
```

## 3. 使用示例

### 3.1 启用体素化的新训练

```bash
python train_pointlk.py \
  -o /path/to/output \
  -i /path/to/C3VD_datasets \
  -c /path/to/c3vd.txt \
  --dataset-type c3vd \
  --num-points 1024 \
  --epochs 200 \
  --batch-size 16 \
  --device cuda:0 \
  --model-type cformer \
  --use-voxelization \
  --voxel-size 0.05 \
  --voxel-grid-size 32
```

### 3.2 从checkpoint恢复训练

```bash
python train_pointlk.py \
  -o /path/to/output \
  -i /path/to/C3VD_datasets \
  -c /path/to/c3vd.txt \
  --dataset-type c3vd \
  --resume /path/to/checkpoint.pth \
  --use-voxelization
```

### 3.3 使用预训练分类器权重

```bash
python train_pointlk.py \
  -o /path/to/output \
  -i /path/to/C3VD_datasets \
  -c /path/to/c3vd.txt \
  --dataset-type c3vd \
  --pretrained /path/to/classifier_model_best.pth \
  --use-voxelization
```

### 3.4 禁用体素化使用传统方法

```bash
python train_pointlk.py \
  -o /path/to/output \
  -i /path/to/C3VD_datasets \
  -c /path/to/c3vd.txt \
  --dataset-type c3vd \
  --no-voxelization
```

## 4. train_cformer_c3vd.sh 脚本使用

### 4.1 脚本功能

- **两阶段训练**：先训练分类器，再训练PointLK配准
- **自动权重传递**：分类器权重自动传递给PointLK训练
- **体素化支持**：完整的体素化参数配置
- **场景分割**：PointLK使用场景分割进行数据划分

### 4.2 配置体素化

在脚本中修改体素化配置：

```bash
# 启用体素化
USE_VOXELIZATION=true
VOXEL_SIZE=0.05
VOXEL_GRID_SIZE=32

# 禁用体素化
USE_VOXELIZATION=false
```

### 4.3 运行脚本

```bash
# 直接运行
./train_cformer_c3vd.sh

# 或提交到集群
qsub train_cformer_c3vd.sh
```

## 5. 体素化技术优势

### 5.1 智能预处理
- 基于体素网格保持点云空间结构
- 重点采样两个点云的重叠区域
- 提升配准任务的性能

### 5.2 鲁棒性
- 自动回退机制：体素化失败时自动使用重采样
- 错误处理：处理边界情况和无效数据
- 完全兼容：不破坏原有训练流程

### 5.3 可配置性
- 灵活的体素化参数调整
- 支持运行时启用/禁用
- 保持向后兼容性

## 6. 最佳实践

### 6.1 参数调优建议

- **体素大小**：根据点云密度调整，密集点云使用较小值
- **网格尺寸**：平衡精度和计算效率，通常32-64效果较好
- **最大体素点数**：根据内存和性能需求调整

### 6.2 训练策略

1. **首次训练**：使用预训练分类器权重初始化
2. **恢复训练**：使用`--resume`从checkpoint恢复
3. **参数实验**：先用小数据集测试体素化参数效果
4. **性能监控**：观察训练日志中的体素化处理信息

### 6.3 故障排除

- **体素化失败**：检查点云质量和体素化参数设置
- **内存不足**：减少`max_voxels`或`max_voxel_points`
- **训练中断**：使用`--resume`从最新checkpoint恢复

## 7. 文件结构

```
PointNetLK_c3vd/
├── ptlk/data/datasets.py          # 包含体素化功能的数据集类
├── experiments/
│   ├── train_pointlk.py           # 支持体素化和恢复训练的训练脚本
│   └── train_cformer_c3vd.sh      # 带体素化参数的完整训练脚本
└── TRAINING_GUIDE.md              # 本指南文档
```

## 8. 技术细节

### 8.1 体素化流程

1. **重叠区域检测**：计算两个点云的重叠边界框
2. **体素网格划分**：将重叠区域划分为3D体素网格
3. **点云体素化**：将点分配到对应体素
4. **交集计算**：找到两个点云共同占用的体素
5. **智能采样**：优先从交集体素采样，不足时补充其他点

### 8.2 后备机制

- 体素化失败时自动切换到传统重采样
- 保证训练的连续性和稳定性
- 详细的错误日志和警告信息

---

## 联系信息

如有问题或建议，请查看项目文档或联系维护者。 