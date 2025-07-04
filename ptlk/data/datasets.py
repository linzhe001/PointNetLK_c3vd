""" datasets """

import numpy
import torch
import torch.utils.data
import os
import glob
from plyfile import PlyData
import numpy as np
# 新增导入：用于体素化处理
import copy
import six

from . import globset
from . import mesh
from . import transforms
from .. import so3
from .. import se3

# 从mesh模块导入plyread函数
try:
    from .mesh import plyread
except ImportError:
    print("Warning: plyread not available from mesh module")

# 添加体素化相关函数（从PointNetLK_com移植）
def find_voxel_overlaps(p0, p1, voxel):
    """计算两个点云的重叠边界框和体素参数"""
    # 计算两个点云的重叠边界框
    xmin, ymin, zmin = np.max(np.stack([np.min(p0, 0), np.min(p1, 0)]), 0)
    xmax, ymax, zmax = np.min(np.stack([np.max(p0, 0), np.max(p1, 0)]), 0)
    
    # 检查是否有有效的重叠区域
    if xmin >= xmax or ymin >= ymax or zmin >= zmax:
        print(f"警告: 点云无重叠区域 - 边界框: x[{xmin:.3f}, {xmax:.3f}], y[{ymin:.3f}, {ymax:.3f}], z[{zmin:.3f}, {zmax:.3f}]")
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), xmin, ymin, zmin, xmax, ymax, zmax, 0.1, 0.1, 0.1
    
    # 裁剪点云到重叠区域
    eps = 1e-6
    p0_ = p0[np.all(p0>[xmin+eps,ymin+eps,zmin+eps], axis=1) & np.all(p0<[xmax-eps,ymax-eps,zmax-eps], axis=1)]
    p1_ = p1[np.all(p1>[xmin+eps,ymin+eps,zmin+eps], axis=1) & np.all(p1<[xmax-eps,ymax-eps,zmax-eps], axis=1)]
    
    # 检查裁剪后的点云是否为空
    if len(p0_) == 0 or len(p1_) == 0:
        print(f"警告: 裁剪后点云为空 - 源点云: {len(p0_)}, 目标点云: {len(p1_)}")
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3), xmin, ymin, zmin, xmax, ymax, zmax, 0.1, 0.1, 0.1
    
    # 使用裁剪后的点云重新计算更精确的边界框
    xmin_refined, ymin_refined, zmin_refined = np.max(np.stack([np.min(p0_, 0), np.min(p1_, 0)]), 0)
    xmax_refined, ymax_refined, zmax_refined = np.min(np.stack([np.max(p0_, 0), np.max(p1_, 0)]), 0)
    
    # 计算体素大小
    vx = (xmax_refined - xmin_refined) / voxel
    vy = (ymax_refined - ymin_refined) / voxel
    vz = (zmax_refined - zmin_refined) / voxel
    
    return p0_, p1_, xmin_refined, ymin_refined, zmin_refined, xmax_refined, ymax_refined, zmax_refined, vx, vy, vz


def _points_to_voxel_kernel(points,
                            voxel_size,
                            coords_range,
                            num_points_per_voxel,
                            coor_to_voxelidx,
                            voxels,
                            coors,
                            max_points=35,
                            max_voxels=20000):
    """体素化核心函数"""
    N = points.shape[0]
    ndim = 3
    grid_size = (coords_range[3:] - coords_range[:3]) / voxel_size
    grid_size = np.around(grid_size, 0, grid_size).astype(np.int32)

    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coords_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            voxelidx = voxel_num
            if voxel_num > max_voxels:
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            num_points_per_voxel[voxelidx] += 1

    return voxel_num


def points_to_voxel_second(points,
                     coords_range,
                     voxel_size,
                     max_points=100,
                     reverse_index=False,
                     max_voxels=20000):
    """将点云转换为体素表示"""
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coords_range, np.ndarray):
        coords_range = np.array(coords_range, dtype=points.dtype)
    voxelmap_shape = (coords_range[3:] - coords_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.around(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
        
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.ones(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype) * np.mean(points, 0)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    voxel_num = _points_to_voxel_kernel(
        points, voxel_size, coords_range, num_points_per_voxel,
        coor_to_voxelidx, voxels, coors, max_points, max_voxels)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]

    return voxels, coors, num_points_per_voxel


class VoxelizationConfig:
    """体素化配置类"""
    def __init__(self, voxel_size=0.05, voxel_grid_size=32, max_voxel_points=100, 
                 max_voxels=20000, min_voxel_points_ratio=0.1):
        self.voxel_size = voxel_size
        self.voxel_grid_size = voxel_grid_size
        self.max_voxel_points = max_voxel_points
        self.max_voxels = max_voxels
        self.min_voxel_points_ratio = min_voxel_points_ratio


def voxel_grid_centroid_downsample(points, voxel_size=0.05, target_points=None):
    """
    标准体素网格质心降采样 (Voxel Grid Centroid Downsampling)
    这是最经典、最常用的体素降采样方法。
    
    工作原理：
    1. 在整个点云空间上创建一个三维体素网格（Voxel Grid），网格大小由 voxel_size 参数决定。
    2. 遍历所有点，将它们分配到各自所在的体素中。
    3. 对于每一个包含一个或多个点的非空体素，计算体素内所有点的质心（平均坐标）。
    4. 用这个计算出的质心点来代表该体素，作为降采样后的新点。
    
    Args:
        points: 输入点云 (numpy array, shape: [N, 3])
        voxel_size: 体素大小
        target_points: 目标点数（如果设置，会通过调整体素大小来尽量达到目标点数）
    
    Returns:
        downsampled_points: 降采样后的点云 (numpy array, shape: [M, 3])
    """
    if len(points) == 0:
        return np.array([]).reshape(0, 3)
    
    # 如果指定了目标点数，尝试自动调整体素大小
    if target_points is not None and target_points > 0:
        # 估算合适的体素大小以达到目标点数
        bbox_size = np.max(points, axis=0) - np.min(points, axis=0)
        volume = np.prod(bbox_size)
        
        # 根据点云密度估算体素大小
        # 假设目标是将点云降采样到目标点数
        estimated_voxel_volume = volume / target_points
        estimated_voxel_size = np.cbrt(estimated_voxel_volume)
        
        # 限制体素大小的范围，避免过大或过小
        voxel_size = np.clip(estimated_voxel_size, voxel_size * 0.1, voxel_size * 10)
    
    # 计算点云的边界框
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)
    
    # 计算每个点对应的体素索引
    voxel_indices = np.floor((points - min_bound) / voxel_size).astype(np.int32)
    
    # 创建体素索引的唯一标识符
    # 使用字典存储每个体素中的点
    voxel_dict = {}
    
    for i, voxel_idx in enumerate(voxel_indices):
        # 将体素索引转换为唯一的键
        key = tuple(voxel_idx)
        
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(points[i])
    
    # 计算每个非空体素的质心
    centroids = []
    for voxel_points in voxel_dict.values():
        if len(voxel_points) > 0:
            # 计算体素内所有点的质心（平均坐标）
            centroid = np.mean(voxel_points, axis=0)
            centroids.append(centroid)
    
    if len(centroids) == 0:
        # 如果没有有效的体素，返回原始点云的第一个点
        return points[:1] if len(points) > 0 else np.array([]).reshape(0, 3)
    
    downsampled_points = np.array(centroids)
    
    # 如果指定了目标点数且降采样后的点数不足，进行补充
    if target_points is not None and len(downsampled_points) < target_points:
        # 使用随机重复采样补足
        shortage = target_points - len(downsampled_points)
        if shortage > 0:
            indices = np.random.choice(len(downsampled_points), shortage, replace=True)
            repeated_points = downsampled_points[indices]
            downsampled_points = np.concatenate([downsampled_points, repeated_points], axis=0)
    elif target_points is not None and len(downsampled_points) > target_points:
        # 如果点数过多，随机选择目标数量的点
        indices = np.random.choice(len(downsampled_points), target_points, replace=False)
        downsampled_points = downsampled_points[indices]
    
    return downsampled_points


def adaptive_voxel_grid_downsample(points, target_points, initial_voxel_size=0.05, max_iterations=10):
    """
    自适应体素网格降采样，自动调整体素大小以达到目标点数
    
    Args:
        points: 输入点云
        target_points: 目标点数
        initial_voxel_size: 初始体素大小
        max_iterations: 最大迭代次数
    
    Returns:
        降采样后的点云
    """
    if len(points) <= target_points:
        return points
    
    voxel_size = initial_voxel_size
    
    for iteration in range(max_iterations):
        # 尝试当前体素大小
        downsampled = voxel_grid_centroid_downsample(points, voxel_size=voxel_size)
        current_count = len(downsampled)
        
        # 如果点数接近目标（误差在20%以内），则接受结果
        if abs(current_count - target_points) / target_points <= 0.2:
            # 如果点数不足，补充到目标点数
            if current_count < target_points:
                shortage = target_points - current_count
                indices = np.random.choice(current_count, shortage, replace=True)
                repeated_points = downsampled[indices]
                downsampled = np.concatenate([downsampled, repeated_points], axis=0)
            elif current_count > target_points:
                # 如果点数过多，随机选择
                indices = np.random.choice(current_count, target_points, replace=False)
                downsampled = downsampled[indices]
            
            return downsampled
        
        # 调整体素大小
        if current_count > target_points:
            # 点数太多，增大体素大小
            voxel_size *= 1.5
        else:
            # 点数太少，减小体素大小
            voxel_size *= 0.7
        
        # 避免体素大小过小或过大
        voxel_size = np.clip(voxel_size, 0.001, 1.0)
    
    # 如果迭代结束仍未达到理想效果，使用最后一次的结果并补充到目标点数
    final_downsampled = voxel_grid_centroid_downsample(points, voxel_size=voxel_size)
    if len(final_downsampled) < target_points:
        shortage = target_points - len(final_downsampled)
        indices = np.random.choice(len(final_downsampled), shortage, replace=True)
        repeated_points = final_downsampled[indices]
        final_downsampled = np.concatenate([final_downsampled, repeated_points], axis=0)
    elif len(final_downsampled) > target_points:
        indices = np.random.choice(len(final_downsampled), target_points, replace=False)
        final_downsampled = final_downsampled[indices]
    
    return final_downsampled


def joint_normalization(source_tensor, target_tensor):
    """
    对源点云和目标点云进行联合归一化
    使用相同的中心点和缩放因子，保持两个点云之间的相对空间关系
    
    Args:
        source_tensor: 源点云 (torch.Tensor, shape: [N, 3])
        target_tensor: 目标点云 (torch.Tensor, shape: [M, 3])
    
    Returns:
        source_normalized, target_normalized: 联合归一化后的点云
    """
    # 合并两个点云来计算联合的边界框
    combined_points = torch.cat([source_tensor, target_tensor], dim=0)
    
    # 计算联合的最小值和最大值
    combined_min_vals = combined_points.min(dim=0)[0]
    combined_max_vals = combined_points.max(dim=0)[0]
    
    # 计算联合的中心点和缩放因子
    combined_center = (combined_min_vals + combined_max_vals) / 2
    combined_scale = (combined_max_vals - combined_min_vals).max()
    
    if combined_scale < 1e-10:
        raise ValueError(f"联合归一化尺度过小: {combined_scale}")
    
    # 使用相同的中心点和缩放因子对两个点云进行归一化
    source_normalized = (source_tensor - combined_center) / combined_scale
    target_normalized = (target_tensor - combined_center) / combined_scale
    
    return source_normalized, target_normalized


def voxelize_point_clouds(source_points, target_points, num_points, voxel_config=None, 
                         fallback_to_sampling=True):
    """
    对点云对进行体素化处理，如果失败则回退到随机采样
    
    Args:
        source_points: 源点云 (numpy array)
        target_points: 目标点云 (numpy array) 
        num_points: 目标点数
        voxel_config: 体素化配置
        fallback_to_sampling: 是否在体素化失败时回退到随机采样
    
    Returns:
        final_source, final_target: 处理后的点云
    """
    if voxel_config is None:
        voxel_config = VoxelizationConfig()
    
    try:
        # 1. 寻找重叠区域并计算体素网格参数
        source_overlap, target_overlap, xmin, ymin, zmin, xmax, ymax, zmax, vx, vy, vz = \
            find_voxel_overlaps(source_points, target_points, voxel_config.voxel_grid_size)
        
        if len(source_overlap) == 0 or len(target_overlap) == 0:
            if fallback_to_sampling:
                print("警告: 点云无重叠区域，回退到随机采样")
                return fallback_random_sampling(source_points, target_points, num_points)
            else:
                raise ValueError("点云无重叠区域")
        
        # 2. 体素化转换
        coords_range = (xmin, ymin, zmin, xmax, ymax, zmax)
        voxel_size = (vx, vy, vz)
        
        # 对源点云体素化
        source_voxels, source_coords, source_num_points = points_to_voxel_second(
            source_overlap, coords_range, voxel_size,
            max_points=voxel_config.max_voxel_points,
            max_voxels=voxel_config.max_voxels
        )
        
        # 对目标点云体素化
        target_voxels, target_coords, target_num_points = points_to_voxel_second(
            target_overlap, coords_range, voxel_size,
            max_points=voxel_config.max_voxel_points,
            max_voxels=voxel_config.max_voxels
        )
        
        # 3. 体素筛选和交集计算
        intersection_indices = compute_voxel_intersection(
            source_coords, target_coords, source_num_points, target_num_points, voxel_config
        )
        
        # 4. 智能采样策略
        final_source = smart_sampling_from_voxels(
            source_voxels, source_coords, source_num_points, 
            intersection_indices, num_points, voxel_config
        )
        
        final_target = smart_sampling_from_voxels(
            target_voxels, target_coords, target_num_points,
            intersection_indices, num_points, voxel_config
        )
        
        return final_source, final_target
        
    except Exception as e:
        if fallback_to_sampling:
            print(f"体素化处理失败: {e}，回退到随机采样")
            return fallback_random_sampling(source_points, target_points, num_points)
        else:
            raise e


def compute_voxel_intersection(source_coords, target_coords, 
                              source_num_points, target_num_points, voxel_config):
    """计算体素交集"""
    # 基于点密度筛选有效体素
    min_points_threshold = int(voxel_config.min_voxel_points_ratio * voxel_config.max_voxel_points)
    
    # 筛选有效体素
    valid_source_mask = source_num_points >= min_points_threshold
    valid_target_mask = target_num_points >= min_points_threshold
    
    # 计算体素索引
    grid_size = voxel_config.voxel_grid_size
    source_indices = (source_coords[:, 1] * (grid_size**2) + 
                     source_coords[:, 0] * grid_size + 
                     source_coords[:, 2])
    target_indices = (target_coords[:, 1] * (grid_size**2) + 
                     target_coords[:, 0] * grid_size + 
                     target_coords[:, 2])
    
    # 获取有效体素的索引
    valid_source_indices = source_indices[valid_source_mask]
    valid_target_indices = target_indices[valid_target_mask]
    
    # 计算交集
    intersection_indices, _, _ = np.intersect1d(
        valid_source_indices, valid_target_indices, 
        assume_unique=True, return_indices=True
    )
    
    return intersection_indices


def smart_sampling_from_voxels(voxels, coords, num_points_per_voxel, 
                              intersection_indices, target_points, voxel_config):
    """从体素中智能采样点云"""
    # 1. 提取交集体素的点
    intersection_points = []
    
    if len(intersection_indices) > 0:
        grid_size = voxel_config.voxel_grid_size
        voxel_indices = (coords[:, 1] * (grid_size**2) + 
                        coords[:, 0] * grid_size + 
                        coords[:, 2])
        
        for intersection_idx in intersection_indices:
            # 找到对应的体素
            voxel_mask = voxel_indices == intersection_idx
            
            if np.any(voxel_mask):
                voxel_idx = np.where(voxel_mask)[0][0]
                voxel_points = voxels[voxel_idx]
                num_points = num_points_per_voxel[voxel_idx]
                intersection_points.append(voxel_points[:num_points])
    
    # 合并交集点
    if intersection_points:
        intersection_points = np.concatenate(intersection_points, axis=0)
    else:
        intersection_points = np.array([]).reshape(0, 3)
    
    # 2. 获取非交集体素的点
    non_intersection_points = get_non_intersection_points(
        voxels, coords, num_points_per_voxel, intersection_indices, voxel_config
    )
    
    # 3. 应用采样策略
    num_intersection = len(intersection_points)
    
    if num_intersection >= target_points:
        # 情况1：交集点已经足够，直接从交集中采样
        indices = np.random.choice(num_intersection, target_points, replace=False)
        return intersection_points[indices]
    else:
        # 情况2：交集点不够，保留所有交集点 + 随机采样其他点
        remaining_points = target_points - num_intersection
        
        if len(non_intersection_points) >= remaining_points:
            # 从非交集点中随机采样
            indices = np.random.choice(len(non_intersection_points), 
                                     remaining_points, replace=False)
            sampled_non_intersection = non_intersection_points[indices]
            return np.concatenate([intersection_points, sampled_non_intersection], axis=0)
        else:
            # 非交集点也不够，使用重复采样
            all_points = np.concatenate([intersection_points, non_intersection_points], axis=0)
            shortage = target_points - len(all_points)
            
            if shortage > 0:
                # 重复采样补足
                indices = np.random.choice(len(all_points), shortage, replace=True)
                repeated_points = all_points[indices]
                return np.concatenate([all_points, repeated_points], axis=0)
            else:
                return all_points


def get_non_intersection_points(voxels, coords, num_points_per_voxel, 
                               intersection_indices, voxel_config):
    """获取非交集体素中的所有点"""
    grid_size = voxel_config.voxel_grid_size
    voxel_indices = (coords[:, 1] * (grid_size**2) + 
                    coords[:, 0] * grid_size + 
                    coords[:, 2])
    
    non_intersection_points = []
    
    for i, voxel_idx in enumerate(voxel_indices):
        if voxel_idx not in intersection_indices:
            num_points = num_points_per_voxel[i]
            if num_points > 0:
                non_intersection_points.append(voxels[i][:num_points])
    
    if non_intersection_points:
        return np.concatenate(non_intersection_points, axis=0)
    else:
        return np.array([]).reshape(0, 3)


def fallback_random_sampling(source_points, target_points, target_size):
    """回退到随机采样策略，确保返回固定大小的点云"""
    # 对源点云采样
    if len(source_points) > target_size:
        indices = np.random.choice(len(source_points), target_size, replace=False)
        source_sampled = source_points[indices]
    elif len(source_points) < target_size:
        # 点数不足，重复采样补足
        shortage = target_size - len(source_points)
        indices = np.random.choice(len(source_points), shortage, replace=True)
        repeated_points = source_points[indices]
        source_sampled = np.concatenate([source_points, repeated_points], axis=0)
    else:
        source_sampled = source_points
        
    # 对目标点云采样
    if len(target_points) > target_size:
        indices = np.random.choice(len(target_points), target_size, replace=False)
        target_sampled = target_points[indices]
    elif len(target_points) < target_size:
        # 点数不足，重复采样补足
        shortage = target_size - len(target_points)
        indices = np.random.choice(len(target_points), shortage, replace=True)
        repeated_points = target_points[indices]
        target_sampled = np.concatenate([target_points, repeated_points], axis=0)
    else:
        target_sampled = target_points
        
    return source_sampled, target_sampled


class ModelNet(globset.Globset):
    """ [Princeton ModelNet](http://modelnet.cs.princeton.edu/) """
    def __init__(self, dataset_path, train=1, transform=None, classinfo=None):
        loader = mesh.offread
        if train > 0:
            pattern = 'train/*.off'
        elif train == 0:
            pattern = 'test/*.off'
        else:
            pattern = ['train/*.off', 'test/*.off']
        super().__init__(dataset_path, pattern, loader, transform, classinfo)

class ShapeNet2(globset.Globset):
    """ [ShapeNet](https://www.shapenet.org/) v2 """
    def __init__(self, dataset_path, transform=None, classinfo=None):
        loader = mesh.objread
        pattern = '*/models/model_normalized.obj'
        super().__init__(dataset_path, pattern, loader, transform, classinfo)


class CADset4tracking(torch.utils.data.Dataset):
    def __init__(self, dataset, rigid_transform, source_modifier=None, template_modifier=None):
        self.dataset = dataset
        self.rigid_transform = rigid_transform
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pm, _ = self.dataset[index]
        if self.source_modifier is not None:
            p_ = self.source_modifier(pm)
            p1 = self.rigid_transform(p_)
        else:
            p1 = self.rigid_transform(pm)
        igt = self.rigid_transform.igt

        if self.template_modifier is not None:
            p0 = self.template_modifier(pm)
        else:
            p0 = pm

        # p0: template, p1: source, igt: transform matrix from p0 to p1
        return p0, p1, igt


class CADset4tracking_fixed_perturbation(torch.utils.data.Dataset):
    @staticmethod
    def generate_perturbations(batch_size, mag, randomly=False):
        if randomly:
            amp = torch.rand(batch_size, 1) * mag
        else:
            amp = mag
        x = torch.randn(batch_size, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp
        return x.numpy()

    @staticmethod
    def generate_rotations(batch_size, mag, randomly=False):
        if randomly:
            amp = torch.rand(batch_size, 1) * mag
        else:
            amp = mag
        w = torch.randn(batch_size, 3)
        w = w / w.norm(p=2, dim=1, keepdim=True) * amp
        v = torch.zeros(batch_size, 3)
        x = torch.cat((w, v), dim=1)
        return x.numpy()

    def __init__(self, dataset, perturbation, source_modifier=None, template_modifier=None,
                 fmt_trans=False):
        self.dataset = dataset
        self.perturbation = numpy.array(perturbation) # twist (len(dataset), 6)
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier
        self.fmt_trans = fmt_trans # twist or (rotation and translation)

    def do_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        if not self.fmt_trans:
            # x: twist-vector
            g = se3.exp(x).to(p0) # [1, 4, 4]
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0) # igt: p0 -> p1
        else:
            # x: rotation and translation
            w = x[:, 0:3]
            q = x[:, 3:6]
            R = so3.exp(w).to(p0) # [1, 3, 3]
            g = torch.zeros(1, 4, 4)
            g[:, 3, 3] = 1
            g[:, 0:3, 0:3] = R # rotation
            g[:, 0:3, 3] = q   # translation
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0) # igt: p0 -> p1
        return p1, igt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        twist = torch.from_numpy(numpy.array(self.perturbation[index])).contiguous().view(1, 6)
        pm, _ = self.dataset[index]
        x = twist.to(pm)
        if self.source_modifier is not None:
            p_ = self.source_modifier(pm)
            p1, igt = self.do_transform(p_, x)
        else:
            p1, igt = self.do_transform(pm, x)

        if self.template_modifier is not None:
            p0 = self.template_modifier(pm)
        else:
            p0 = pm

        # p0: template, p1: source, igt: transform matrix from p0 to p1
        return p0, p1, igt


class CADset4tracking_fixed_perturbation_random_sample(torch.utils.data.Dataset):
    """
    专用于gt_poses.csv文件的数据集类
    每个扰动随机选择一个测试样本，总共测试扰动数量次
    Special dataset class for gt_poses.csv files
    Each perturbation randomly selects a test sample, testing perturbation count times total
    """
    def __init__(self, dataset, perturbation, source_modifier=None, template_modifier=None,
                 fmt_trans=False, random_seed=42, num_points=1024, use_voxelization=True, voxel_config=None):
        self.dataset = dataset
        self.perturbation = numpy.array(perturbation) # twist (perturbation_count, 6)
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier
        self.fmt_trans = fmt_trans # twist or (rotation and translation)
        
        # 添加降采样相关属性
        self.num_points = num_points
        self.use_voxelization = use_voxelization
        
        # 设置体素化配置
        if voxel_config is None:
            self.voxel_config = VoxelizationConfig(
                voxel_size=0.05,
                voxel_grid_size=32,
                max_voxel_points=100,
                max_voxels=20000,
                min_voxel_points_ratio=0.1
            )
        else:
            self.voxel_config = voxel_config
        
        # 保留原有的重采样器作为后备
        self.resampler = transforms.Resampler(num_points)
        
        # 保存文件索引和点云对信息的字典
        self.cloud_info = {}
        
        # 收集所有可能的点云对信息
        if hasattr(dataset, 'pairs'):
            # 如果是C3VDDataset，直接使用其pair属性
            self.original_pairs = dataset.pairs
            self.original_pair_scenes = dataset.pair_scenes if hasattr(dataset, 'pair_scenes') else None
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'pairs'):
            # 如果是Subset，访问其底层dataset
            self.original_pairs = dataset.dataset.pairs
            self.original_pair_scenes = dataset.dataset.pair_scenes if hasattr(dataset.dataset, 'pair_scenes') else None
        else:
            print("Warning: Unable to find original point cloud pair information")
            self.original_pairs = None
            self.original_pair_scenes = None
        
        # 设置随机种子以确保可复现性
        # Set random seed for reproducibility
        self.random_seed = random_seed
        numpy.random.seed(random_seed)
        
        # 为每个扰动预先生成随机样本索引
        # Pre-generate random sample indices for each perturbation
        self.sample_indices = numpy.random.randint(0, len(self.dataset), size=len(self.perturbation))
        
        print(f"Random sampling mode activated for gt_poses.csv:")
        print(f"- Total perturbations: {len(self.perturbation)}")
        print(f"- Dataset size: {len(self.dataset)}")
        print(f"- Random seed: {random_seed}")
        print(f"- Target points: {num_points}")
        print(f"- 降采样方法: {'复杂体素化处理' if use_voxelization else '标准体素网格质心降采样'}")

    def do_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        if not self.fmt_trans:
            # x: twist-vector
            g = se3.exp(x).to(p0) # [1, 4, 4]
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0) # igt: p0 -> p1
        else:
            # x: rotation and translation
            w = x[:, 0:3]
            q = x[:, 3:6]
            R = so3.exp(w).to(p0) # [1, 3, 3]
            g = torch.zeros(1, 4, 4)
            g[:, 3, 3] = 1
            g[:, 0:3, 0:3] = R # rotation
            g[:, 0:3, 3] = q   # translation
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0) # igt: p0 -> p1
        return p1, igt

    def __len__(self):
        # 返回扰动数量，而不是数据集大小
        # Return perturbation count, not dataset size
        return len(self.perturbation)

    def __getitem__(self, index):
        # index是扰动索引 (0 到 len(perturbation)-1)
        # index is perturbation index (0 to len(perturbation)-1)
        
        # 使用预先生成的随机样本索引
        # Use pre-generated random sample index
        sample_idx = self.sample_indices[index]
        
        try:
            # 获取原始点云对（从随机选择的样本）
            source, target, _ = self.dataset[sample_idx]  # 忽略原始的igt
            
            # 转换为numpy数组进行处理
            source_np = source.numpy() if isinstance(source, torch.Tensor) else source
            target_np = target.numpy() if isinstance(target, torch.Tensor) else target
            
            # 数据清理：移除无效点
            source_mask = np.isfinite(source_np).all(axis=1)
            target_mask = np.isfinite(target_np).all(axis=1)
            
            source_clean = source_np[source_mask]
            target_clean = target_np[target_mask]
            
            if len(source_clean) < 100 or len(target_clean) < 100:
                raise ValueError(f"点云在样本索引{sample_idx}处清理后点数不足100个")
            
            # 选择处理策略：复杂体素化 vs 标准体素网格质心降采样
            if self.use_voxelization:
                try:
                    # 使用体素化处理
                    processed_source, processed_target = voxelize_point_clouds(
                        source_clean, target_clean, self.num_points, 
                        self.voxel_config, fallback_to_sampling=True
                    )
                except Exception as e:
                    print(f"体素化处理失败，回退到标准体素网格质心降采样: {e}")
                    # 使用标准体素网格质心降采样作为回退方案
                    processed_source = adaptive_voxel_grid_downsample(source_clean, self.num_points)
                    processed_target = adaptive_voxel_grid_downsample(target_clean, self.num_points)
            else:
                # 标准体素网格质心降采样 (Voxel Grid Centroid Downsampling)
                processed_source = adaptive_voxel_grid_downsample(source_clean, self.num_points)
                processed_target = adaptive_voxel_grid_downsample(target_clean, self.num_points)
            
            # 转换回torch tensor
            source_tensor = torch.from_numpy(processed_source).float()
            target_tensor = torch.from_numpy(processed_target).float()
            
            # 根据降采样方法选择归一化策略
            if self.use_voxelization:
                # 复杂体素化：分别归一化，配合体素化预处理的裁剪操作
                # 源点云归一化
                source_min_vals = source_tensor.min(dim=0)[0]
                source_max_vals = source_tensor.max(dim=0)[0]
                source_center = (source_min_vals + source_max_vals) / 2
                source_scale = (source_max_vals - source_min_vals).max()
                
                if source_scale < 1e-10:
                    raise ValueError(f"源点云归一化尺度过小，样本索引{sample_idx}: {source_scale}")
                
                source_normalized = (source_tensor - source_center) / source_scale
                
                # 目标点云归一化
                target_min_vals = target_tensor.min(dim=0)[0]
                target_max_vals = target_tensor.max(dim=0)[0]
                target_center = (target_min_vals + target_max_vals) / 2
                target_scale = (target_max_vals - target_min_vals).max()
                
                if target_scale < 1e-10:
                    raise ValueError(f"目标点云归一化尺度过小，样本索引{sample_idx}: {target_scale}")
                
                target_normalized = (target_tensor - target_center) / target_scale
            else:
                # 标准体素网格质心降采样：联合归一化，保持相对空间关系
                source_normalized, target_normalized = joint_normalization(source_tensor, target_tensor)
            
            # 应用特定的扰动（而不是随机变换）
            # Apply specific perturbation (instead of random transform)
            perturbation = self.perturbation[index]
            twist = torch.from_numpy(numpy.array(perturbation)).contiguous().view(1, 6)
            x = twist.to(source_normalized)
            
            # 应用扰动变换
            if not self.fmt_trans:
                # x: twist-vector
                g = se3.exp(x).to(source_normalized) # [1, 4, 4]
                transformed_source = se3.transform(g, source_normalized)
                igt = g.squeeze(0) # igt: source_normalized -> transformed_source
            else:
                # x: rotation and translation
                w = x[:, 0:3]
                q = x[:, 3:6]
                R = so3.exp(w).to(source_normalized) # [1, 3, 3]
                g = torch.zeros(1, 4, 4)
                g[:, 3, 3] = 1
                g[:, 0:3, 0:3] = R # rotation
                g[:, 0:3, 3] = q   # translation
                transformed_source = se3.transform(g, source_normalized)
                igt = g.squeeze(0) # igt: source_normalized -> transformed_source
            
            # 收集原始点云信息（基于随机选择的样本）
            # 尝试获取原始点云文件路径
            if hasattr(self.dataset, 'indices') and self.original_pairs:
                # 如果是子集，需要映射索引
                orig_index = self.dataset.indices[sample_idx]
                source_file, target_file = self.original_pairs[orig_index]
                scene = self.original_pair_scenes[orig_index] if self.original_pair_scenes else "unknown"
            elif self.original_pairs and sample_idx < len(self.original_pairs):
                # 直接使用索引
                source_file, target_file = self.original_pairs[sample_idx]
                scene = self.original_pair_scenes[sample_idx] if self.original_pair_scenes else "unknown"
            else:
                source_file = None
                target_file = None
                scene = "unknown"
            
            # 尝试提取场景名称和序列号
            scene_name = scene
            source_seq = "0000"
            
            if source_file:
                try:
                    # 标准化路径分隔符
                    norm_path = source_file.replace('\\', '/')
                    
                    # 如果未能从数据集获取场景名称，则从路径中提取
                    if scene_name == "unknown" and 'C3VD_ply_source' in norm_path:
                        # 查找C3VD_ply_source之后的第一个目录
                        parts = norm_path.split('/')
                        idx = [i for i, part in enumerate(parts) if part == 'C3VD_ply_source']
                        if idx and idx[0] + 1 < len(parts):
                            scene_name = parts[idx[0] + 1]
                    
                    # 提取源序号
                    basename = os.path.basename(source_file)
                    
                    # 假设源文件名格式为 "XXXX_depth_pcd.ply"
                    if basename.endswith("_depth_pcd.ply") and basename[:4].isdigit():
                        source_seq = basename[:4]
                    else:
                        # 尝试从文件名中提取数字序列
                        import re
                        numbers = re.findall(r'\d+', basename)
                        if numbers:
                            source_seq = numbers[0].zfill(4)
                except Exception as e:
                    print(f"Warning: Error extracting scene name: {str(e)}")
            
            # 创建唯一标识符（包含扰动索引）
            identifier = f"{scene_name}_{source_seq}_pert{index:04d}"
            
            # 保存点云信息（使用测试索引而不是样本索引）
            self.cloud_info[index] = {
                'identifier': identifier,
                'scene': scene_name,
                'sequence': source_seq,
                'source_file': source_file,
                'target_file': target_file,
                'sample_index': sample_idx,  # 记录使用的随机样本索引
                'perturbation_index': index,  # 记录扰动索引
                'original_source': source_normalized.clone(),
                'original_target': target_normalized.clone(),
                'igt': igt.clone() if igt is not None else None
            }
            
            # 返回：模板、变换后的源、变换矩阵
            return target_normalized, transformed_source, igt
            
        except Exception as e:
            print(f"处理扰动索引{index}（样本索引{sample_idx}）的点云时出错: {str(e)}")
            raise

    def get_cloud_info(self, index):
        """获取指定索引的点云信息"""
        return self.cloud_info.get(index, {})
    
    def get_original_clouds(self, index):
        """获取指定索引的原始点云对"""
        info = self.cloud_info.get(index, {})
        return info.get('original_source'), info.get('original_target')
    
    def get_identifier(self, index):
        """获取指定索引的点云标识符"""
        info = self.cloud_info.get(index, {})
        return info.get('identifier', f"unknown_pert{index:04d}")

class SinglePairDataset(torch.utils.data.Dataset):
    """单对点云数据集，用于处理指定的源点云和目标点云对
    Single pair point cloud dataset for processing specified source and target point cloud pairs
    """
    def __init__(self, source_cloud_path, target_cloud_path, num_points=1024, 
                 use_voxelization=True, voxel_config=None):
        """
        Args:
            source_cloud_path: 源点云文件路径
            target_cloud_path: 目标点云文件路径
            num_points: 目标点云点数
            use_voxelization: 是否使用体素化处理
            voxel_config: 体素化配置
        """
        self.source_cloud_path = source_cloud_path
        self.target_cloud_path = target_cloud_path
        self.num_points = num_points
        self.use_voxelization = use_voxelization
        
        # 设置体素化配置
        if voxel_config is None:
            self.voxel_config = VoxelizationConfig(
                voxel_size=0.05,
                voxel_grid_size=32,
                max_voxel_points=100,
                max_voxels=20000,
                min_voxel_points_ratio=0.1
            )
        else:
            self.voxel_config = voxel_config
        
        # 保留原有的重采样器作为后备
        self.resampler = transforms.Resampler(num_points)
        
        print(f"SinglePairDataset初始化:")
        print(f"  源点云: {source_cloud_path}")
        print(f"  目标点云: {target_cloud_path}")
        print(f"  目标点数: {num_points}")
        print(f"  降采样方法: {'复杂体素化处理' if use_voxelization else '标准体素网格质心降采样'}")
        if use_voxelization:
            print(f"  体素化配置: 网格大小={self.voxel_config.voxel_grid_size}, 体素大小={self.voxel_config.voxel_size}")
        else:
            print(f"  降采样说明: 使用经典的体素网格质心降采样方法，保持空间结构")
    
    def __len__(self):
        return 1  # 只有一对点云
    
    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("SinglePairDataset只包含一对点云，索引必须为0")
        
        try:
            # 读取点云文件
            source = plyread(self.source_cloud_path)
            target = plyread(self.target_cloud_path)
            
            # 转换为numpy数组进行处理
            source_np = source.numpy() if isinstance(source, torch.Tensor) else source
            target_np = target.numpy() if isinstance(target, torch.Tensor) else target
            
            print(f"原始点云大小: 源={source_np.shape}, 目标={target_np.shape}")
            
            # 数据清理：移除无效点
            source_mask = np.isfinite(source_np).all(axis=1)
            target_mask = np.isfinite(target_np).all(axis=1)
            
            source_clean = source_np[source_mask]
            target_clean = target_np[target_mask]
            
            print(f"清理后点云大小: 源={source_clean.shape}, 目标={target_clean.shape}")
            
            if len(source_clean) < 100 or len(target_clean) < 100:
                raise ValueError(f"点云清理后点数不足100个: 源={len(source_clean)}, 目标={len(target_clean)}")
            
            # 选择处理策略：复杂体素化 vs 标准体素网格质心降采样
            if self.use_voxelization:
                try:
                    print("使用体素化处理...")
                    # 使用体素化处理
                    processed_source, processed_target = voxelize_point_clouds(
                        source_clean, target_clean, self.num_points, 
                        self.voxel_config, fallback_to_sampling=True
                    )
                    print(f"体素化处理后点云大小: 源={processed_source.shape}, 目标={processed_target.shape}")
                except Exception as e:
                    print(f"体素化处理失败，回退到标准体素网格质心降采样: {e}")
                    # 使用标准体素网格质心降采样作为回退方案
                    processed_source = adaptive_voxel_grid_downsample(source_clean, self.num_points)
                    processed_target = adaptive_voxel_grid_downsample(target_clean, self.num_points)
                    print(f"体素网格质心降采样处理后点云大小: 源={processed_source.shape}, 目标={processed_target.shape}")
            else:
                print("使用标准体素网格质心降采样...")
                # 标准体素网格质心降采样 (Voxel Grid Centroid Downsampling)
                processed_source = adaptive_voxel_grid_downsample(source_clean, self.num_points)
                processed_target = adaptive_voxel_grid_downsample(target_clean, self.num_points)
                print(f"体素网格质心降采样处理后点云大小: 源={processed_source.shape}, 目标={processed_target.shape}")
            
            # 转换回torch tensor
            source_tensor = torch.from_numpy(processed_source).float()
            target_tensor = torch.from_numpy(processed_target).float()
            
            # 根据降采样方法选择归一化策略
            if self.use_voxelization:
                # 复杂体素化：分别归一化，配合体素化预处理的裁剪操作
                # 源点云归一化
                source_min_vals = source_tensor.min(dim=0)[0]
                source_max_vals = source_tensor.max(dim=0)[0]
                source_center = (source_min_vals + source_max_vals) / 2
                source_scale = (source_max_vals - source_min_vals).max()
                
                if source_scale < 1e-10:
                    raise ValueError(f"源点云归一化尺度过小: {source_scale}")
                
                source_normalized = (source_tensor - source_center) / source_scale
                print(f"源点云分别归一化: 中心={source_center.numpy()}, 尺度={source_scale.item():.6f}")
                
                # 目标点云归一化
                target_min_vals = target_tensor.min(dim=0)[0]
                target_max_vals = target_tensor.max(dim=0)[0]
                target_center = (target_min_vals + target_max_vals) / 2
                target_scale = (target_max_vals - target_min_vals).max()
                
                if target_scale < 1e-10:
                    raise ValueError(f"目标点云归一化尺度过小: {target_scale}")
                    
                target_normalized = (target_tensor - target_center) / target_scale
                print(f"目标点云分别归一化: 中心={target_center.numpy()}, 尺度={target_scale.item():.6f}")
            else:
                # 标准体素网格质心降采样：联合归一化，保持相对空间关系
                source_normalized, target_normalized = joint_normalization(source_tensor, target_tensor)
                print(f"联合归一化完成: 保持了源点云和目标点云之间的相对空间关系")
            
            # 创建一个默认的4x4单位矩阵作为变换矩阵
            igt = torch.eye(4)
            
            print(f"最终输出点云大小: 源={source_normalized.shape}, 目标={target_normalized.shape}")
            
            return source_normalized, target_normalized, igt
            
        except Exception as e:
            print(f"处理单对点云时出错: {str(e)}")
            raise


class SinglePairTrackingDataset(torch.utils.data.Dataset):
    """单对点云跟踪数据集，支持指定扰动的刚性变换
    Single pair point cloud tracking dataset with specified perturbation rigid transformation
    """
    def __init__(self, source_cloud_path, target_cloud_path, perturbation, 
                 num_points=1024, use_voxelization=True, voxel_config=None, fmt_trans=False):
        """
        Args:
            source_cloud_path: 源点云文件路径
            target_cloud_path: 目标点云文件路径
            perturbation: 扰动值 (6维向量: rx,ry,rz,tx,ty,tz)
            num_points: 目标点云点数
            use_voxelization: 是否使用体素化处理
            voxel_config: 体素化配置
            fmt_trans: 扰动格式 (False: twist, True: rotation+translation)
        """
        # 创建基础数据集
        self.base_dataset = SinglePairDataset(
            source_cloud_path, target_cloud_path, num_points, use_voxelization, voxel_config
        )
        
        self.perturbation = np.array(perturbation)
        self.fmt_trans = fmt_trans
        
        print(f"SinglePairTrackingDataset初始化:")
        print(f"  扰动值: {self.perturbation}")
        print(f"  扰动格式: {'rotation+translation' if fmt_trans else 'twist'}")
    
    def do_transform(self, p0, x):
        """应用扰动变换"""
        # p0: [N, 3]
        # x: [1, 6]
        if not self.fmt_trans:
            # x: twist-vector
            g = se3.exp(x).to(p0) # [1, 4, 4]
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0) # igt: p0 -> p1
        else:
            # x: rotation and translation
            w = x[:, 0:3]
            q = x[:, 3:6]
            R = so3.exp(w).to(p0) # [1, 3, 3]
            g = torch.zeros(1, 4, 4)
            g[:, 3, 3] = 1
            g[:, 0:3, 0:3] = R # rotation
            g[:, 0:3, 3] = q   # translation
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0) # igt: p0 -> p1
        return p1, igt
    
    def __len__(self):
        return 1  # 只有一对点云
    
    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError("SinglePairTrackingDataset只包含一对点云，索引必须为0")
        
        try:
            # 获取归一化的点云对
            source, target, _ = self.base_dataset[0]
            
            # 准备扰动变换
            twist = torch.from_numpy(self.perturbation).contiguous().view(1, 6).float()
            x = twist.to(source)
            
            # 应用扰动变换 (target作为模板，source经过变换后作为源点云)
            transformed_source, igt = self.do_transform(source, x)
            
            print(f"应用扰动变换:")
            print(f"  扰动向量: {twist.squeeze().numpy()}")
            print(f"  变换矩阵形状: {igt.shape}")
            
            # 返回：模板(目标)、变换后的源、变换矩阵
            return target, transformed_source, igt
            
        except Exception as e:
            print(f"处理单对点云跟踪时出错: {str(e)}")
            raise

# 添加 C3VD 数据集支持: C3VDDataset, C3VDset4tracking, C3VDset4tracking_test
class C3VDDataset(torch.utils.data.Dataset):
    """C3VD 数据集加载器"""
    def __init__(self, source_root, target_root=None, transform=None, pair_mode='one_to_one', reference_name=None):
        import os, glob
        import torch
        self.source_root = source_root
        self.target_root = target_root or os.path.join(os.path.dirname(source_root), 'visible_point_cloud_ply_depth')
        self.transform = transform
        self.pair_mode = pair_mode
        self.reference_name = reference_name
        if not os.path.exists(self.source_root):
            raise FileNotFoundError(f"源点云目录不存在: {self.source_root}")
        if not os.path.exists(self.target_root):
            raise FileNotFoundError(f"目标点云目录不存在: {self.target_root}")
        self.folder_names = [d for d in sorted(os.listdir(self.source_root)) if os.path.isdir(os.path.join(self.source_root, d))]
        self.pairs = []
        self.pair_scenes = []
        for folder in self.folder_names:
            src_dir = os.path.join(self.source_root, folder)
            tgt_dir = os.path.join(self.target_root, folder)
            src_files = sorted(glob.glob(os.path.join(src_dir, '*.ply')))
            if self.pair_mode == 'one_to_one':
                for src in src_files:
                    basename = os.path.basename(src)
                    tgt = os.path.join(tgt_dir, basename)
                    if not os.path.exists(tgt):
                        alt = basename.replace('_depth_pcd.ply', '_visible.ply')
                        tgt = os.path.join(tgt_dir, alt)
                    self.pairs.append((src, tgt))
                    self.pair_scenes.append(folder)
            elif self.pair_mode == 'scene_reference':
                if self.reference_name:
                    ref = os.path.join(tgt_dir, self.reference_name)
                else:
                    tfs = sorted(glob.glob(os.path.join(tgt_dir, '*.ply')))
                    ref = tfs[0] if tfs else None
                for src in src_files:
                    if ref:
                        self.pairs.append((src, ref))
                        self.pair_scenes.append(folder)
            else:
                raise ValueError(f"不支持的配对模式: {self.pair_mode}")
    def __len__(self):
        return len(self.pairs)
    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        source = plyread(src)
        target = plyread(tgt)
        if self.transform:
            source = self.transform(source)
            target = self.transform(target)
        igt = torch.eye(4)
        return source, target, igt

class C3VDset4tracking(torch.utils.data.Dataset):
    """C3VD 跟踪数据集 (训练用)"""
    def __init__(self, dataset, rigid_transform, num_points=1024, use_voxelization=True, voxel_config=None):
        self.dataset = dataset
        self.rigid_transform = rigid_transform
        self.num_points = num_points
        self.resampler = transforms.Resampler(num_points)
        self.use_voxelization = use_voxelization
        self.voxel_config = voxel_config
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        source, target, _ = self.dataset[idx]
        source = source[torch.isfinite(source).all(dim=1)]
        target = target[torch.isfinite(target).all(dim=1)]
        source = self.resampler(source)
        target = self.resampler(target)
        source_norm, target_norm = joint_normalization(source, target)
        p1 = self.rigid_transform(source_norm)
        igt = self.rigid_transform.igt
        return target_norm, p1, igt

class C3VDset4tracking_test(torch.utils.data.Dataset):
    """C3VD 跟踪数据集 (测试用)"""
    def __init__(self, dataset, rigid_transform, use_joint_normalization=False, num_points=1024):
        self.dataset = dataset
        self.rigid_transform = rigid_transform
        self.use_joint_normalization = use_joint_normalization
        self.num_points = num_points
        self.resampler = transforms.Resampler(num_points)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        source, target, _ = self.dataset[idx]
        source = source[torch.isfinite(source).all(dim=1)]
        target = target[torch.isfinite(target).all(dim=1)]
        source = self.resampler(source)
        target = self.resampler(target)
        if self.use_joint_normalization:
            source_norm, target_norm = joint_normalization(source, target)
        else:
            def normalize(p):
                minv = p.min(dim=0)[0]
                maxv = p.max(dim=0)[0]
                center = (minv + maxv) / 2
                scale = (maxv - minv).max()
                return (p - center) / scale
            source_norm = normalize(source)
            target_norm = normalize(target)
        p1 = self.rigid_transform(source_norm)
        igt = self.rigid_transform.igt
        return target_norm, p1, igt

# EOF
