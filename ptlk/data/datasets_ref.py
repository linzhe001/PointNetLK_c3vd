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
                 fmt_trans=False, random_seed=42):
        self.dataset = dataset
        self.perturbation = numpy.array(perturbation) # twist (perturbation_count, 6)
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier
        self.fmt_trans = fmt_trans # twist or (rotation and translation)
        
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
        twist = torch.from_numpy(numpy.array(self.perturbation[index])).contiguous().view(1, 6)
        
        # 使用预先生成的随机样本索引
        # Use pre-generated random sample index
        sample_idx = self.sample_indices[index]
        pm, _ = self.dataset[sample_idx]
        
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


def plyread(file_path):
    """Read point cloud from PLY file"""
    ply_data = PlyData.read(file_path)
    pc = np.vstack([
        ply_data['vertex']['x'],
        ply_data['vertex']['y'],
        ply_data['vertex']['z']
    ]).T
    return torch.from_numpy(pc.astype(np.float32))

class C3VDDataset(torch.utils.data.Dataset):
    """C3VD dataset loader that handles pairs of source and target point clouds."""
    def __init__(self, source_root, target_root=None, transform=None, pair_mode='one_to_one', reference_name=None, use_prefix=False):
        """
        Args:
            source_root (str): 源点云根目录 (C3VD_ply_source)
            target_root (str, optional): 目标点云根目录。如果为None，会根据pair_mode自动设置
            transform: 点云变换
            pair_mode (str): 配对模式，可选值如下:
                - 'one_to_one': 每个源点云对应一个特定的目标点云 (原有模式)
                - 'scene_reference': 每个场景使用一个共享的目标点云 (原有模式)
                - 'source_to_source': 源点云和源点云之间的配对 (数据增强)
                - 'target_to_target': 目标点云和目标点云之间的配对 (数据增强)
                - 'all': 包含所有配对方式 (完整数据增强)
            reference_name (str, optional): 场景参考模式下目标点云的名称，若为None则使用场景中的第一个点云
            use_prefix (bool): 是否使用场景名称前缀（第一个下划线前的部分）作为类别
        """
        self.source_root = source_root
        self.pair_mode = pair_mode
        self.reference_name = reference_name
        self.use_prefix = use_prefix
        
        # 根据配对模式设置目标点云路径
        if target_root is None:
            if pair_mode == 'scene_reference':
                # 场景参考模式使用 C3VD_ref 目录
                base_dir = os.path.dirname(source_root)
                self.target_root = os.path.join(base_dir, 'C3VD_ref')
            else:
                # 其他模式使用 visible_point_cloud_ply_depth 目录
                base_dir = os.path.dirname(source_root)
                self.target_root = os.path.join(base_dir, 'visible_point_cloud_ply_depth')
        else:
            self.target_root = target_root
            
        print(f"\n====== C3VD Dataset Configuration ======")
        print(f"Pairing Mode: {pair_mode}")
        print(f"Source Directory: {self.source_root}")
        print(f"Target Directory: {self.target_root}")
        
        self.transform = transform
        
        # Get point cloud size from transform if available
        self.num_points = 1024  # Default value
        if transform and hasattr(transform, 'transforms'):
            for t in transform.transforms:
                if hasattr(t, 'num_points'):
                    self.num_points = t.num_points
                    break
        
        # Get scene folders
        self.scenes = []
        self.scene_prefixes = {}  # 映射场景到前缀
        for scene_dir in glob.glob(os.path.join(source_root, "*")):
            if os.path.isdir(scene_dir):
                scene_name = os.path.basename(scene_dir)
                self.scenes.append(scene_name)
                
                # 提取场景前缀（第一个下划线前的部分）
                if self.use_prefix:
                    prefix = scene_name.split('_')[0]
                    self.scene_prefixes[scene_name] = prefix
        
        # Get point cloud pairs for each scene
        self.pairs = []
        self.pair_scenes = []  # Added: track the scene for each point cloud pair
        
        # 如果是数据增强模式，用于记录每个配对的类型
        if pair_mode == 'all':
            self.pair_types = []
            
        # 获取每个场景中的所有点云文件（用于数据增强模式）
        self.scene_source_files = {}
        self.scene_target_files = {}
        
        for scene in self.scenes:
            # 获取场景中的所有源点云
            source_files = glob.glob(os.path.join(source_root, scene, "????_depth_pcd.ply"))
            self.scene_source_files[scene] = sorted(source_files)
            
            # 获取场景中的所有目标点云
            target_files = glob.glob(os.path.join(self.target_root, scene, "frame_????_visible.ply"))
            self.scene_target_files[scene] = sorted(target_files)
        
        # 根据配对模式创建点云对
        if pair_mode == 'one_to_one' or pair_mode == 'all':
            # 处理原有的one_to_one模式
            self._create_one_to_one_pairs()
            
        elif pair_mode == 'scene_reference':
            # 处理原有的scene_reference模式
            self._create_scene_reference_pairs()
            
        elif pair_mode == 'source_to_source':
            # 新增：源点云之间配对
            self._create_source_to_source_pairs()
            
        elif pair_mode == 'target_to_target':
            # 新增：目标点云之间配对
            self._create_target_to_target_pairs()
        
        # 如果是all模式，需要创建所有配对
        if pair_mode == 'all':
            # 已经创建了one_to_one配对，再添加其他配对
            self._create_scene_reference_pairs()
            self._create_source_to_source_pairs()
            self._create_target_to_target_pairs()
            
            # 打印各类配对数量
            source_target_one_to_one_count = sum(1 for i, t in enumerate(self.pair_types) if t == 'one_to_one')
            source_target_reference_count = sum(1 for i, t in enumerate(self.pair_types) if t == 'scene_reference')
            source_source_count = sum(1 for i, t in enumerate(self.pair_types) if t == 'source_to_source')
            target_target_count = sum(1 for i, t in enumerate(self.pair_types) if t == 'target_to_target')
            
            print(f"- Source-to-Target pairs (one_to_one): {source_target_one_to_one_count}")
            print(f"- Source-to-Reference pairs: {source_target_reference_count}")
            print(f"- Source-to-Source pairs: {source_source_count}")
            print(f"- Target-to-Target pairs: {target_target_count}")
        
        print(f"Total point cloud pairs loaded: {len(self.pairs)}")
    
    def _create_one_to_one_pairs(self):
        """创建源点云到目标点云的一一对应配对"""
        pair_count = 0
        
        for scene in self.scenes:
            source_files = self.scene_source_files.get(scene, [])
            
            for source_file in source_files:
                # 从源点云文件名提取序号
                basename = os.path.basename(source_file)
                frame_idx = basename[:4]  # 提取文件名前4位数字作为序号
                
                # 构建对应的目标点云文件名
                target_file = os.path.join(self.target_root, scene, f"frame_{frame_idx}_visible.ply")
                
                # 确认目标点云文件存在
                if os.path.exists(target_file):
                    self.pairs.append((source_file, target_file))
                    self.pair_scenes.append(scene)
                    
                    # 记录配对类型（如果是all模式）
                    if hasattr(self, 'pair_types'):
                        self.pair_types.append('one_to_one')
                    
                    pair_count += 1
        
        if pair_count > 0:
            print(f"Source-to-Target pairs: {pair_count}")
    
    def _create_scene_reference_pairs(self):
        """创建源点云到参考目标点云的配对"""
        pair_count = 0
        
        for scene in self.scenes:
            source_files = self.scene_source_files.get(scene, [])
            
            # 找出参考点云
            if self.reference_name:
                # 使用指定的参考点云名称
                reference_file = os.path.join(self.target_root, scene, self.reference_name)
            else:
                # 使用第一个目标点云作为参考
                target_files = self.scene_target_files.get(scene, [])
                if not target_files:
                    print(f"Warning: No target point clouds found for scene {scene}, skipping")
                    continue
                reference_file = target_files[0]  # 已排序，取第一个
            
            # 确认参考点云文件存在
            if os.path.exists(reference_file):
                # 每个源点云都对应同一个参考点云
                for source_file in source_files:
                    # 如果是all模式，避免重复添加已有的配对
                    if self.pair_mode == 'all':
                        # 检查这个配对是否已经在one_to_one模式中添加过了
                        basename = os.path.basename(source_file)
                        frame_idx = basename[:4]
                        target_name = f"frame_{frame_idx}_visible.ply"
                        
                        # 如果参考点云恰好是对应的目标点云，就跳过
                        if os.path.basename(reference_file) == target_name:
                            continue
                    
                    self.pairs.append((source_file, reference_file))
                    self.pair_scenes.append(scene)
                    
                    # 记录配对类型（如果是all模式）
                    if hasattr(self, 'pair_types'):
                        self.pair_types.append('scene_reference')
                    
                    pair_count += 1
            else:
                print(f"Warning: Reference point cloud {reference_file} for scene {scene} does not exist, skipping")
        
        if pair_count > 0:
            print(f"Source-to-Reference pairs: {pair_count}")
    
    def _create_source_to_source_pairs(self):
        """创建源点云到源点云的配对（数据增强）"""
        pair_count = 0
        
        for scene in self.scenes:
            source_files = self.scene_source_files.get(scene, [])
            
            # 需要至少两个源点云文件才能配对
            if len(source_files) < 2:
                continue
            
            # 配对相邻帧的源点云
            for i in range(len(source_files) - 1):
                source_file1 = source_files[i]
                source_file2 = source_files[i + 1]
                
                self.pairs.append((source_file1, source_file2))
                self.pair_scenes.append(scene)
                
                # 记录配对类型（如果是all模式）
                if hasattr(self, 'pair_types'):
                    self.pair_types.append('source_to_source')
                
                pair_count += 1
        
        if pair_count > 0:
            print(f"Source-to-Source pairs: {pair_count}")
    
    def _create_target_to_target_pairs(self):
        """创建目标点云到目标点云的配对（数据增强）"""
        pair_count = 0
        
        for scene in self.scenes:
            target_files = self.scene_target_files.get(scene, [])
            
            # 需要至少两个目标点云文件才能配对
            if len(target_files) < 2:
                continue
            
            # 配对相邻帧的目标点云
            for i in range(len(target_files) - 1):
                target_file1 = target_files[i]
                target_file2 = target_files[i + 1]
                
                self.pairs.append((target_file1, target_file2))
                self.pair_scenes.append(scene)
                
                # 记录配对类型（如果是all模式）
                if hasattr(self, 'pair_types'):
                    self.pair_types.append('target_to_target')
                
                pair_count += 1
        
        if pair_count > 0:
            print(f"Target-to-Target pairs: {pair_count}")
    
    def get_scene_indices(self, scene_names):
        """获取指定场景的样本索引"""
        indices = []
        
        # 检查pair_scenes属性是否存在
        if not hasattr(self, 'pair_scenes') or not self.pair_scenes:
            print(f"警告: 没有pair_scenes属性，尝试从文件路径提取场景信息")
            # 从文件路径提取场景信息
        else:
            # 正常使用pair_scenes
            for i, scene in enumerate(self.pair_scenes):
                if scene in scene_names:
                    indices.append(i)
        
        print(f"找到 {len(indices)} 个属于指定场景的样本")
        return indices

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        source_file, target_file = self.pairs[idx]
        
        # Read source and target point clouds
        source = plyread(source_file)
        target = plyread(target_file)
        
        # Check point cloud validity
        if not torch.isfinite(source).all() or not torch.isfinite(target).all():
            raise ValueError(f"Point cloud at index {idx} contains invalid values")
        
        # Ensure point clouds are not empty
        if source.shape[0] == 0 or target.shape[0] == 0:
            raise ValueError(f"Point cloud at index {idx} is empty")
        
        # Apply transformations
        if self.transform:
            source = self.transform(source)
            target = self.transform(target)
        
        # 创建一个默认的4x4单位矩阵作为变换矩阵，而不是返回None
        igt = torch.eye(4)
        
        return source, target, igt

class C3VDset4tracking(torch.utils.data.Dataset):
    """C3VD配准跟踪数据集，支持体素化预处理和刚性变换"""
    
    def __init__(self, dataset, rigid_transform, num_points=1024, use_voxelization=True, voxel_config=None):
        """
        Args:
            dataset: C3VD基础数据集
            rigid_transform: 刚性变换生成器
            num_points: 目标点云点数
            use_voxelization: 是否使用体素化处理
            voxel_config: 体素化配置，如果为None则使用默认配置
        """
        self.dataset = dataset
        self.rigid_transform = rigid_transform
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
        
        print(f"C3VDset4tracking初始化:")
        print(f"  目标点数: {num_points}")
        print(f"  体素化处理: {'启用' if use_voxelization else '禁用'}")
        if use_voxelization:
            print(f"  体素化配置: 网格大小={self.voxel_config.voxel_grid_size}, 体素大小={self.voxel_config.voxel_size}")
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        try:
            # 获取原始点云对
            source, target, _ = self.dataset[idx]  # 忽略原始的igt
            
            # 转换为numpy数组进行处理
            source_np = source.numpy() if isinstance(source, torch.Tensor) else source
            target_np = target.numpy() if isinstance(target, torch.Tensor) else target
            
            # 数据清理：移除无效点
            source_mask = np.isfinite(source_np).all(axis=1)
            target_mask = np.isfinite(target_np).all(axis=1)
            
            source_clean = source_np[source_mask]
            target_clean = target_np[target_mask]
            
            if len(source_clean) < 100 or len(target_clean) < 100:
                raise ValueError(f"点云在索引{idx}处清理后点数不足100个")
            
            # 选择处理策略：体素化 vs 简单重采样
            if self.use_voxelization:
                try:
                    # 使用体素化处理
                    processed_source, processed_target = voxelize_point_clouds(
                        source_clean, target_clean, self.num_points, 
                        self.voxel_config, fallback_to_sampling=True
                    )
                except Exception as e:
                    print(f"体素化处理失败，回退到重采样: {e}")
                    # 转换回torch tensor用于重采样
                    source_tensor = torch.from_numpy(source_clean).float()
                    target_tensor = torch.from_numpy(target_clean).float()
                    processed_source = self.resampler(source_tensor).numpy()
                    processed_target = self.resampler(target_tensor).numpy()
            else:
                # 使用简单重采样
                source_tensor = torch.from_numpy(source_clean).float()
                target_tensor = torch.from_numpy(target_clean).float()
                processed_source = self.resampler(source_tensor).numpy()
                processed_target = self.resampler(target_tensor).numpy()
            
            # 转换回torch tensor
            source_tensor = torch.from_numpy(processed_source).float()
            target_tensor = torch.from_numpy(processed_target).float()
            
            # 分别归一化：对每个点云单独进行归一化，配合体素化预处理的裁剪操作
            # 源点云归一化
            source_min_vals = source_tensor.min(dim=0)[0]
            source_max_vals = source_tensor.max(dim=0)[0]
            source_center = (source_min_vals + source_max_vals) / 2
            source_scale = (source_max_vals - source_min_vals).max()
            
            if source_scale < 1e-10:
                raise ValueError(f"源点云归一化尺度过小，索引{idx}: {source_scale}")
            
            source_normalized = (source_tensor - source_center) / source_scale
            
            # 目标点云归一化
            target_min_vals = target_tensor.min(dim=0)[0]
            target_max_vals = target_tensor.max(dim=0)[0]
            target_center = (target_min_vals + target_max_vals) / 2
            target_scale = (target_max_vals - target_min_vals).max()
            
            if target_scale < 1e-10:
                raise ValueError(f"目标点云归一化尺度过小，索引{idx}: {target_scale}")
                
            target_normalized = (target_tensor - target_center) / target_scale
            
            # 应用随机刚性变换
            # target作为模板，source经过变换后作为源点云
            transformed_source = self.rigid_transform(source_normalized)
            igt = self.rigid_transform.igt  # 获取真实的变换矩阵
            
            # 返回：模板、变换后的源、变换矩阵
            return target_normalized, transformed_source, igt
            
        except Exception as e:
            print(f"处理索引{idx}的点云时出错: {str(e)}")
            raise

class C3VDset4tracking_test(C3VDset4tracking):
    """用于测试的C3VD跟踪数据集，保留原始点云引用和文件路径
    
    此类扩展了C3VDset4tracking类，添加了以下功能：
    1. 保留原始点云数据的引用和文件路径
    2. 为每个点云对创建唯一索引，方便后续引用
    3. 提供获取原始未变换点云的方法
    """
    def __init__(self, dataset, rigid_transform, num_points=1024, 
                 use_voxelization=True, voxel_config=None):
        """
        Args:
            dataset: C3VD基础数据集
            rigid_transform: 刚性变换生成器
            num_points: 目标点云点数
            use_voxelization: 是否使用体素化处理
            voxel_config: 体素化配置
        """
        # 调用父类构造函数
        super().__init__(dataset, rigid_transform, num_points, use_voxelization, voxel_config)
        
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
    
    def __getitem__(self, index):
        """获取测试数据项，同时保存原始点云信息"""
        try:
            # 获取原始点云对
            source, target, _ = self.dataset[index]  # 忽略原始的igt
            
            # 转换为numpy数组进行处理
            source_np = source.numpy() if isinstance(source, torch.Tensor) else source
            target_np = target.numpy() if isinstance(target, torch.Tensor) else target
            
            # 数据清理：移除无效点
            source_mask = np.isfinite(source_np).all(axis=1)
            target_mask = np.isfinite(target_np).all(axis=1)
            
            source_clean = source_np[source_mask]
            target_clean = target_np[target_mask]
            
            if len(source_clean) < 100 or len(target_clean) < 100:
                raise ValueError(f"点云在索引{index}处清理后点数不足100个")
            
            # 选择处理策略：体素化 vs 简单重采样
            if self.use_voxelization:
                try:
                    # 使用体素化处理
                    processed_source, processed_target = voxelize_point_clouds(
                        source_clean, target_clean, self.num_points, 
                        self.voxel_config, fallback_to_sampling=True
                    )
                except Exception as e:
                    print(f"体素化处理失败，回退到重采样: {e}")
                    # 转换回torch tensor用于重采样
                    source_tensor = torch.from_numpy(source_clean).float()
                    target_tensor = torch.from_numpy(target_clean).float()
                    processed_source = self.resampler(source_tensor).numpy()
                    processed_target = self.resampler(target_tensor).numpy()
            else:
                # 使用简单重采样
                source_tensor = torch.from_numpy(source_clean).float()
                target_tensor = torch.from_numpy(target_clean).float()
                processed_source = self.resampler(source_tensor).numpy()
                processed_target = self.resampler(target_tensor).numpy()
            
            # 转换回torch tensor
            source_tensor = torch.from_numpy(processed_source).float()
            target_tensor = torch.from_numpy(processed_target).float()
            
            # 分别归一化：对每个点云单独进行归一化，配合体素化预处理的裁剪操作
            # 源点云归一化
            source_min_vals = source_tensor.min(dim=0)[0]
            source_max_vals = source_tensor.max(dim=0)[0]
            source_center = (source_min_vals + source_max_vals) / 2
            source_scale = (source_max_vals - source_min_vals).max()
            
            if source_scale < 1e-10:
                raise ValueError(f"源点云归一化尺度过小，索引{index}: {source_scale}")
            
            source_normalized = (source_tensor - source_center) / source_scale
            
            # 目标点云归一化
            target_min_vals = target_tensor.min(dim=0)[0]
            target_max_vals = target_tensor.max(dim=0)[0]
            target_center = (target_min_vals + target_max_vals) / 2
            target_scale = (target_max_vals - target_min_vals).max()
            
            if target_scale < 1e-10:
                raise ValueError(f"目标点云归一化尺度过小，索引{index}: {target_scale}")
            
            target_normalized = (target_tensor - target_center) / target_scale
            
            # 应用随机刚性变换
            # target作为模板，source经过变换后作为源点云
            transformed_source = self.rigid_transform(source_normalized)
            igt = self.rigid_transform.igt  # 获取真实的变换矩阵
            
            # 收集原始点云信息
            # 尝试获取原始点云文件路径
            if hasattr(self.dataset, 'indices') and self.original_pairs:
                # 如果是子集，需要映射索引
                orig_index = self.dataset.indices[index]
                source_file, target_file = self.original_pairs[orig_index]
                scene = self.original_pair_scenes[orig_index] if self.original_pair_scenes else "unknown"
            elif self.original_pairs and index < len(self.original_pairs):
                # 直接使用索引
                source_file, target_file = self.original_pairs[index]
                scene = self.original_pair_scenes[index] if self.original_pair_scenes else "unknown"
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
            
            # 创建唯一标识符
            identifier = f"{scene_name}_{source_seq}"
            
            # 保存点云信息
            self.cloud_info[index] = {
                'identifier': identifier,
                'scene': scene_name,
                'sequence': source_seq,
                'source_file': source_file,
                'target_file': target_file,
                'original_source': source_normalized.clone(),
                'original_target': target_normalized.clone(),
                'igt': igt.clone() if igt is not None else None
            }
            
            # 返回：模板、变换后的源、变换矩阵
            return target_normalized, transformed_source, igt
            
        except Exception as e:
            print(f"处理索引{index}的点云时出错: {str(e)}")
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
        return info.get('identifier', f"unknown_{index:04d}")

class C3VDset4tracking_test_random_sample(C3VDset4tracking_test):
    """
    专用于gt_poses.csv文件的C3VD测试数据集类
    每个扰动随机选择一个测试样本，总共测试扰动数量次
    Special C3VD test dataset class for gt_poses.csv files
    Each perturbation randomly selects a test sample, testing perturbation count times total
    """
    def __init__(self, dataset, rigid_transform, num_points=1024, 
                 use_voxelization=True, voxel_config=None, random_seed=42):
        """
        Args:
            dataset: C3VD基础数据集
            rigid_transform: 刚性变换生成器（包含所有扰动）
            num_points: 目标点云点数
            use_voxelization: 是否使用体素化处理
            voxel_config: 体素化配置
            random_seed: 随机种子
        """
        # 调用父类构造函数
        super().__init__(dataset, rigid_transform, num_points, use_voxelization, voxel_config)
        
        # 检查是否有扰动数据
        if not hasattr(rigid_transform, 'perturbations'):
            raise ValueError("rigid_transform must have perturbations attribute for random sampling mode")
        
        self.perturbations = rigid_transform.perturbations
        self.original_dataset_size = len(dataset)
        
        # 设置随机种子以确保可复现性
        # Set random seed for reproducibility
        self.random_seed = random_seed
        numpy.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # 为每个扰动预先生成随机样本索引
        # Pre-generate random sample indices for each perturbation
        self.sample_indices = numpy.random.randint(0, self.original_dataset_size, size=len(self.perturbations))
        
        print(f"C3VD Random sampling mode activated for gt_poses.csv:")
        print(f"- Total perturbations: {len(self.perturbations)}")
        print(f"- Original dataset size: {self.original_dataset_size}")
        print(f"- Random seed: {random_seed}")
        print(f"- Test iterations: {len(self.perturbations)} (one per perturbation)")

    def __len__(self):
        # 返回扰动数量，而不是数据集大小
        # Return perturbation count, not dataset size
        return len(self.perturbations)

    def __getitem__(self, index):
        """
        获取测试数据项，使用随机选择的样本和指定的扰动
        Args:
            index: 扰动索引 (0 到 len(perturbations)-1)
        """
        # index是扰动索引 (0 到 len(perturbations)-1)
        # index is perturbation index (0 to len(perturbations)-1)
        
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
            
            # 选择处理策略：体素化 vs 简单重采样
            if self.use_voxelization:
                try:
                    # 使用体素化处理
                    processed_source, processed_target = voxelize_point_clouds(
                        source_clean, target_clean, self.num_points, 
                        self.voxel_config, fallback_to_sampling=True
                    )
                except Exception as e:
                    print(f"体素化处理失败，回退到重采样: {e}")
                    # 转换回torch tensor用于重采样
                    source_tensor = torch.from_numpy(source_clean).float()
                    target_tensor = torch.from_numpy(target_clean).float()
                    processed_source = self.resampler(source_tensor).numpy()
                    processed_target = self.resampler(target_tensor).numpy()
            else:
                # 使用简单重采样
                source_tensor = torch.from_numpy(source_clean).float()
                target_tensor = torch.from_numpy(target_clean).float()
                processed_source = self.resampler(source_tensor).numpy()
                processed_target = self.resampler(target_tensor).numpy()
            
            # 转换回torch tensor
            source_tensor = torch.from_numpy(processed_source).float()
            target_tensor = torch.from_numpy(processed_target).float()
            
            # 分别归一化
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
            
            # 应用特定的扰动（而不是随机变换）
            # Apply specific perturbation (instead of random transform)
            perturbation = self.perturbations[index]
            twist = torch.from_numpy(numpy.array(perturbation)).contiguous().view(1, 6)
            x = twist.to(source_normalized)
            
            # 应用扰动变换
            if not getattr(self.rigid_transform, 'fmt_trans', False):
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

# EOF
