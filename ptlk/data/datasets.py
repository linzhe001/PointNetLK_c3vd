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
# 新增：导入复杂体素化函数
try:
    from .datasets_ref import voxelize_point_clouds, VoxelizationConfig
except Exception:
    voxelize_point_clouds = None
    VoxelizationConfig = None


def voxel_down_sample_numpy(points, num_points):
    """
    使用基于VoxelGrid的均匀降采样，近似open3d的voxel_down_sample
    Args:
        points: numpy array (N,3)
        num_points: 目标点数
    Returns:
        sampled_points: numpy array (num_points,3)
    """
    # 计算边界框
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
        
    # 计算每轴体素格数量（立方根）
    n_side = int(np.ceil(num_points ** (1/3)))
        
    # 防止零范围
    extents = max_bounds - min_bounds
    extents[extents == 0] = 1e-6
    
    # 计算体素大小
    voxel_size = extents / n_side
    
    # 计算体素索引
    coords = np.floor((points - min_bounds) / voxel_size).astype(np.int64)
    coords = np.minimum(coords, n_side - 1)
    
    # 将三维体素索引映射为一维键
    keys = coords[:,0] + coords[:,1] * n_side + coords[:,2] * (n_side ** 2)
    
    # 获取唯一体素和映射
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    
    # 计算每个体素的质心
    counts = np.bincount(inverse)
    sums = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
    for dim in range(3):
        sums[:, dim] = np.bincount(inverse, weights=points[:, dim])
    centroids = sums / counts[:, None]
    
    M = centroids.shape[0]
    if M >= num_points:
        indices = np.random.choice(M, num_points, replace=False)
        sampled = centroids[indices]
    else:
        shortage = num_points - M
        extra_idx = np.random.choice(M, shortage, replace=True)
        sampled = np.concatenate([centroids, centroids[extra_idx]], axis=0)
    
    return sampled


def plyread(file_path):
    """Read point cloud from PLY file"""
    ply_data = PlyData.read(file_path)
    pc = np.vstack([
        ply_data['vertex']['x'],
        ply_data['vertex']['y'],
        ply_data['vertex']['z']
    ]).T
    return torch.from_numpy(pc.astype(np.float32))


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

    def __getitem__(self, idx):
        pm, _ = self.dataset[idx]
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

    def __getitem__(self, idx):
        twist = torch.from_numpy(numpy.array(self.perturbation[idx])).contiguous().view(1, 6)
        pm, _ = self.dataset[idx]
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
    """
    def __init__(self, dataset, perturbation, source_modifier=None, template_modifier=None,
                 fmt_trans=False, random_seed=42):
        self.dataset = dataset
        self.perturbation = numpy.array(perturbation) # twist (perturbation_count, 6)
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier
        self.fmt_trans = fmt_trans # twist or (rotation and translation)
        
        # 设置随机种子以确保可复现性
        self.random_seed = random_seed
        numpy.random.seed(random_seed)
        
        # 为每个扰动预先生成随机样本索引
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
        return len(self.perturbation)

    def __getitem__(self, idx):
        # idx是扰动索引 (0 到 len(perturbation)-1)
        twist = torch.from_numpy(numpy.array(self.perturbation[idx])).contiguous().view(1, 6)
        
        # 使用预先生成的随机样本索引
        sample_idx = self.sample_indices[idx]
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
            use_voxelization: 是否使用复杂体素化处理，False时使用Open3D风格的VoxelGrid降采样
            voxel_config: 体素化配置（当use_voxelization=True时使用）
        """
        self.dataset = dataset
        self.rigid_transform = rigid_transform
        self.num_points = num_points
        self.use_voxelization = use_voxelization
        # 设置体素化配置
        if self.use_voxelization and voxelize_point_clouds is not None:
            if voxel_config is None and VoxelizationConfig is not None:
                self.voxel_config = VoxelizationConfig()
            else:
                self.voxel_config = voxel_config
        else:
            self.voxel_config = None
        
        # 保留原有的重采样器作为后备
        self.resampler = transforms.Resampler(num_points)
        
        print(f"C3VDset4tracking初始化:")
        print(f"  目标点数: {num_points}")
        print(f"  体素化处理: {'复杂体素化' if use_voxelization else 'Open3D风格VoxelGrid降采样'}")
        
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
            
            # 选择处理策略：复杂体素化 vs Open3D风格VoxelGrid降采样
            if self.use_voxelization and voxelize_point_clouds is not None:
                try:
                    processed_source, processed_target = voxelize_point_clouds(
                        source_clean, target_clean, self.num_points, self.voxel_config, fallback_to_sampling=True)
                except Exception as e:
                    print(f"体素化处理失败，回退到重采样: {e}")
                    processed_source = self.resampler(torch.from_numpy(source_clean).float()).numpy()
                    processed_target = self.resampler(torch.from_numpy(target_clean).float()).numpy()
            else:
                processed_source = voxel_down_sample_numpy(source_clean, self.num_points)
                processed_target = voxel_down_sample_numpy(target_clean, self.num_points)

            # 新增: 确保点数等于 self.num_points
            if processed_source.shape[0] != self.num_points:
                processed_source = voxel_down_sample_numpy(processed_source, self.num_points)
            if processed_target.shape[0] != self.num_points:
                processed_target = voxel_down_sample_numpy(processed_target, self.num_points)

            # 根据 use_voxelization 分支归一化
            if self.use_voxelization and voxelize_point_clouds is not None:
                # 单独归一化 source
                source_tensor = torch.from_numpy(processed_source).float()
                min_vals = source_tensor.min(dim=0)[0]
                max_vals = source_tensor.max(dim=0)[0]
                center_s = (min_vals + max_vals) / 2.0
                scale_s = (max_vals - min_vals).max() if (max_vals - min_vals).max() > 1e-6 else 1.0
                source_tensor = (source_tensor - center_s) / scale_s

                # 单独归一化 target
                target_tensor = torch.from_numpy(processed_target).float()
                min_vals_t = target_tensor.min(dim=0)[0]
                max_vals_t = target_tensor.max(dim=0)[0]
                center_t = (min_vals_t + max_vals_t) / 2.0
                scale_t = (max_vals_t - min_vals_t).max() if (max_vals_t - min_vals_t).max() > 1e-6 else 1.0
                target_tensor = (target_tensor - center_t) / scale_t
            else:
                # 联合归一化（在 numpy 中统一中心化+缩放）
                all_pts = np.vstack([processed_source, processed_target])
                min_bounds = all_pts.min(axis=0)
                max_bounds = all_pts.max(axis=0)
                center = (min_bounds + max_bounds) / 2.0
                scales = max_bounds - min_bounds
                scale = scales.max() if scales.max() > 1e-6 else 1.0
                processed_source = (processed_source - center) / scale
                processed_target = (processed_target - center) / scale
                source_tensor = torch.from_numpy(processed_source).float()
                target_tensor = torch.from_numpy(processed_target).float()

            # 应用随机刚性变换
            transformed_source = self.rigid_transform(source_tensor)
            igt = self.rigid_transform.igt  # 获取真实的变换矩阵

            # 返回：模板、变换后的源、变换矩阵
            return target_tensor, transformed_source, igt
        except Exception as e:
            print(f"处理索引{idx}的点云时出错: {str(e)}")
            raise


class C3VDset4tracking_test(C3VDset4tracking):
    """用于测试的C3VD跟踪数据集，保留原始点云引用和文件路径"""
    
    def __init__(self, dataset, rigid_transform, num_points=1024, 
                 use_voxelization=True, voxel_config=None):
        """
        Args:
            dataset: C3VD基础数据集
            rigid_transform: 刚性变换生成器
            num_points: 目标点云点数
            use_voxelization: 是否使用复杂体素化处理，False时使用Open3D风格的VoxelGrid降采样
            voxel_config: 体素化配置
        """
        # 调用父类构造函数
        super().__init__(dataset, rigid_transform, num_points, use_voxelization, voxel_config)
        
        # 保存文件索引和点云对信息的字典
        self.cloud_info = {}
        
        # 收集所有可能的点云对信息
        if hasattr(dataset, 'pairs'):
            self.original_pairs = dataset.pairs
            self.original_pair_scenes = dataset.pair_scenes if hasattr(dataset, 'pair_scenes') else None
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'pairs'):
            self.original_pairs = dataset.dataset.pairs
            self.original_pair_scenes = dataset.dataset.pair_scenes if hasattr(dataset.dataset, 'pair_scenes') else None
        else:
            print("Warning: Unable to find original point cloud pair information")
            self.original_pairs = None
            self.original_pair_scenes = None
    
    def __getitem__(self, index):
        """获取测试数据项，同时保存原始点云信息"""
        try:
            # 1. 获取原始点云对
            source, target, _ = self.dataset[index]

            # 2. 转换为numpy数组并清理无效点
            source_np = source.numpy() if isinstance(source, torch.Tensor) else source
            target_np = target.numpy() if isinstance(target, torch.Tensor) else target
            
            source_mask = np.isfinite(source_np).all(axis=1)
            target_mask = np.isfinite(target_np).all(axis=1)
            
            source_clean = source_np[source_mask]
            target_clean = target_np[target_mask]
            
            if len(source_clean) < 100 or len(target_clean) < 100:
                raise ValueError(f"点云在索引{index}处清理后点数不足100个")

            # 3. 选择处理策略：复杂体素化 vs Open3D风格VoxelGrid降采样
            if self.use_voxelization and voxelize_point_clouds is not None:
                try:
                    processed_source, processed_target = voxelize_point_clouds(
                        source_clean, target_clean, self.num_points, self.voxel_config, fallback_to_sampling=True)
                except Exception as e:
                    print(f"体素化处理失败，回退到重采样: {e}")
                    source_tensor_tmp = torch.from_numpy(source_clean).float()
                    target_tensor_tmp = torch.from_numpy(target_clean).float()
                    processed_source = self.resampler(source_tensor_tmp).numpy()
                    processed_target = self.resampler(target_tensor_tmp).numpy()
            else:
                processed_source = voxel_down_sample_numpy(source_clean, self.num_points)
                processed_target = voxel_down_sample_numpy(target_clean, self.num_points)

            # 4. 确保点数等于 self.num_points (关键修复)
            if processed_source.shape[0] != self.num_points:
                processed_source = voxel_down_sample_numpy(processed_source, self.num_points)
            if processed_target.shape[0] != self.num_points:
                processed_target = voxel_down_sample_numpy(processed_target, self.num_points)

            # 转换回torch tensor
            source_tensor = torch.from_numpy(processed_source).float()
            target_tensor = torch.from_numpy(processed_target).float()
            
            # 5. 分别归一化
            source_min_vals = source_tensor.min(dim=0)[0]
            source_max_vals = source_tensor.max(dim=0)[0]
            source_center = (source_min_vals + source_max_vals) / 2
            source_scale = (source_max_vals - source_min_vals).max()
            if source_scale < 1e-10:
                raise ValueError(f"源点云归一化尺度过小，索引{index}: {source_scale}")
            source_normalized = (source_tensor - source_center) / source_scale
            
            target_min_vals = target_tensor.min(dim=0)[0]
            target_max_vals = target_tensor.max(dim=0)[0]
            target_center = (target_min_vals + target_max_vals) / 2
            target_scale = (target_max_vals - target_min_vals).max()
            if target_scale < 1e-10:
                raise ValueError(f"目标点云归一化尺度过小，索引{index}: {target_scale}")
            target_normalized = (target_tensor - target_center) / target_scale
            
            # 6. 应用刚性变换
            # 在测试模式下，rigid_transform 是一个特殊的类，它会根据索引应用固定的扰动
            transformed_source = self.rigid_transform(source_normalized)
            igt = self.rigid_transform.igt

            # 7. 收集原始点云信息
            if hasattr(self.dataset, 'indices') and self.original_pairs:
                orig_index = self.dataset.indices[index]
                source_file, target_file = self.original_pairs[orig_index]
                scene = self.original_pair_scenes[orig_index] if self.original_pair_scenes else "unknown"
            elif self.original_pairs and index < len(self.original_pairs):
                source_file, target_file = self.original_pairs[index]
                scene = self.original_pair_scenes[index] if self.original_pair_scenes else "unknown"
            else:
                source_file, target_file, scene = None, None, "unknown"

            scene_name = scene
            source_seq = "0000"
            if source_file:
                try:
                    norm_path = source_file.replace('\\', '/')
                    if scene_name == "unknown" and 'C3VD_ply_source' in norm_path:
                        parts = norm_path.split('/')
                        idx_part = [i for i, part in enumerate(parts) if part == 'C3VD_ply_source']
                        if idx_part and idx_part[0] + 1 < len(parts):
                            scene_name = parts[idx_part[0] + 1]
                    basename = os.path.basename(source_file)
                    if basename.endswith("_depth_pcd.ply") and basename[:4].isdigit():
                        source_seq = basename[:4]
                    else:
                        import re
                        numbers = re.findall(r'\\d+', basename)
                        if numbers:
                            source_seq = numbers[0].zfill(4)
                except Exception as e:
                    print(f"Warning: Error extracting scene name: {str(e)}")
            
            identifier = f"{scene_name}_{source_seq}_pert{index:04d}"

            # 使用 current_perturbation_index 可能是 test_pointlk.py 里的一个自定义实现，这里做兼容
            pert_idx = index
            if hasattr(self.rigid_transform, 'current_perturbation_index'):
                pert_idx = self.rigid_transform.current_perturbation_index - 1

            self.cloud_info[index] = {
                'identifier': identifier,
                'scene': scene_name,
                'sequence': source_seq,
                'source_file': source_file,
                'target_file': target_file,
                'sample_index': index,
                'perturbation_index': pert_idx,
                'original_source': source_normalized.clone(),
                'original_target': target_normalized.clone(),
                'igt': igt.clone() if igt is not None else None
            }

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
    """
    def __init__(self, dataset, rigid_transform, num_points=1024, 
                 use_voxelization=True, voxel_config=None, random_seed=42):
        """
        Args:
            dataset: C3VD基础数据集
            rigid_transform: 刚性变换生成器（包含所有扰动）
            num_points: 目标点云点数
            use_voxelization: 是否使用复杂体素化处理，False时使用Open3D风格的VoxelGrid降采样
            voxel_config: 体素化配置
            random_seed: 随机种子
        """
        # 调用父类构造函数
        super().__init__(dataset, rigid_transform, num_points, use_voxelization, voxel_config)
        
        # 从 rigid_transform 获取扰动数据
        if hasattr(rigid_transform, 'perturbations'):
            self.perturbations = rigid_transform.perturbations
        else:
            raise ValueError("rigid_transform must have perturbations attribute for random sampling mode")
        
        self.original_dataset_size = len(dataset)
        
        # 设置随机种子以确保可复现性
        self.random_seed = random_seed
        numpy.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # 为每个扰动预先生成随机样本索引
        self.sample_indices = numpy.random.randint(0, self.original_dataset_size, size=len(self.perturbations))
        
        print(f"C3VD Random sampling mode activated for gt_poses.csv:")
        print(f"- Total perturbations: {len(self.perturbations)}")
        print(f"- Original dataset size: {self.original_dataset_size}")
        print(f"- Random seed: {random_seed}")
        print(f"- Test iterations: {len(self.perturbations)} (one per perturbation)")
    
    def __len__(self):
        # 返回扰动数量，而不是数据集大小
        return len(self.perturbations)
    
    def __getitem__(self, index):
        """
        获取测试数据项，使用随机选择的样本和指定的扰动
        Args:
            index: 扰动索引 (0 到 len(perturbations)-1)
        """
        # 使用预先生成的随机样本索引
        sample_idx = self.sample_indices[index]
        
        try:
            # 1. 获取原始点云对（从随机选择的样本）
            source, target, _ = self.dataset[sample_idx]
            
            # 2. 转换为numpy数组并清理无效点
            source_np = source.numpy() if isinstance(source, torch.Tensor) else source
            target_np = target.numpy() if isinstance(target, torch.Tensor) else target
            
            source_mask = np.isfinite(source_np).all(axis=1)
            target_mask = np.isfinite(target_np).all(axis=1)
            
            source_clean = source_np[source_mask]
            target_clean = target_np[target_mask]
            
            if len(source_clean) < 100 or len(target_clean) < 100:
                raise ValueError(f"点云在样本索引{sample_idx}处清理后点数不足100个")
            
            # 3. 选择处理策略：复杂体素化 vs Open3D风格VoxelGrid降采样
            if self.use_voxelization and voxelize_point_clouds is not None:
                try:
                    processed_source, processed_target = voxelize_point_clouds(
                        source_clean, target_clean, self.num_points, self.voxel_config, fallback_to_sampling=True)
                except Exception as e:
                    print(f"体素化处理失败，回退到重采样: {e}")
                    source_tensor_tmp = torch.from_numpy(source_clean).float()
                    target_tensor_tmp = torch.from_numpy(target_clean).float()
                    processed_source = self.resampler(source_tensor_tmp).numpy()
                    processed_target = self.resampler(target_tensor_tmp).numpy()
            else:
                processed_source = voxel_down_sample_numpy(source_clean, self.num_points)
                processed_target = voxel_down_sample_numpy(target_clean, self.num_points)
            
            # 4. 确保点数等于 self.num_points (关键修复)
            if processed_source.shape[0] != self.num_points:
                processed_source = voxel_down_sample_numpy(processed_source, self.num_points)
            if processed_target.shape[0] != self.num_points:
                processed_target = voxel_down_sample_numpy(processed_target, self.num_points)

            # 转换回torch tensor
            source_tensor = torch.from_numpy(processed_source).float()
            target_tensor = torch.from_numpy(processed_target).float()
            
            # 5. 分别归一化
            source_min_vals = source_tensor.min(dim=0)[0]
            source_max_vals = source_tensor.max(dim=0)[0]
            source_center = (source_min_vals + source_max_vals) / 2
            source_scale = (source_max_vals - source_min_vals).max()
            if source_scale < 1e-10:
                raise ValueError(f"源点云归一化尺度过小，样本索引{sample_idx}: {source_scale}")
            source_normalized = (source_tensor - source_center) / source_scale
            
            target_min_vals = target_tensor.min(dim=0)[0]
            target_max_vals = target_tensor.max(dim=0)[0]
            target_center = (target_min_vals + target_max_vals) / 2
            target_scale = (target_max_vals - target_min_vals).max()
            if target_scale < 1e-10:
                raise ValueError(f"目标点云归一化尺度过小，样本索引{sample_idx}: {target_scale}")
            target_normalized = (target_tensor - target_center) / target_scale
            
            # 6. 应用特定的扰动
            perturbation = self.perturbations[index]
            twist = torch.from_numpy(numpy.array(perturbation)).contiguous().view(1, 6)
            x = twist.to(source_normalized)
            
            if not getattr(self.rigid_transform, 'fmt_trans', False):
                g = se3.exp(x).to(source_normalized)
                transformed_source = se3.transform(g, source_normalized)
                igt = g.squeeze(0)
            else:
                w = x[:, 0:3]
                q = x[:, 3:6]
                R = so3.exp(w).to(source_normalized)
                g = torch.zeros(1, 4, 4)
                g[:, 3, 3] = 1
                g[:, 0:3, 0:3] = R
                g[:, 0:3, 3] = q
                transformed_source = se3.transform(g, source_normalized)
                igt = g.squeeze(0)

            # 7. 收集原始点云信息
            if hasattr(self.dataset, 'indices') and self.original_pairs:
                orig_index = self.dataset.indices[sample_idx]
                source_file, target_file = self.original_pairs[orig_index]
                scene = self.original_pair_scenes[orig_index] if self.original_pair_scenes else "unknown"
            elif self.original_pairs and sample_idx < len(self.original_pairs):
                source_file, target_file = self.original_pairs[sample_idx]
                scene = self.original_pair_scenes[sample_idx] if self.original_pair_scenes else "unknown"
            else:
                source_file, target_file, scene = None, None, "unknown"

            scene_name = scene
            source_seq = "0000"
            if source_file:
                try:
                    norm_path = source_file.replace('\\', '/')
                    if scene_name == "unknown" and 'C3VD_ply_source' in norm_path:
                        parts = norm_path.split('/')
                        idx_part = [i for i, part in enumerate(parts) if part == 'C3VD_ply_source']
                        if idx_part and idx_part[0] + 1 < len(parts):
                            scene_name = parts[idx_part[0] + 1]
                    basename = os.path.basename(source_file)
                    if basename.endswith("_depth_pcd.ply") and basename[:4].isdigit():
                        source_seq = basename[:4]
                    else:
                        import re
                        numbers = re.findall(r'\\d+', basename)
                        if numbers:
                            source_seq = numbers[0].zfill(4)
                except Exception as e:
                    print(f"Warning: Error extracting scene name: {str(e)}")

            identifier = f"{scene_name}_{source_seq}_pert{index:04d}"

            self.cloud_info[index] = {
                'identifier': identifier,
                'scene': scene_name,
                'sequence': source_seq,
                'source_file': source_file,
                'target_file': target_file,
                'sample_index': sample_idx,
                'perturbation_index': index,
                'original_source': source_normalized.clone(),
                'original_target': target_normalized.clone(),
                'igt': igt.clone() if igt is not None else None
            }

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