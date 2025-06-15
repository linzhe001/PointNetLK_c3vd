""" datasets """

import numpy
import torch
import torch.utils.data
import os
import glob
from plyfile import PlyData
import numpy as np

from . import globset
from . import mesh
from . import transforms
from .. import so3
from .. import se3


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
    """Tracking dataset for C3VD that applies rigid transformations to point clouds."""
    
    def __init__(self, dataset, rigid_transform, num_points=1024):
        self.dataset = dataset
        self.rigid_transform = rigid_transform  # 添加刚性变换
        self.num_points = num_points
        self.resampler = transforms.Resampler(num_points)
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        try:
            # 获取原始点云对
            source, target, _ = self.dataset[idx]  # 忽略原始的igt
            
            # 数据清理和重采样
            source = source[torch.isfinite(source).all(dim=1)]
            target = target[torch.isfinite(target).all(dim=1)]
            
            if source.shape[0] < 100 or target.shape[0] < 100:
                raise ValueError(f"Point cloud at index {idx} has less than 100 points after cleaning")
            
            source = self.resampler(source)
            target = self.resampler(target)
            
            # 联合归一化
            combined = torch.cat([source, target], dim=0)
            min_vals = combined.min(dim=0)[0]
            max_vals = combined.max(dim=0)[0]
            center = (min_vals + max_vals) / 2
            scale = (max_vals - min_vals).max()
            
            if scale < 1e-10:
                raise ValueError(f"Scale too small for point cloud at index {idx}: {scale}")
                
            source = (source - center) / scale
            target = (target - center) / scale
            
            # 应用随机刚性变换
            # target作为模板，source经过变换后作为源点云
            transformed_source = self.rigid_transform(source)
            igt = self.rigid_transform.igt  # 获取真实的变换矩阵
            
            # 返回：模板、变换后的源、变换矩阵
            return target, transformed_source, igt
            
        except Exception as e:
            print(f"Error processing point cloud at index {idx}: {str(e)}")
            raise

class C3VDset4tracking_test(C3VDset4tracking):
    """用于测试的C3VD跟踪数据集，保留原始点云引用和文件路径
    
    此类扩展了C3VDset4tracking类，添加了以下功能：
    1. 保留原始点云数据的引用和文件路径
    2. 为每个点云对创建唯一索引，方便后续引用
    3. 提供获取原始未变换点云的方法
    """
    def __init__(self, dataset, rigid_transform, use_joint_normalization=True, num_points=1024):
        # 只传递父类所需的参数
        super().__init__(dataset, num_points)
        
        # 保存其他参数
        self.rigid_transform = rigid_transform
        self.use_joint_normalization = use_joint_normalization
        
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
        # 获取原始点云数据
        try:
            source, target, _ = self.dataset[index]
            
            # 移除无效点 (NaN/Inf)
            source = source[torch.isfinite(source).all(dim=1)]
            target = target[torch.isfinite(target).all(dim=1)]
            
            # 确保处理后的点云有足够的点
            if source.shape[0] < 100 or target.shape[0] < 100:
                raise ValueError(f"Point cloud at index {index} has less than 100 points after cleaning")
            
            # 重采样点云以确保相同数量的点
            source = self.resampler(source)
            target = self.resampler(target)
            
            # 如果启用联合归一化，对两个点云一起归一化
            if hasattr(self, 'use_joint_normalization') and self.use_joint_normalization:
                # 合并两个点云
                combined = torch.cat([source, target], dim=0)
                
                # 计算联合边界框
                min_vals = combined.min(dim=0)[0]
                max_vals = combined.max(dim=0)[0]
                center = (min_vals + max_vals) / 2
                scale = (max_vals - min_vals).max()
                
                if scale < 1e-10:
                    raise ValueError(f"Scale too small for point cloud at index {index}: {scale}")
                    
                # 对两个点云应用相同的归一化
                source = (source - center) / scale
                target = (target - center) / scale
            
            # 应用刚性变换
            if hasattr(self, 'rigid_transform') and self.rigid_transform is not None:
                transformed_source = self.rigid_transform(source)
                igt = self.rigid_transform.igt
            else:
                transformed_source = source
                igt = torch.eye(4)
            
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
                'original_source': source.clone(),
                'original_target': target.clone(),
                'igt': igt.clone() if igt is not None else None
            }
            
            return target, transformed_source, igt
            
        except Exception as e:
            print(f"Error processing point cloud at index {index}: {str(e)}")
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

# EOF
