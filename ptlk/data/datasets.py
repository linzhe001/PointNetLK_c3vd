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
    """C3VD dataset loader that handles pairs of source and target point clouds.
    
    This dataset is designed for the C3VD (Challenging 3D Vision Dataset) which contains
    multiple scenes with source and target point cloud pairs for point cloud registration tasks.
    """
    def __init__(self, source_root, target_root, transform=None):
        self.source_root = source_root
        self.target_root = target_root
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
        for scene_dir in glob.glob(os.path.join(source_root, "*")):
            if os.path.isdir(scene_dir):
                scene_name = os.path.basename(scene_dir)
                self.scenes.append(scene_name)
        
        # Get point cloud pairs for each scene
        self.pairs = []
        self.pair_scenes = []  # Added: track the scene for each point cloud pair
        
        for scene in self.scenes:
            # Confirm target point cloud exists
            target_file = os.path.join(target_root, scene, "coverage_mesh.ply")
            if not os.path.exists(target_file):
                continue
                
            # Get all source point clouds in the scene
            source_files = glob.glob(os.path.join(source_root, scene, "????_adjusted.ply"))
            for source_file in source_files:
                # All source point clouds are paired with the scene's coverage_mesh.ply
                self.pairs.append((source_file, target_file))
                self.pair_scenes.append(scene)  # Record the scene name
    
    def get_scene_indices(self, scene_names):
        """Get indices of all samples from specified scenes
        
        Args:
            scene_names: List of scene names to filter by
            
        Returns:
            List of indices that correspond to the specified scenes
        """
        indices = []
        for i, scene in enumerate(self.pair_scenes):
            if scene in scene_names:
                indices.append(i)
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
        
        # Final check for point cloud size
        if source.shape[0] != self.num_points or target.shape[0] != self.num_points:
            raise ValueError(f"Incorrect point cloud size at index {idx}: source={source.shape[0]}, target={target.shape[0]}, expected={self.num_points}")
        
        return source, target, None

class C3VDset4tracking(torch.utils.data.Dataset):
    """Tracking dataset for C3VD that applies rigid transformations to point clouds.
    
    This dataset wraps a C3VDDataset and applies rigid transforms to prepare data
    for point cloud registration/tracking tasks.
    """
    def __init__(self, dataset, rigid_transform):
        self.dataset = dataset
        self.rigid_transform = rigid_transform
        
        # Get point cloud size
        self.num_points = 1024  # Default value
        if hasattr(dataset, 'transform') and hasattr(dataset.transform, 'transforms'):
            for t in dataset.transform.transforms:
                if hasattr(t, 'num_points'):
                    self.num_points = t.num_points
                    break
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        try:
            source, target, _ = self.dataset[index]
            
            # Check validity of input point clouds
            if not torch.isfinite(source).all():
                print(f"Warning: Source point cloud at index {index} contains invalid values, attempting to fix")
                source = torch.nan_to_num(source, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if not torch.isfinite(target).all():
                print(f"Warning: Target point cloud at index {index} contains invalid values, attempting to fix")
                target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Ensure point clouds are not empty and have correct size
            if source.shape[0] != self.num_points or target.shape[0] != self.num_points:
                raise ValueError(f"Incorrect point cloud size: source={source.shape[0]}, target={target.shape[0]}, expected={self.num_points}")
            
            # Check point cloud range before applying transformation
            if torch.any(torch.abs(source) > 100) or torch.any(torch.abs(target) > 100):
                print(f"Warning: Point cloud at index {index} has abnormal value range, normalizing")
                source = source / max(1.0, source.abs().max())
                target = target / max(1.0, target.abs().max())
            
            # Apply rigid transformation
            try:
                transformed_source = self.rigid_transform(source)
                igt = self.rigid_transform.igt  # Get inverse transformation matrix
            except Exception as e:
                print(f"Warning: Error applying transformation at index {index}: {str(e)}")
                raise
            
            # Check transformation results
            if not torch.isfinite(transformed_source).all():
                print(f"Warning: Transformed source at index {index} contains invalid values, attempting to fix")
                transformed_source = torch.nan_to_num(transformed_source, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if not torch.isfinite(igt).all():
                print(f"Warning: Transformation matrix at index {index} contains invalid values")
                raise ValueError("Transformation matrix contains invalid values")
            
            # Final check
            if torch.any(torch.abs(transformed_source) > 100):
                print(f"Warning: Transformed point cloud at index {index} has abnormal value range, normalizing")
                transformed_source = transformed_source / max(1.0, transformed_source.abs().max())
            
            return target, transformed_source, igt
            
        except Exception as e:
            print(f"Error: Failed to process index {index}: {str(e)}")
            raise


#EOF
