"""
    Example for testing PointNet-LK.

    No-noise version.
"""

import argparse
import os
import sys
import logging
import numpy
import torch
import torch.utils.data
import torchvision
import time
import random
import shutil
from plyfile import PlyData, PlyElement
import numpy as np
import traceback
import warnings
from scipy.spatial.transform import Rotation
import math
import glob

# 抑制CUDA警告
# Suppress CUDA warnings
warnings.filterwarnings('ignore', category=UserWarning)

# addpath('../')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import ptlk
# 添加attention模块导入
# Import attention module
from ptlk import attention_v1
from ptlk import mamba3d_v1  # 导入Mamba3D模块
from ptlk import fast_point_attention  # 导入快速点注意力模块
from ptlk import cformer  # 导入Cformer模块

# 添加必要的导入
# Add necessary imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ptlk.data.datasets as datasets
from ptlk.data.datasets import ModelNet, ShapeNet2, C3VDDataset, C3VDset4tracking, C3VDset4tracking_test, VoxelizationConfig
#from ptlk.data.datasets import SinglePairDataset, SinglePairTrackingDataset  # 新增导入
from ptlk.data.datasets import C3VDset4tracking_test_random_sample, CADset4tracking_fixed_perturbation_random_sample
import ptlk.data.transforms as transforms
import ptlk.pointlk as pointlk
from ptlk import so3, se3  # 修改导入方式


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def options(argv=None):
    parser = argparse.ArgumentParser(description='PointNet-LK')

    # required.
    parser.add_argument('-o', '--outfile', required=True, type=str,
                        metavar='FILENAME', help='output filename (.csv)')
    parser.add_argument('-i', '--dataset-path', required=True, type=str,
                        metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
    parser.add_argument('-c', '--categoryfile', required=True, type=str,
                        metavar='PATH', help='path to the categories to be tested') # eg. './sampledata/modelnet40_half1.txt'
    parser.add_argument('-p', '--perturbations', required=False, type=str,
                        metavar='PATH', help='path to the perturbation file') # see. generate_perturbations.py

    # settings for input data
    parser.add_argument('--dataset-type', default='modelnet', choices=['modelnet', 'c3vd'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--format', default='wv', choices=['wv', 'wt'],
                        help='perturbation format (default: wv (twist)) (wt: rotation and translation)') # the output is always in twist format
    parser.add_argument('--num-points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')
    # C3VD 配对模式设置
    # C3VD pairing mode settings
    parser.add_argument('--pair-mode', default='one_to_one', choices=['one_to_one', 'scene_reference'],
                        help='Point cloud pairing mode: one_to_one (each source cloud pairs with specific target cloud) or scene_reference (each scene uses one shared target cloud)')
    parser.add_argument('--reference-name', default=None, type=str,
                        help='Target cloud name used in scene reference mode, default uses first cloud in scene')

    # settings for PointNet-LK
    parser.add_argument('--max-iter', default=20, type=int,
                        metavar='N', help='max-iter on LK. (default: 20)')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to trained model file (default: null (no-use))')
    parser.add_argument('--transfer-from', default='', type=str,
                        metavar='PATH', help='path to classifier feature (default: null (no-use))')
    parser.add_argument('--dim-k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')
    parser.add_argument('--delta', default=1.0e-2, type=float,
                        metavar='D', help='step size for approx. Jacobian (default: 1.0e-2)')
    
    # 添加模型选择参数 (与train_pointlk.py保持一致)
    # Add model selection parameter (consistent with train_pointlk.py)
    parser.add_argument('--model-type', default='pointnet', choices=['pointnet', 'attention', 'mamba3d', 'fast_attention', 'cformer'],
                        help='Select model type: pointnet, attention, mamba3d, fast_attention or cformer (default: pointnet)')
    
    # 添加attention模型特定参数 (与train_pointlk.py保持一致)
    # Add attention model specific parameters (consistent with train_pointlk.py)
    parser.add_argument('--num-attention-blocks', default=3, type=int,
                        metavar='N', help='Number of attention blocks in attention module (default: 3)')
    parser.add_argument('--num-heads', default=8, type=int,
                        metavar='N', help='Number of heads in multi-head attention (default: 8)')
    
    # 添加Mamba3D模型特定参数
    parser.add_argument('--num-mamba-blocks', default=3, type=int,
                        metavar='N', help='Number of Mamba blocks in Mamba3D module (default: 3)')
    parser.add_argument('--d-state', default=16, type=int,
                        metavar='N', help='Mamba state space dimension (default: 16)')
    parser.add_argument('--expand', default=2, type=float,
                        metavar='N', help='Mamba expansion factor (default: 2)')
    
    # 添加快速点注意力模型特定参数
    parser.add_argument('--num-fast-attention-blocks', default=2, type=int,
                        metavar='N', help='Number of attention blocks in fast point attention module (default: 2)')
    parser.add_argument('--fast-attention-scale', default=1, type=int,
                        metavar='N', help='Scale factor for fast point attention model (default: 1, larger values mean lighter model)')
    
    # 添加Cformer模型特定参数
    parser.add_argument('--num-proxy-points', default=8, type=int,
                        metavar='N', help='Number of proxy points in Cformer model (default: 8)')
    parser.add_argument('--num-blocks', default=2, type=int,
                        metavar='N', help='Number of blocks in Cformer model (default: 2)')

    # settings for on testing
    parser.add_argument('-l', '--logfile', default='', type=str,
                        metavar='LOGNAME', help='path to logfile (default: null (no logging))')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--device', default='cpu', type=str,
                        metavar='DEVICE', help='use CUDA if available (default: cpu)')
    parser.add_argument('--max-samples', default=2000, type=int,
                        metavar='N', help='Maximum number of test samples (default: 2000)')
                        
    # 新增 - 可视化设置
    # New - visualization settings
    parser.add_argument('--visualize-pert', default=None, type=str, nargs='*',
                        help='List of perturbation filenames to visualize (e.g. pert_010.csv pert_020.csv), path not included')
    parser.add_argument('--visualize-samples', default=1, type=int,
                        help='Number of samples to visualize per perturbation file (default: 1)')
                        

    # 体素化相关参数（参考train_pointlk.py）
    # Voxelization related parameters (reference train_pointlk.py)
    parser.add_argument('--use-voxelization', action='store_true', default=True,
                        help='启用体素化预处理方法 (默认: True)')
    parser.add_argument('--no-voxelization', dest='use_voxelization', action='store_false',
                        help='禁用体素化，使用简单重采样方法')
    parser.add_argument('--voxel-size', default=0.05, type=float,
                        metavar='SIZE', help='体素大小 (默认: 0.05)')
    parser.add_argument('--voxel-grid-size', default=32, type=int,
                        metavar='SIZE', help='体素网格尺寸 (默认: 32)')
    parser.add_argument('--max-voxel-points', default=100, type=int,
                        metavar='N', help='每个体素最大点数 (默认: 100)')
    parser.add_argument('--max-voxels', default=20000, type=int,
                        metavar='N', help='最大体素数量 (默认: 20000)')
    parser.add_argument('--min-voxel-points-ratio', default=0.1, type=float,
                        metavar='RATIO', help='最小体素点数比例阈值 (默认: 0.1)')

    # 1. 在options函数中添加新参数，用于接收扰动文件夹
    # 1. Add new parameter in options function to receive perturbation directory
    parser.add_argument('--perturbation-dir', default=None, type=str,
                        metavar='PATH', help='Perturbation directory path, will process all perturbation files in the directory')
                        
    # 添加单个扰动文件参数支持 (与脚本兼容)
    # Add single perturbation file parameter support (compatible with scripts)
    parser.add_argument('--perturbation-file', default=None, type=str,
                        metavar='PATH', help='Single perturbation file path (e.g., gt_poses.csv)')

    # 新增：单对点云输入模式参数
    # New: Single pair point cloud input mode parameters
    parser.add_argument('--single-pair-mode', action='store_true', default=False,
                        help='启用单对点云输入模式 (Enable single pair point cloud input mode)')
    parser.add_argument('--source-cloud', default=None, type=str,
                        metavar='PATH', help='源点云文件路径 (Source point cloud file path)')
    parser.add_argument('--target-cloud', default=None, type=str,
                        metavar='PATH', help='目标点云文件路径 (Target point cloud file path)')
    parser.add_argument('--single-perturbation', default=None, type=str,
                        metavar='VALUES', help='单行扰动值，逗号分隔 (Single perturbation values, comma-separated). 格式: rx,ry,rz,tx,ty,tz')
    parser.add_argument('--enhanced-output', action='store_true', default=False,
                        help='输出增强信息，包括传入的扰动和预测的变换 (Output enhanced information including input perturbation and predicted transformation)')

    args = parser.parse_args(argv)
    return args

def main(args):
    # 新增：单对点云输入模式处理
    # New: Single pair point cloud input mode processing
    if args.single_pair_mode:
        # 验证单对点云模式的必要参数
        if not args.source_cloud or not args.target_cloud:
            print("错误: 单对点云模式需要指定 --source-cloud 和 --target-cloud 参数")
            print("Error: Single pair mode requires --source-cloud and --target-cloud parameters")
            return
        
        if not args.single_perturbation:
            print("错误: 单对点云模式需要指定 --single-perturbation 参数")
            print("Error: Single pair mode requires --single-perturbation parameter")
            return
        
        # 检查点云文件是否存在
        if not os.path.exists(args.source_cloud):
            print(f"错误: 源点云文件不存在: {args.source_cloud}")
            print(f"Error: Source cloud file does not exist: {args.source_cloud}")
            return
        
        if not os.path.exists(args.target_cloud):
            print(f"错误: 目标点云文件不存在: {args.target_cloud}")
            print(f"Error: Target cloud file does not exist: {args.target_cloud}")
            return
        
        print(f"\n====== 单对点云输入模式 Single Pair Point Cloud Mode ======")
        print(f"源点云 Source cloud: {args.source_cloud}")
        print(f"目标点云 Target cloud: {args.target_cloud}")
        print(f"扰动值 Perturbation: {args.single_perturbation}")
        print(f"增强输出 Enhanced output: {args.enhanced_output}")
        
        # 调用单对点云处理函数
        process_single_pair(args)
        return
    
    # 原有的批量处理逻辑保持不变
    # 创建空列表存储所有要处理的扰动文件
    # Create empty list to store all perturbation files to process
    perturbation_files = []
    
    # 如果指定了扰动文件夹，先添加文件夹中的所有.csv文件
    # If perturbation directory is specified, first add all .csv files in the directory
    if args.perturbation_dir and os.path.exists(args.perturbation_dir):
        print(f"\n====== Perturbation Directory ======")
        print(f"Scanning perturbation directory: {args.perturbation_dir}")
        for filename in sorted(os.listdir(args.perturbation_dir)):
            if filename.endswith('.csv'):
                full_path = os.path.join(args.perturbation_dir, filename)
                perturbation_files.append(full_path)
                print(f"Found perturbation file: {filename}")
    
    # 如果指定了单独的扰动文件（通过--perturbations），也添加进列表
    # If individual perturbation file is specified (via --perturbations), add it to the list
    if args.perturbations and os.path.exists(args.perturbations):
        if args.perturbations not in perturbation_files:
            perturbation_files.append(args.perturbations)
            print(f"Added individually specified perturbation file: {os.path.basename(args.perturbations)}")
    
    # 如果指定了单独的扰动文件（通过--perturbation-file），也添加进列表
    # If individual perturbation file is specified (via --perturbation-file), add it to the list
    if args.perturbation_file and os.path.exists(args.perturbation_file):
        if args.perturbation_file not in perturbation_files:
            perturbation_files.append(args.perturbation_file)
            print(f"Added perturbation file: {os.path.basename(args.perturbation_file)}")
    
    # 检查是否有扰动文件要处理
    # Check if there are perturbation files to process
    if not perturbation_files:
        print("Error: No perturbation files found. Please use --perturbation-dir to specify perturbation directory, --perturbations to specify perturbation file, or --perturbation-file to specify perturbation file.")
        return
    
    print(f"Total found {len(perturbation_files)} perturbation files to process")
    
    # 创建动作执行器，但不立即传入扰动文件
    # Create action executor, but don't pass perturbation files immediately
    act = Action(args)
    
    # 依次处理每个扰动文件
    # Process each perturbation file sequentially
    for i, pert_file in enumerate(perturbation_files):
        filename = os.path.basename(pert_file)
        print(f"\n====== Processing perturbation file [{i+1}/{len(perturbation_files)}]: {filename} ======")
        
        # 保存原始参数值
        # Save original parameter values
        original_perturbations = args.perturbations
        original_outfile = args.outfile
        original_logfile = args.logfile
        
        # 判断是否为单个文件处理（如果只有一个文件且是直接提供的文件）
        # Determine if it's single file processing (if only one file and directly provided)
        is_single_file = len(perturbation_files) == 1 and (args.perturbations == pert_file or args.perturbation_file == pert_file)
        
        # 提取扰动角度信息（如果有）
        # Extract perturbation angle information (if any)
        angle_str = ""
        if filename.startswith("pert_") and "_" in filename:
            parts = filename.split("_")
            if len(parts) >= 2:
                angle_str = parts[1].split(".")[0]
        
        # 为当前扰动文件创建输出文件名
        # Create output filename for current perturbation file
        if angle_str:
            # 为每个角度创建单独目录
            output_dir = os.path.join(os.path.dirname(args.outfile), f"angle_{angle_str}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 保持与原始CSV文件相同的命名结构，只更改扩展名为.log
            base_filename = os.path.splitext(filename)[0]
            current_outfile = os.path.join(output_dir, f"results_{base_filename}.log")
            log_file = os.path.join(output_dir, f"log_{base_filename}.log")
        else:
            # 对于没有角度信息的扰动文件，创建基本目录
            output_dir = os.path.dirname(args.outfile)
            os.makedirs(output_dir, exist_ok=True)
            
            # 使用扰动文件名作为输出文件名
            base_filename = os.path.splitext(filename)[0]
            current_outfile = os.path.join(output_dir, f"results_{base_filename}.log")
            log_file = os.path.join(output_dir, f"log_{base_filename}.log")
        
        # 设置当前参数值
        # Set current parameter values
        args.perturbations = pert_file
        args.outfile = current_outfile
        args.logfile = log_file
        
        print(f"Output file: {args.outfile}")
        print(f"Log file: {args.logfile}")
        
        # 为当前扰动文件获取数据集
        # Get dataset for current perturbation file
        testset = get_datasets(args)
        
        # 更新动作执行器的文件名和当前扰动文件
        # Update action executor's filename and current perturbation file
        act.update_perturbation(args.perturbations, current_outfile)
        
        # 运行测试
        # Run test
        run(args, testset, act)
        
        # 恢复原始参数
        # Restore original parameters
        args.perturbations = original_perturbations
        args.outfile = original_outfile
        args.logfile = original_logfile
        
        # 清理内存
        # Clean up memory
        del testset
        torch.cuda.empty_cache()
        import gc
        gc.collect()


def run(args, testset, action):
    # Custom dataset wrapper that handles exceptions
    class DatasetWrapper(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.valid_indices = []
            self.pre_check_dataset()
            
        def pre_check_dataset(self):
            """预检查数据集中的少量样本"""
            """Pre-check a few samples in the dataset"""
            print(f"Starting to check samples in the dataset...")
            total_samples = len(self.dataset)
            
            # 只检查前10个样本或总数的1%，取较小值
            # Only check first 10 samples or 1% of total, whichever is smaller
            check_count = min(10, max(1, total_samples // 100))
            print(f"Checking {check_count} samples out of {total_samples} total samples...")
            
            # 快速检查少量样本
            # Quick check of few samples
            valid_count = 0
            for idx in range(check_count):
                try:
                    # 尝试获取样本
                    # Try to get sample
                    p0, p1, igt = self.dataset[idx]
                    
                    # 检查点云格式和大小
                    # Check point cloud format and size
                    if not isinstance(p0, torch.Tensor) or not isinstance(p1, torch.Tensor):
                        print(f"Warning: Sample {idx} has incorrect point cloud format")
                        continue
                        
                    # 检查变换矩阵
                    # Check transformation matrix
                    if igt is None or not isinstance(igt, torch.Tensor):
                        print(f"Warning: Sample {idx} has invalid transformation matrix")
                        continue
                    
                    # 检查是否有有效值
                    # Check if there are valid values
                    if not torch.isfinite(p0).all() or not torch.isfinite(p1).all() or not torch.isfinite(igt).all():
                        print(f"Warning: Sample {idx} contains non-finite values")
                        continue
                    
                    # 检查点云大小是否合理
                    # Check if point cloud size is reasonable
                    if p0.shape[0] == 0 or p1.shape[0] == 0:
                        print(f"Warning: Sample {idx} has empty point cloud")
                        continue
                    
                    # 如果样本有效，计数增加
                    # If sample is valid, increment count
                    valid_count += 1
                    
                except Exception as e:
                    print(f"Warning: Sample {idx} failed validation: {str(e)}")
                    continue
            
            print(f"Validation completed: {valid_count}/{check_count} samples are valid")
            
            # 假设所有样本都有效（基于少量样本的检查结果）
            # Assume all samples are valid (based on small sample check results)
            if valid_count > 0:
                print(f"Dataset appears healthy, assuming all {total_samples} samples are usable")
                self.valid_indices = list(range(total_samples))
            else:
                print(f"Warning: No valid samples found in initial check, but continuing with full dataset")
                self.valid_indices = list(range(total_samples))
        
        def __len__(self):
            return len(self.valid_indices)
        
        def __getitem__(self, idx):
            try:
                return self.dataset[self.valid_indices[idx]]
            except Exception as e:
                print(f"Warning: Failed to get sample {idx}: {str(e)}")
                # 返回最后一个有效样本
                # Return the last valid sample
                return self.dataset[self.valid_indices[-1]]
            
        def get_cloud_info(self, idx):
            """获取点云信息"""
            """Get point cloud information"""
            try:
                # 通过自定义索引访问数据集
                # Access dataset through custom index
                real_idx = self.valid_indices[idx]
                
                # 尝试通过调用数据集的方法获取点云信息
                # Try to get point cloud info through dataset method
                if hasattr(self.dataset, 'get_cloud_info'):
                    info = self.dataset.get_cloud_info(real_idx)
                    return info
                    
                # 如果数据集没有get_cloud_info方法，尝试直接访问cloud_info属性
                # If dataset doesn't have get_cloud_info method, try to access cloud_info attribute directly
                elif hasattr(self.dataset, 'cloud_info'):
                    if real_idx in self.dataset.cloud_info:
                        return self.dataset.cloud_info[real_idx]
                
                # 如果数据集是Subset，尝试访问原始数据集
                # If dataset is Subset, try to access original dataset
                elif hasattr(self.dataset, 'dataset'):
                    if hasattr(self.dataset.dataset, 'get_cloud_info'):
                        # 如果Subset有自己的indices
                        # If Subset has its own indices
                        if hasattr(self.dataset, 'indices'):
                            orig_idx = self.dataset.indices[real_idx]
                            return self.dataset.dataset.get_cloud_info(orig_idx)
                    
                    # 尝试直接访问原始数据集的cloud_info
                    # Try to access cloud_info of original dataset directly
                    elif hasattr(self.dataset.dataset, 'cloud_info'):
                        if hasattr(self.dataset, 'indices'):
                            orig_idx = self.dataset.indices[real_idx]
                            if orig_idx in self.dataset.dataset.cloud_info:
                                return self.dataset.dataset.cloud_info[orig_idx]
                
                # 构建默认信息
                # 尝试获取源文件和目标文件信息
                source_file = None
                target_file = None
                
                try:
                    # 尝试获取点云对文件路径信息
                    if hasattr(self.dataset, 'pairs'):
                        if real_idx < len(self.dataset.pairs):
                            source_file, target_file = self.dataset.pairs[real_idx]
                    elif hasattr(self.dataset, 'dataset') and hasattr(self.dataset.dataset, 'pairs'):
                        if hasattr(self.dataset, 'indices'):
                            orig_idx = self.dataset.indices[real_idx]
                            if orig_idx < len(self.dataset.dataset.pairs):
                                source_file, target_file = self.dataset.dataset.pairs[orig_idx]
                except Exception as e:
                    print(f"获取文件路径时出错: {str(e)}")
                
                # 从路径中提取信息
                scene_name = "unknown"
                source_seq = f"{real_idx:04d}"
                
                if source_file:
                    # 标准化路径分隔符
                    norm_path = source_file.replace('\\', '/')
                    
                    # 提取场景名称
                    if 'C3VD_ply_source' in norm_path:
                        parts = norm_path.split('/')
                        idx = [i for i, part in enumerate(parts) if part == 'C3VD_ply_source']
                        if idx and idx[0] + 1 < len(parts):
                            scene_name = parts[idx[0] + 1]
                    
                    # 提取序列号
                    basename = os.path.basename(source_file)
                    if basename.endswith("_depth_pcd.ply") and basename[:4].isdigit():
                        source_seq = basename[:4]
                    else:
                        import re
                        numbers = re.findall(r'\d+', basename)
                        if numbers:
                            source_seq = numbers[0].zfill(4)
                
                # 构造信息字典
                return {
                    'identifier': f"{scene_name}_{source_seq}",
                    'scene': scene_name,
                    'sequence': source_seq,
                    'source_file': source_file,
                    'target_file': target_file
                }
                
            except Exception as e:
                print(f"警告: 获取点云信息失败: {str(e)}")
                return None
            
        def get_original_clouds(self, idx):
            """获取原始点云"""
            try:
                # 通过自定义索引访问数据集
                real_idx = self.valid_indices[idx]
                
                # 尝试通过数据集的方法获取原始点云
                if hasattr(self.dataset, 'get_original_clouds'):
                    original_source, original_target = self.dataset.get_original_clouds(real_idx)
                    if original_source is not None and original_target is not None:
                        return original_source, original_target
                
                # 如果是Subset，尝试访问底层数据集
                elif hasattr(self.dataset, 'dataset'):
                    if hasattr(self.dataset.dataset, 'get_original_clouds'):
                        if hasattr(self.dataset, 'indices'):
                            orig_idx = self.dataset.indices[real_idx]
                            return self.dataset.dataset.get_original_clouds(orig_idx)
                
                # 尝试从cloud_info中获取
                info = self.get_cloud_info(idx)
                if info and 'original_source' in info and 'original_target' in info:
                    return info['original_source'], info['original_target']
                
                # 如果还是获取不到，尝试直接使用当前样本作为"原始"点云
                try:
                    p0, p1, _ = self.dataset[real_idx]
                    return p1.clone(), p0.clone()  # 注意这里的顺序，p1是源，p0是目标
                except:
                    pass
                
                # 如果都失败了，返回None
                return None, None
                
            except Exception as e:
                print(f"警告: 获取原始点云失败: {str(e)}")
                return None, None
            
        def get_identifier(self, idx):
            """获取点云标识符"""
            try:
                # 通过自定义索引访问数据集
                real_idx = self.valid_indices[idx]
                
                # 尝试调用数据集的get_identifier方法
                if hasattr(self.dataset, 'get_identifier'):
                    identifier = self.dataset.get_identifier(real_idx)
                    if identifier:
                        return identifier
                
                # 如果是Subset，尝试访问原始数据集
                elif hasattr(self.dataset, 'dataset'):
                    if hasattr(self.dataset.dataset, 'get_identifier'):
                        if hasattr(self.dataset, 'indices'):
                            orig_idx = self.dataset.indices[real_idx]
                            identifier = self.dataset.dataset.get_identifier(orig_idx)
                            if identifier:
                                return identifier
                
                # 尝试从cloud_info获取
                info = self.get_cloud_info(idx)
                if info and 'identifier' in info:
                    return info['identifier']
                
                # 构造一个基本标识符
                if info:
                    scene = info.get('scene', 'unknown')
                    seq = info.get('sequence', f"{real_idx:04d}")
                    return f"{scene}_{seq}"
                
                # 最后的后备选项
                return f"unknown_{real_idx:04d}"
                
            except Exception as e:
                print(f"警告: 获取标识符失败: {str(e)}")
                return f"unknown_{idx:04d}"

    # CUDA检查
    # CUDA availability check
    print(f"\n====== CUDA Availability Check ======")
    if torch.cuda.is_available():
        print(f"CUDA available: Yes")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print(f"CUDA available: No (testing will run on CPU, which will be slow)")
        args.device = 'cpu'
    
    args.device = torch.device(args.device)
    print(f"Using device: {args.device}")

    LOGGER.debug('Testing (PID=%d), %s', os.getpid(), args)

    model = action.create_model()
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    model.to(args.device)

    # 显示模型信息
    # Display model information
    print(f"\n====== Model Information ======")
    print(f"Parameter count: {sum(p.numel() for p in model.parameters())}")
    print(f"Model parameters on CUDA: {next(model.parameters()).is_cuda}")
    if str(args.device) != 'cpu':
        print(f"Current GPU memory usage: {torch.cuda.memory_allocated()/1024**2:.1f}MB")

    # 包装数据集以处理异常
    # Wrap dataset to handle exceptions
    print(f"\n====== Dataset Preparation ======")
    print(f"Original dataset size: {len(testset)}")
    testset = DatasetWrapper(testset)
    print(f"Filtered dataset size: {len(testset)}")

    # 自定义collate函数，处理None值
    # Custom collate function to handle None values
    def custom_collate_fn(batch):
        # 过滤掉None值
        # Filter out None values
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            raise ValueError("All samples in batch are invalid")
        return torch.utils.data.dataloader.default_collate(batch)

    # 如果数据集为空，则不继续执行
    # If dataset is empty, don't continue execution
    if len(testset) == 0:
        print("Error: No valid samples in dataset, cannot continue testing.")
        return

    # dataloader
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=1, shuffle=False, 
        num_workers=min(args.workers, 1),  # 减少worker数量，降低出错率 # Reduce worker count to lower error rate
        collate_fn=custom_collate_fn)
    
    print(f"\n====== Dataset Information ======")
    print(f"Test samples: {len(testset)}")
    print(f"Points per point cloud: {args.num_points}")
    print(f"Batch size: 1")

    # testing
    print(f"\n====== Starting Test ======")
    LOGGER.debug('tests, begin')

    # 创建结果目录
    # Create result directory
    os.makedirs(os.path.dirname(action.filename), exist_ok=True)
    
    print(f"\n====== Using Perturbation File Test Mode ======")
    success_count, total_count = action.eval_1(model, testloader, args.device)
    print(f"Perturbation file test evaluation completed, successfully processed {success_count}/{total_count} samples")
    LOGGER.debug('tests, end')


class Action:
    def __init__(self, args):
        self.filename = args.outfile
        # PointNet
        self.transfer_from = args.transfer_from
        self.dim_k = args.dim_k
        
        # 添加新的属性 (与train_pointlk.py保持一致)
        self.model_type = args.model_type
        self.num_attention_blocks = args.num_attention_blocks
        self.num_heads = args.num_heads
        
        # 添加Mamba3D属性
        self.num_mamba_blocks = args.num_mamba_blocks
        self.d_state = args.d_state
        self.expand = args.expand
        
        # 添加快速点注意力属性
        self.num_fast_attention_blocks = args.num_fast_attention_blocks
        self.fast_attention_scale = args.fast_attention_scale
        
        # 添加Cformer属性
        self.num_proxy_points = getattr(args, 'num_proxy_points', 8)
        self.num_blocks = getattr(args, 'num_blocks', 2)
        
        # 聚合函数设置 (与train_pointlk.py保持一致)
        self.sym_fn = None
        if args.model_type == 'attention':
            # 为attention模型设置聚合函数
            if args.symfn == 'max':
                self.sym_fn = attention_v1.symfn_max
            elif args.symfn == 'avg':
                self.sym_fn = attention_v1.symfn_avg
            else:
                self.sym_fn = attention_v1.symfn_attention_pool  # attention特有的聚合
        elif args.model_type == 'mamba3d':
            # 为Mamba3D模型设置聚合函数
            if args.symfn == 'max':
                self.sym_fn = mamba3d_v1.symfn_max
            elif args.symfn == 'avg':
                self.sym_fn = mamba3d_v1.symfn_avg
            elif args.symfn == 'selective':
                self.sym_fn = mamba3d_v1.symfn_selective
            else:
                self.sym_fn = mamba3d_v1.symfn_max  # 默认使用最大池化
        elif args.model_type == 'fast_attention':
            # 为快速点注意力模型设置聚合函数
            if args.symfn == 'max':
                self.sym_fn = fast_point_attention.symfn_max
            elif args.symfn == 'avg':
                self.sym_fn = fast_point_attention.symfn_avg
            elif args.symfn == 'selective':
                self.sym_fn = fast_point_attention.symfn_fast_attention_pool  # 快速注意力特有的聚合
            else:
                self.sym_fn = fast_point_attention.symfn_max  # 默认使用最大池化
        elif args.model_type == 'cformer':
            # 为Cformer模型设置聚合函数
            if args.symfn == 'max':
                self.sym_fn = cformer.symfn_max
            elif args.symfn == 'avg':
                self.sym_fn = cformer.symfn_avg
            elif args.symfn == 'cd_pool':
                self.sym_fn = cformer.symfn_cd_pool
            else:
                self.sym_fn = cformer.symfn_max  # 默认使用最大池化
        else:
            # 为pointnet模型设置聚合函数
            if args.symfn == 'max':
                self.sym_fn = ptlk.pointnet.symfn_max
            elif args.symfn == 'avg':
                self.sym_fn = ptlk.pointnet.symfn_avg
        
        # LK
        self.delta = args.delta
        self.max_iter = args.max_iter
        self.xtol = 1.0e-7
        self.p0_zero_mean = True
        self.p1_zero_mean = True
        # 保存num_points参数
        self.num_points = args.num_points
        
        # 新增 - 可视化参数
        self.visualize_pert = args.visualize_pert
        self.visualize_samples = args.visualize_samples
        
        # 当前扰动文件名（从参数中提取）- 修改这里
        if args.perturbations:
            self.current_pert_file = os.path.basename(args.perturbations)
        else:
            self.current_pert_file = "unknown"  # 默认值
        
        # 可视化文件夹
        if self.visualize_pert is not None and self.current_pert_file in self.visualize_pert:
            vis_dir = os.path.join(os.path.dirname(self.filename), 'visualize')
            os.makedirs(vis_dir, exist_ok=True)
            # 为当前扰动文件创建子文件夹
            self.vis_subdir = os.path.join(vis_dir, os.path.splitext(self.current_pert_file)[0])
            os.makedirs(self.vis_subdir, exist_ok=True)
            
            # 创建日志文件
            self.vis_log_file = os.path.join(self.vis_subdir, 'visualization_log.txt')
            with open(self.vis_log_file, 'w') as f:
                f.write("# PointNetLK 配准可视化日志\n")
                f.write("# 扰动文件: {}\n".format(self.current_pert_file))
                f.write("# 创建时间: {}\n\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
                f.write("点云对,扰动文件,预测值(w1,w2,w3,v1,v2,v3),真实扰动(w1,w2,w3,v1,v2,v3)\n")
                f.write("--------------------------------------------------------------------\n")
                
        # 可视化计数
        self.vis_count = 0

    def create_model(self):
        ptnet = self.create_pointnet_features()
        return self.create_from_pointnet_features(ptnet)

    def create_pointnet_features(self):
        if self.model_type == 'attention':
            # 创建attention模型
            ptnet = attention_v1.AttentionNet_features(
                dim_k=self.dim_k, 
                sym_fn=self.sym_fn,
                scale=1,
                num_attention_blocks=self.num_attention_blocks,
                num_heads=self.num_heads
            )
            # 支持从attention分类器加载预训练权重
            if self.transfer_from and os.path.isfile(self.transfer_from):
                try:
                    pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                    ptnet.load_state_dict(pretrained_dict)
                    print(f"成功加载attention预训练权重: {self.transfer_from}")
                except Exception as e:
                    print(f"加载attention预训练权重失败: {e}")
                    print("继续使用随机初始化权重")
        elif self.model_type == 'mamba3d':
            # 创建Mamba3D模型
            ptnet = mamba3d_v1.Mamba3D_features(
                dim_k=self.dim_k, 
                sym_fn=self.sym_fn,
                scale=1,
                num_mamba_blocks=self.num_mamba_blocks,
                d_state=self.d_state,
                expand=self.expand
            )
            # 支持从Mamba3D分类器加载预训练权重
            if self.transfer_from and os.path.isfile(self.transfer_from):
                try:
                    pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                    ptnet.load_state_dict(pretrained_dict)
                    print(f"成功加载Mamba3D预训练权重: {self.transfer_from}")
                except Exception as e:
                    print(f"加载Mamba3D预训练权重失败: {e}")
                    print("继续使用随机初始化权重")
        elif self.model_type == 'fast_attention':
            # 创建快速点注意力模型
            ptnet = fast_point_attention.FastPointAttention_features(
                dim_k=self.dim_k, 
                sym_fn=self.sym_fn,
                scale=self.fast_attention_scale,
                num_attention_blocks=self.num_fast_attention_blocks
            )
            # 支持从快速点注意力分类器加载预训练权重
            if self.transfer_from and os.path.isfile(self.transfer_from):
                try:
                    pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                    ptnet.load_state_dict(pretrained_dict)
                    print(f"成功加载快速点注意力预训练权重: {self.transfer_from}")
                except Exception as e:
                    print(f"加载快速点注意力预训练权重失败: {e}")
                    print("继续使用随机初始化权重")
        elif self.model_type == 'cformer':
            # 创建Cformer模型
            ptnet = cformer.CFormer_features(
                dim_k=self.dim_k, 
                sym_fn=self.sym_fn,
                scale=1,
                num_proxy_points=self.num_proxy_points,
                num_blocks=self.num_blocks
            )
            # 支持从Cformer分类器加载预训练权重
            if self.transfer_from and os.path.isfile(self.transfer_from):
                try:
                    pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                    ptnet.load_state_dict(pretrained_dict)
                    print(f"成功加载Cformer预训练权重: {self.transfer_from}")
                except Exception as e:
                    print(f"加载Cformer预训练权重失败: {e}")
                    print("继续使用随机初始化权重")
        else:
            # 创建原始pointnet模型
            ptnet = ptlk.pointnet.PointNet_features(self.dim_k, sym_fn=self.sym_fn)
            if self.transfer_from and os.path.isfile(self.transfer_from):
                try:
                    pretrained_dict = torch.load(self.transfer_from, map_location='cpu')
                    
                    # 检查是否是分类器权重（包含 features. 前缀）
                    # Check if it's classifier weights (contains features. prefix)
                    if any(key.startswith('features.') for key in pretrained_dict.keys()):
                        print(f"检测到分类器权重，提取特征提取器部分...")
                        # 从分类器权重中提取特征提取器权重
                        # Extract feature extractor weights from classifier weights
                        feature_dict = {}
                        for key, value in pretrained_dict.items():
                            if key.startswith('features.'):
                                # 移除 'features.' 前缀
                                # Remove 'features.' prefix
                                new_key = key[9:]  # 'features.' 有9个字符 # 'features.' has 9 characters
                                feature_dict[new_key] = value
                        
                        # 加载提取的特征权重
                        # Load extracted feature weights
                        ptnet.load_state_dict(feature_dict)
                        print(f"成功从分类器权重中提取并加载PointNet特征权重: {self.transfer_from}")
                    else:
                        # 直接加载特征提取器权重
                        # Directly load feature extractor weights
                        ptnet.load_state_dict(pretrained_dict)
                        print(f"成功加载PointNet预训练权重: {self.transfer_from}")
                        
                except Exception as e:
                    print(f"加载PointNet预训练权重失败: {e}")
                    print("继续使用随机初始化权重")
        
        return ptnet

    def create_from_pointnet_features(self, ptnet):
        return ptlk.pointlk.PointLK(ptnet, self.delta)

    def eval_1__header(self, fout):
        # 修改header，输出旋转误差和平移误差的列名
        # Modified header to output rotation error and translation error column names
        cols = ['sample_id', 'scene_name', 'sequence', 'rotation_error', 'translation_error', 'total_error']
        print(','.join(map(str, cols)), file=fout)
        fout.flush()

    def eval_1__write(self, fout, ig_gt, g_hat, sample_info=None):
        # 计算配准误差
        # Calculate registration error
        dg = g_hat.bmm(ig_gt) # if correct, dg == identity matrix.
        dx = ptlk.se3.log(dg) # --> [1, 6] (if correct, dx == zero vector)
        
        # 分别计算旋转误差和平移误差
        # Calculate rotation error and translation error separately
        rot_error = dx[:, :3]  # 旋转误差 [1, 3] # Rotation error [1, 3]
        trans_error = dx[:, 3:]  # 平移误差 [1, 3] # Translation error [1, 3]
        
        rot_norm = rot_error.norm(p=2, dim=1)  # 旋转误差L2范数 # Rotation error L2 norm
        trans_norm = trans_error.norm(p=2, dim=1)  # 平移误差L2范数 # Translation error L2 norm
        total_norm = dx.norm(p=2, dim=1)  # 总误差L2范数 # Total error L2 norm
        
        for i in range(g_hat.size(0)):
            # 获取样本信息
            # Get sample information
            if sample_info:
                sample_id = sample_info.get('identifier', f'sample_{i}')
                scene_name = sample_info.get('scene', 'unknown')
                sequence = sample_info.get('sequence', f'{i:04d}')
            else:
                sample_id = f'sample_{i}'
                scene_name = 'unknown'
                sequence = f'{i:04d}'
            
            # 输出误差信息
            # Output error information
            vals = [sample_id, scene_name, sequence, 
                   rot_norm[i].item(), trans_norm[i].item(), total_norm[i].item()]
            print(','.join(map(str, vals)), file=fout)
        fout.flush()

    def eval_1(self, model, testloader, device):
        model.eval()
        success_count = 0
        total_count = 0
        error_count = 0
        
        # 添加误差统计变量
        total_rot_error = 0.0
        total_trans_error = 0.0
        total_total_error = 0.0
        all_rot_errors = []
        all_trans_errors = []
        all_total_errors = []
        
        # 可视化样本计数
        current_vis_count = 0
        need_visualization = (self.visualize_pert is not None and 
                             self.current_pert_file in self.visualize_pert and
                             current_vis_count < self.visualize_samples)
        
        # 如果需要可视化，随机选择样本
        vis_indices = []
        if need_visualization:
            # 随机选择样本索引
            num_samples = min(len(testloader), 100)  # 限制采样范围
            vis_indices = random.sample(range(num_samples), min(num_samples, self.visualize_samples))
            print(f"\n====== Visualization Settings ======")
            print(f"Will visualize the following sample indices for perturbation file {self.current_pert_file}: {vis_indices}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        
        with open(self.filename, 'w') as fout:
            # 写入文件头部信息
            # Write file header information
            print(f"# PointNetLK Registration Test Results", file=fout)
            print(f"# Perturbation file: {self.current_pert_file}", file=fout)
            print(f"# Test time: {time.strftime('%Y-%m-%d %H:%M:%S')}", file=fout)
            print(f"# Rotation error unit: radians", file=fout)
            print(f"# Translation error unit: meters", file=fout)
            print(f"# =====================================", file=fout)
            
            self.eval_1__header(fout)
            with torch.no_grad():
                # 计算总共需要测试的样本数
                # Calculate total number of samples to test
                total_samples = len(testloader)
                start_time = time.time()
                
                for i, data in enumerate(testloader):
                    batch_start_time = time.time()
                    total_count += 1
                    
                    # 检查是否需要可视化这个样本
                    do_visualize = need_visualization and i in vis_indices
                    
                    try:
                        # 定期清理GPU内存，特别是在长时间运行后
                        # Periodic GPU memory cleanup, especially after long runs
                        if i > 0 and i % 100 == 0:
                            torch.cuda.empty_cache()
                            print(f"GPU memory cleanup at sample {i}")
                        
                        # 检查CUDA状态
                        # Check CUDA status
                        if i > 0 and i % 500 == 0:
                            if torch.cuda.is_available():
                                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                                memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                                print(f"Sample {i}: GPU memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
                            else:
                                print(f"Warning: CUDA not available at sample {i}")
                        
                        p0, p1, igt = data
                        
                        # 检查点云形状和有效性
                        if p0.shape[1] != self.num_points or p1.shape[1] != self.num_points:
                            print(f"警告: 批次 {i} 点云形状不符合预期: p0={p0.shape}, p1={p1.shape}, 期望点数={self.num_points}")
                        
                        # 检查点云有效性
                        if not torch.isfinite(p0).all() or not torch.isfinite(p1).all():
                            print(f"警告: 批次 {i} 包含NaN值，尝试修复")
                            p0 = torch.nan_to_num(p0, nan=0.0, posinf=1.0, neginf=-1.0)
                            p1 = torch.nan_to_num(p1, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                        # 检查扰动矩阵有效性
                        if not torch.isfinite(igt).all():
                            print(f"警告: 批次 {i} 扰动矩阵包含无效值，跳过此样本")
                            error_count += 1
                            dummy_vals = ['nan'] * 6
                            print(','.join(dummy_vals), file=fout)
                            fout.flush()
                            continue
                        
                        # 获取场景信息和源文件路径（用于命名可视化文件）
                        scene_name = "unknown"
                        source_seq = "0000"
                        source_file_path = None
                        target_file_path = None
                        
                        # 获取样本信息
                        sample_info = {}
                        if hasattr(testloader.dataset, 'get_cloud_info'):
                            cloud_info = testloader.dataset.get_cloud_info(i)
                            if cloud_info:
                                sample_info = cloud_info
                                scene_name = cloud_info.get('scene', scene_name)
                                source_seq = cloud_info.get('sequence', source_seq)
                                source_file_path = cloud_info.get('source_file')
                                target_file_path = cloud_info.get('target_file')
                        
                        # 如果没有获取到信息，尝试其他方法
                        if not sample_info:
                            try:
                                # 尝试从数据集中获取源文件和目标文件路径
                                if hasattr(testloader.dataset, 'dataset') and hasattr(testloader.dataset.dataset, 'pairs'):
                                    idx = testloader.dataset.valid_indices[i] if hasattr(testloader.dataset, 'valid_indices') else i
                                    if idx < len(testloader.dataset.dataset.pairs):
                                        source_file_path, target_file_path = testloader.dataset.dataset.pairs[idx]
                                elif hasattr(testloader.dataset, 'dataset') and hasattr(testloader.dataset.dataset, 'dataset'):
                                    if hasattr(testloader.dataset.dataset.dataset, 'pairs'):
                                        idx = testloader.dataset.valid_indices[i] if hasattr(testloader.dataset, 'valid_indices') else i
                                        if idx < len(testloader.dataset.dataset.dataset.pairs):
                                            source_file_path, target_file_path = testloader.dataset.dataset.dataset.pairs[idx]
                            except Exception as e:
                                pass
                            
                            # 提取场景名称和序号
                            if source_file_path:
                                try:
                                    norm_path = source_file_path.replace('\\', '/')
                                    if 'C3VD_ply_source' in norm_path:
                                        parts = norm_path.split('/')
                                        idx = [i for i, part in enumerate(parts) if part == 'C3VD_ply_source']
                                        if idx and idx[0] + 1 < len(parts):
                                            scene_name = parts[idx[0] + 1]
                                    
                                    basename = os.path.basename(source_file_path)
                                    if basename.endswith("_depth_pcd.ply") and basename[:4].isdigit():
                                        source_seq = basename[:4]
                                    else:
                                        import re
                                        numbers = re.findall(r'\d+', basename)
                                        if numbers:
                                            source_seq = numbers[0].zfill(4)
                                except Exception as e:
                                    pass
                            
                            # 构建样本信息
                            sample_info = {
                                'identifier': f"{scene_name}_{source_seq}",
                                'scene': scene_name,
                                'sequence': source_seq,
                                'source_file': source_file_path,
                                'target_file': target_file_path
                            }
                        
                        # 执行配准，加入额外的错误检查
                        # Perform registration with additional error checking
                        try:
                            res = self.do_estimate(p0, p1, model, device) # --> [1, 4, 4]
                            
                            # 检查配准结果有效性
                            if not torch.isfinite(res).all():
                                print(f"警告: 批次 {i} 配准结果包含无效值")
                                error_count += 1
                                dummy_vals = ['nan'] * 6
                                print(','.join(dummy_vals), file=fout)
                                fout.flush()
                                continue
                                
                        except Exception as registration_error:
                            print(f"配准错误: 批次 {i} 配准失败: {str(registration_error)}")
                            error_count += 1
                            dummy_vals = ['nan'] * 6
                            print(','.join(dummy_vals), file=fout)
                            fout.flush()
                            continue
                        
                        ig_gt = igt.cpu().contiguous().view(-1, 4, 4) # --> [1, 4, 4]
                        g_hat = res.cpu().contiguous().view(-1, 4, 4) # --> [1, 4, 4]

                        # 计算配准误差
                        try:
                            dg = g_hat.bmm(ig_gt) # if correct, dg == identity matrix.
                            dx = ptlk.se3.log(dg) # --> [1, 6] (if correct, dx == zero vector)
                            
                            # 检查误差向量有效性
                            if not torch.isfinite(dx).all():
                                print(f"警告: 批次 {i} 误差计算包含无效值")
                                error_count += 1
                                dummy_vals = ['nan'] * 6
                                print(','.join(dummy_vals), file=fout)
                                fout.flush()
                                continue
                            
                            # 分别计算旋转误差和平移误差
                            rot_error = dx[:, :3]  # 旋转误差 [1, 3]
                            trans_error = dx[:, 3:]  # 平移误差 [1, 3]
                            
                            rot_norm = rot_error.norm(p=2, dim=1)  # 旋转误差L2范数
                            trans_norm = trans_error.norm(p=2, dim=1)  # 平移误差L2范数
                            total_norm = dx.norm(p=2, dim=1)  # 总误差L2范数
                            
                            # 检查计算结果有效性
                            if not (torch.isfinite(rot_norm).all() and torch.isfinite(trans_norm).all() and torch.isfinite(total_norm).all()):
                                print(f"警告: 批次 {i} 误差范数计算包含无效值")
                                error_count += 1
                                dummy_vals = ['nan'] * 6
                                print(','.join(dummy_vals), file=fout)
                                fout.flush()
                                continue
                            
                        except Exception as error_calc_error:
                            print(f"误差计算错误: 批次 {i} 误差计算失败: {str(error_calc_error)}")
                            error_count += 1
                            dummy_vals = ['nan'] * 6
                            print(','.join(dummy_vals), file=fout)
                            fout.flush()
                            continue
                        
                        # 累加误差统计
                        total_rot_error += rot_norm.item()
                        total_trans_error += trans_norm.item()
                        total_total_error += total_norm.item()
                        all_rot_errors.append(rot_norm.item())
                        all_trans_errors.append(trans_norm.item())
                        all_total_errors.append(total_norm.item())
                        
                        # 写入结果
                        self.eval_1__write(fout, ig_gt, g_hat, sample_info)
                        success_count += 1
                        
                        # 可视化处理
                        if do_visualize and current_vis_count < self.visualize_samples:
                            try:
                                print(f"\n====== 开始处理可视化样本 {i} ======")
                                
                                # 获取预测值和扰动真实值
                                x_hat = ptlk.se3.log(g_hat)[0]  # [6]
                                mx_gt = ptlk.se3.log(ig_gt)[0]  # [6]
                                
                                # 获取点云信息 - 使用我们修改后的测试数据集
                                cloud_info = {}
                                identifier = f"unknown_{i:04d}"
                                scene_name = "unknown"
                                source_seq = f"{i:04d}"
                                source_file_path = None
                                target_file_path = None
                                
                                if hasattr(testloader.dataset, 'get_cloud_info'):
                                    # 使用我们新添加的方法获取点云信息
                                    cloud_info = testloader.dataset.get_cloud_info(i)
                                    if cloud_info:
                                        identifier = cloud_info.get('identifier', identifier)
                                        scene_name = cloud_info.get('scene', scene_name)
                                        source_seq = cloud_info.get('sequence', source_seq)
                                        source_file_path = cloud_info.get('source_file')
                                        target_file_path = cloud_info.get('target_file')
                                        
                                        print(f"从测试数据集获取到点云信息:")
                                        print(f"  - 标识符: {identifier}")
                                        print(f"  - 场景: {scene_name}")
                                        print(f"  - 序列号: {source_seq}")
                                        print(f"  - 源文件: {source_file_path}")
                                        print(f"  - 目标文件: {target_file_path}")
                                else:
                                    # 回退到旧方法
                                    try:
                                        # 首先检查完整数据集结构
                                        if hasattr(testloader.dataset, 'dataset') and hasattr(testloader.dataset.dataset, 'pairs'):
                                            idx = testloader.dataset.valid_indices[i] if hasattr(testloader.dataset, 'valid_indices') else i
                                            if idx < len(testloader.dataset.dataset.pairs):
                                                source_file_path, target_file_path = testloader.dataset.dataset.pairs[idx]
                                                print(f"获取到源文件: {source_file_path}")
                                                print(f"获取到目标文件: {target_file_path}")
                                        # 然后检查子集数据集结构
                                        elif hasattr(testloader.dataset, 'dataset') and hasattr(testloader.dataset.dataset, 'dataset'):
                                            if hasattr(testloader.dataset.dataset.dataset, 'pairs'):
                                                idx = testloader.dataset.valid_indices[i] if hasattr(testloader.dataset, 'valid_indices') else i
                                                if idx < len(testloader.dataset.dataset.dataset.pairs):
                                                    source_file_path, target_file_path = testloader.dataset.dataset.dataset.pairs[idx]
                                                    print(f"从子集获取到源文件: {source_file_path}")
                                                    print(f"从子集获取到目标文件: {target_file_path}")
                                    except Exception as e:
                                        print(f"获取文件路径时出错: {e}")
                                        traceback.print_exc()
                                    
                                    # 提取场景名称和序号
                                    if source_file_path:
                                        try:
                                            # 标准化路径分隔符
                                            norm_path = source_file_path.replace('\\', '/')
                                            
                                            # 方法1: 从路径结构提取场景名称
                                            if 'C3VD_ply_source' in norm_path:
                                                # 查找C3VD_ply_source之后的第一个目录
                                                parts = norm_path.split('/')
                                                idx = [i for i, part in enumerate(parts) if part == 'C3VD_ply_source']
                                                if idx and idx[0] + 1 < len(parts):
                                                    scene_name = parts[idx[0] + 1]
                                                    print(f"成功从路径提取场景名称: {scene_name}")
                                            
                                            # 提取源序号
                                            basename = os.path.basename(source_file_path)
                                            
                                            # 假设源文件名格式为 "XXXX_depth_pcd.ply"
                                            if basename.endswith("_depth_pcd.ply") and basename[:4].isdigit():
                                                source_seq = basename[:4]
                                                print(f"从文件名提取序号: {source_seq}")
                                            else:
                                                # 尝试从文件名中提取数字序列
                                                import re
                                                numbers = re.findall(r'\d+', basename)
                                                if numbers:
                                                    source_seq = numbers[0].zfill(4)
                                                    print(f"从文件名提取数字序列: {source_seq}")
                                        except Exception as e:
                                            print(f"提取场景名称时出错: {e}")
                                            traceback.print_exc()
                                
                                # 设置目标文件名 - 使用新的命名格式
                                source_filename = os.path.join(self.vis_subdir, f"{scene_name}_{source_seq}_source.ply")
                                target_filename = os.path.join(self.vis_subdir, f"{scene_name}_{source_seq}_target.ply")
                                
                                # 新增：原始源点云和目标点云的文件名
                                original_source_filename = os.path.join(self.vis_subdir, f"{scene_name}_{source_seq}_original_source.ply")
                                original_target_filename = os.path.join(self.vis_subdir, f"{scene_name}_{source_seq}_original_target.ply")
                                
                                # 记录配准信息到日志文件
                                try:
                                    with open(self.vis_log_file, 'a') as f:
                                        # 记录基本信息
                                        predicted = ",".join([f"{val:.6f}" for val in x_hat.tolist()])
                                        gt_values = ",".join([f"{val:.6f}" for val in (-mx_gt).tolist()])
                                        
                                        # 记录文件信息和扰动详情
                                        f.write(f"{identifier},{self.current_pert_file},{predicted},{gt_values}\n")
                                        
                                        # 记录点云文件路径和配准误差
                                        f.write(f"源文件: {source_file_path}\n")
                                        f.write(f"目标文件: {target_file_path}\n")
                                        f.write(f"源点云尺寸: {p1.shape}, 目标点云尺寸: {p0.shape}\n")
                                        f.write(f"配准误差: 旋转={rot_norm.item():.6f}, 平移={trans_norm.item():.6f}, 总误差={total_norm.item():.6f}\n\n")
                                    
                                    print(f"已将配准信息记录到日志文件: {self.vis_log_file}")
                                except Exception as e:
                                    print(f"写入日志文件时出错: {e}")
                                    traceback.print_exc()
                                
                                # 复制原始PLY文件到可视化目录
                                success_copy = True
                                if source_file_path and os.path.exists(source_file_path):
                                    try:
                                        print(f"复制源点云: {source_file_path} -> {source_filename}")
                                        shutil.copy2(source_file_path, source_filename)
                                    except Exception as e:
                                        print(f"复制源点云文件失败: {e}")
                                        traceback.print_exc()
                                        success_copy = False
                                else:
                                    print(f"警告: 无法找到源点云文件路径: {source_file_path}")
                                    success_copy = False
                                
                                if target_file_path and os.path.exists(target_file_path):
                                    try:
                                        print(f"复制目标点云: {target_file_path} -> {target_filename}")
                                        shutil.copy2(target_file_path, target_filename)
                                    except Exception as e:
                                        print(f"复制目标点云文件失败: {e}")
                                        traceback.print_exc()
                                        success_copy = False
                                else:
                                    print(f"警告: 无法找到目标点云文件路径: {target_file_path}")
                                    success_copy = False
                                
                                # 保存原始点云数据（不经过变换的）
                                try:
                                    # 尝试从测试数据集获取原始点云
                                    original_source, original_target = None, None
                                    
                                    if hasattr(testloader.dataset, 'get_original_clouds'):
                                        original_source, original_target = testloader.dataset.get_original_clouds(i)
                                        if original_source is not None and original_target is not None:
                                            print(f"成功获取到原始点云数据")
                                    
                                    # 创建PLY点云辅助函数
                                    def create_ply(points, filename):
                                        if points is None:
                                            print(f"警告: 无法保存为空的点云到 {filename}")
                                            return False
                                            
                                        # 确保点云是NumPy数组
                                        if isinstance(points, torch.Tensor):
                                            points_np = points.cpu().numpy()
                                            if len(points_np.shape) > 2:  # 如果有批次维度
                                                points_np = points_np.squeeze(0)
                                        else:
                                            points_np = points
                                        
                                        # 创建顶点属性
                                        vertex = np.zeros(points_np.shape[0], dtype=[
                                            ('x', 'f4'), ('y', 'f4'), ('z', 'f4')
                                        ])
                                        vertex['x'] = points_np[:, 0]
                                        vertex['y'] = points_np[:, 1]
                                        vertex['z'] = points_np[:, 2]
                                        
                                        # 创建PLY元素
                                        el = PlyElement.describe(vertex, 'vertex')
                                        
                                        # 保存PLY文件
                                        PlyData([el], text=True).write(filename)
                                        print(f"已保存点云到: {filename}")
                                        return True
                                    
                                    # 保存原始源点云和目标点云
                                    if original_source is not None:
                                        create_ply(original_source, original_source_filename)
                                    if original_target is not None:
                                        create_ply(original_target, original_target_filename)
                                    
                                    # 更新日志
                                    with open(self.vis_log_file, 'a') as f:
                                        if original_source is not None and original_target is not None:
                                            f.write(f"已保存原始点云数据到: {os.path.basename(original_source_filename)} 和 {os.path.basename(original_target_filename)}\n\n")
                                        else:
                                            f.write(f"无法获取原始点云数据\n\n")
                                
                                except Exception as e:
                                    print(f"保存原始点云时出错: {e}")
                                    traceback.print_exc()
                                
                                # 如果文件复制失败但点云数据有效，则直接保存点云
                                if not success_copy:
                                    print("尝试直接保存点云数据...")
                                    try:
                                        # 将PyTorch张量转换为NumPy数组
                                        p0_np = p0.cpu().numpy().squeeze(0)  # 目标点云
                                        p1_np = p1.cpu().numpy().squeeze(0)  # 源点云
                                        
                                        # 创建PLY点云
                                        def create_ply(points, filename):
                                            # 创建顶点属性
                                            vertex = np.zeros(points.shape[0], dtype=[
                                                ('x', 'f4'), ('y', 'f4'), ('z', 'f4')
                                            ])
                                            vertex['x'] = points[:, 0]
                                            vertex['y'] = points[:, 1]
                                            vertex['z'] = points[:, 2]
                                            
                                            # 创建PLY元素
                                            el = PlyElement.describe(vertex, 'vertex')
                                            
                                            # 保存PLY文件
                                            PlyData([el], text=True).write(filename)
                                            print(f"已直接保存点云到: {filename}")
                                        
                                        # 保存源点云和目标点云
                                        create_ply(p1_np, source_filename)
                                        create_ply(p0_np, target_filename)
                                        
                                        # 更新日志
                                        with open(self.vis_log_file, 'a') as f:
                                            f.write(f"注意: 由于无法复制原始文件，已直接保存点云数据\n\n")
                                        
                                        success_copy = True
                                    except Exception as e:
                                        print(f"直接保存点云失败: {e}")
                                        traceback.print_exc()
                                
                                # 无论文件复制是否成功，都计数
                                current_vis_count += 1
                                print(f"已完成样本 {identifier} 的可视化 ({current_vis_count}/{self.visualize_samples})")
                                
                                # 获取完整的4x4变换矩阵
                                g_matrix = g_hat[0].cpu().numpy()  # 预测变换矩阵
                                gt_matrix = ig_gt[0].cpu().numpy()  # 真实变换矩阵

                                # 记录矩阵形式到日志文件
                                with open(self.vis_log_file, 'a') as f:
                                    f.write("\n预测变换矩阵 (4x4):\n")
                                    for row in g_matrix:
                                        f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}\n")
                                    
                                    f.write("\n真实变换矩阵 (4x4):\n")
                                    for row in gt_matrix:
                                        f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}\n")
                                
                            except Exception as e:
                                print(f"处理可视化样本时出错: {e}")
                                traceback.print_exc()
                                # 即使出错也尝试写入错误信息到日志
                                try:
                                    with open(self.vis_log_file, 'a') as f:
                                        f.write(f"错误处理样本_{i},{self.current_pert_file},错误信息: {str(e)}\n")
                                        f.write(f"错误堆栈: {traceback.format_exc()}\n\n")
                                except Exception as log_err:
                                    print(f"尝试记录错误到日志文件时也失败了: {log_err}")
                                
                                # 即使出错也增加计数
                                current_vis_count += 1
                        
                        # 计算进度和时间
                        batch_time = time.time() - batch_start_time
                        elapsed_time = time.time() - start_time
                        estimated_total = elapsed_time / (i + 1) * total_samples
                        remaining_time = max(0, estimated_total - elapsed_time)
                        
                        # 根据误差水平添加标记
                        error_level = ""
                        if total_norm.item() > 0.5:
                            error_level = "【配准失败】"
                            error_count += 1
                        elif total_norm.item() > 0.1:
                            error_level = "【误差较大】"
                        
                        # 打印进度
                        print(f"测试: [{i+1}/{total_samples}] {(i+1)/total_samples*100:.1f}% | "
                              f"旋转误差: {rot_norm.item():.6f}, 平移误差: {trans_norm.item():.6f} | "
                              f"总误差: {total_norm.item():.6f} {error_level} | "
                              f"耗时: {batch_time:.2f}秒 | "
                              f"剩余: {remaining_time/60:.1f}分钟")
                        
                        LOGGER.info('test, %d/%d, rot_error: %f, trans_error: %f, total_error: %f', 
                                   i, total_samples, rot_norm.item(), trans_norm.item(), total_norm.item())
                        
                    except Exception as e:
                        print(f"错误: 处理批次 {i} 时出错: {str(e)}")
                        # 检查是否是CUDA错误
                        error_str = str(e).lower()
                        if 'cuda' in error_str or 'cublas' in error_str or 'cudnn' in error_str:
                            print(f"检测到CUDA错误，尝试清理GPU内存...")
                            torch.cuda.empty_cache()
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                print(f"GPU内存清理完成")
                        
                        # 记录错误详情到日志
                        LOGGER.error('Error in batch %d: %s', i, str(e), exc_info=True)
                        error_count += 1
                        # 写入无效结果
                        dummy_vals = ['nan'] * 6
                        print(','.join(dummy_vals), file=fout)
                        fout.flush()
        
        # 结果统计
        # Result statistics
        total_time = time.time() - start_time
        
        # 添加统计信息到输出文件
        # Add statistical information to output file
        with open(self.filename, 'a') as fout:
            print(f"", file=fout)
            print(f"# =====================================", file=fout)
            print(f"# Test Statistics", file=fout)
            print(f"# =====================================", file=fout)
            print(f"# Total time: {total_time:.2f} seconds", file=fout)
            print(f"# Successfully processed samples: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)", file=fout)
            
            # 计算统计指标
            # Calculate statistical metrics
            if success_count > 0:
                avg_rot_error = total_rot_error / success_count
                avg_trans_error = total_trans_error / success_count
                avg_total_error = total_total_error / success_count
                
                # 计算中位数误差
                # Calculate median error
                all_rot_errors.sort()
                all_trans_errors.sort()
                all_total_errors.sort()
                
                median_rot_error = all_rot_errors[len(all_rot_errors)//2] if all_rot_errors else 0
                median_trans_error = all_trans_errors[len(all_trans_errors)//2] if all_trans_errors else 0
                median_total_error = all_total_errors[len(all_total_errors)//2] if all_total_errors else 0
                
                # 计算标准差
                # Calculate standard deviation
                if len(all_rot_errors) > 1:
                    std_rot_error = math.sqrt(sum((x - avg_rot_error)**2 for x in all_rot_errors) / (len(all_rot_errors) - 1))
                    std_trans_error = math.sqrt(sum((x - avg_trans_error)**2 for x in all_trans_errors) / (len(all_trans_errors) - 1))
                    std_total_error = math.sqrt(sum((x - avg_total_error)**2 for x in all_total_errors) / (len(all_total_errors) - 1))
                else:
                    std_rot_error = 0
                    std_trans_error = 0
                    std_total_error = 0
                
                print(f"#", file=fout)
                print(f"# Error Statistics (radians/meters):", file=fout)
                print(f"# Average rotation error: {avg_rot_error:.6f}", file=fout)
                print(f"# Average translation error: {avg_trans_error:.6f}", file=fout)
                print(f"# Average total error: {avg_total_error:.6f}", file=fout)
                print(f"#", file=fout)
                print(f"# Median errors:", file=fout)
                print(f"# Median rotation error: {median_rot_error:.6f}", file=fout)
                print(f"# Median translation error: {median_trans_error:.6f}", file=fout)
                print(f"# Median total error: {median_total_error:.6f}", file=fout)
                print(f"#", file=fout)
                print(f"# Standard deviation:", file=fout)
                print(f"# Rotation error std: {std_rot_error:.6f}", file=fout)
                print(f"# Translation error std: {std_trans_error:.6f}", file=fout)
                print(f"# Total error std: {std_total_error:.6f}", file=fout)
                print(f"#", file=fout)
                print(f"# Min/Max errors:", file=fout)
                print(f"# Min rotation error: {min(all_rot_errors):.6f}", file=fout)
                print(f"# Max rotation error: {max(all_rot_errors):.6f}", file=fout)
                print(f"# Min translation error: {min(all_trans_errors):.6f}", file=fout)
                print(f"# Max translation error: {max(all_trans_errors):.6f}", file=fout)
                print(f"# Min total error: {min(all_total_errors):.6f}", file=fout)
                print(f"# Max total error: {max(all_total_errors):.6f}", file=fout)
            else:
                print(f"# No successfully processed samples, cannot calculate error statistics", file=fout)
            
            if error_count > 0:
                print(f"# Registration failed samples: {error_count}/{success_count} ({error_count/success_count*100:.1f}%)", file=fout)
            
            print(f"# Completion time: {time.strftime('%Y-%m-%d %H:%M:%S')}", file=fout)
            print(f"# =====================================", file=fout)
        
        print(f"\n====== Test Completed ======")
        print(f"Total time: {total_time:.2f} seconds (average {total_time/total_count:.2f} seconds per sample)")
        print(f"Successfully processed samples: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        # 计算平均误差
        # Calculate average errors
        if success_count > 0:
            avg_rot_error = total_rot_error / success_count
            avg_trans_error = total_trans_error / success_count
            print(f"Average rotation error: {avg_rot_error:.6f}")
            print(f"Average translation error: {avg_trans_error:.6f}")
            
        if error_count > 0:
            print(f"Registration failed samples: {error_count}/{success_count} ({error_count/success_count*100:.1f}%)")
        print(f"Results saved to: {self.filename}")
        
        # 可视化完成信息
        # Visualization completion information
        if need_visualization:
            print(f"\n====== Visualization Completed ======")
            print(f"Visualized {current_vis_count} samples for perturbation file {self.current_pert_file}")
            print(f"Visualization results saved to: {self.vis_subdir}")
            print(f"Visualization log: {self.vis_log_file}")
        
        return success_count, total_count

    def do_estimate(self, p0, p1, model, device):
        p0 = p0.to(device) # template (1, N, 3)
        p1 = p1.to(device) # source (1, M, 3)
        r = ptlk.pointlk.PointLK.do_forward(model, p0, p1, self.max_iter, self.xtol,\
                                            self.p0_zero_mean, self.p1_zero_mean)
        #r = model(p0, p1, self.max_iter)
        est_g = model.g # (1, 4, 4)

        return est_g

    def test_metrics(self, rotations_gt, translation_gt, rotations_ab, translation_ab, filename):
        rotations_gt = np.concatenate(rotations_gt, axis=0).reshape(-1, 3)
        translation_gt = np.concatenate(translation_gt, axis=0).reshape(-1, 3)
        rotations_ab = np.concatenate(rotations_ab, axis=0).reshape(-1, 3)
        translation_ab = np.concatenate(translation_ab, axis=0).reshape(-1,3)

        # root square error
        rot_err = np.sqrt(np.mean((np.degrees(rotations_ab) - np.degrees(rotations_gt)) ** 2, axis=1))
        trans_err = np.sqrt(np.mean((translation_ab - translation_gt) ** 2, axis=1))

        suc_tab = np.zeros(11)
        
        # set the criteria
        rot_err_tab = np.arange(11) * 0.5
        trans_err_tab = np.arange(11) * 0.05
        
        err_count_tab = np.triu(np.ones((11, 11)))
        
        for i in range(rot_err.shape[0]):
            if rot_err[i] <= rot_err_tab[0] and trans_err[i] <= trans_err_tab[0]:
                suc_tab = suc_tab + err_count_tab[0]
            elif rot_err[i] <= rot_err_tab[1] and trans_err[i] <= trans_err_tab[1]:
                suc_tab = suc_tab + err_count_tab[1]
            elif rot_err[i] <= rot_err_tab[2] and trans_err[i] <= trans_err_tab[2]:
                suc_tab = suc_tab + err_count_tab[2]
            elif rot_err[i] <= rot_err_tab[3] and trans_err[i] <= trans_err_tab[3]:
                suc_tab = suc_tab + err_count_tab[3]
            elif rot_err[i] <= rot_err_tab[4] and trans_err[i] <= trans_err_tab[4]:
                suc_tab = suc_tab + err_count_tab[4]
            elif rot_err[i] <= rot_err_tab[5] and trans_err[i] <= trans_err_tab[5]:
                suc_tab = suc_tab + err_count_tab[5]
            elif rot_err[i] <= rot_err_tab[6] and trans_err[i] <= trans_err_tab[6]:
                suc_tab = suc_tab + err_count_tab[6]
            elif rot_err[i] <= rot_err_tab[7] and trans_err[i] <= trans_err_tab[7]:
                suc_tab = suc_tab + err_count_tab[7]
            elif rot_err[i] <= rot_err_tab[8] and trans_err[i] <= trans_err_tab[8]:
                suc_tab = suc_tab + err_count_tab[8]
            elif rot_err[i] <= rot_err_tab[9] and trans_err[i] <= trans_err_tab[9]:
                suc_tab = suc_tab + err_count_tab[9]
            elif rot_err[i] <= rot_err_tab[10] and trans_err[i] <= trans_err_tab[10]:
                suc_tab = suc_tab + err_count_tab[10]

        # 1. use mean error
        rot_mse_ab = np.mean((np.degrees(rotations_ab) - np.degrees(rotations_gt)) ** 2)
        rot_rmse_ab = np.sqrt(rot_mse_ab)
        rot_mae_ab = np.mean(np.abs(np.degrees(rotations_ab) - np.degrees(rotations_gt)))

        trans_mse_ab = np.mean((translation_ab - translation_gt) ** 2)
        trans_rmse_ab = np.sqrt(trans_mse_ab)
        trans_mae_ab = np.mean(np.abs(translation_ab - translation_gt))
        
        # 2. use median error
        rot_mse_ab_02 = np.median((np.degrees(rotations_ab) - np.degrees(rotations_gt)) ** 2)
        rot_rmse_ab_02 = np.sqrt(rot_mse_ab_02)
        rot_mae_ab_02 = np.median(np.abs(np.degrees(rotations_ab) - np.degrees(rotations_gt)))
        
        trans_mse_ab_02 = np.median((translation_ab - translation_gt) ** 2)
        trans_rmse_ab_02 = np.sqrt(trans_mse_ab_02)
        trans_mae_ab_02 = np.median(np.abs(translation_ab - translation_gt))

        # 将结果写入日志文件而不是打印到控制台
        log_message = f'Source to Template:\n{filename}\n'
        log_message += '********************mean********************\n'
        log_message += f'rot_MSE: {rot_mse_ab}, rot_RMSE: {rot_rmse_ab}, rot_MAE: {rot_mae_ab}, trans_MSE: {trans_mse_ab}, trans_RMSE: {trans_rmse_ab}, trans_MAE: {trans_mae_ab}, rot_err: {np.mean(rot_err)}, trans_err: {np.mean(trans_err)}\n'
        log_message += '********************median********************\n'
        log_message += f'rot_MSE: {rot_mse_ab_02}, rot_RMSE: {rot_rmse_ab_02}, rot_MAE: {rot_mae_ab_02}, trans_MSE: {trans_mse_ab_02}, trans_RMSE: {trans_rmse_ab_02}, trans_MAE: {trans_mae_ab_02}\n'
        log_message += f'success cases are {suc_tab}\n'
        
        # 同时写入日志文件和打印到控制台
        LOGGER.info(log_message)
        
        # 将结果写入文件
        metrics_filename = f"{os.path.splitext(filename)[0]}_metrics.txt"
        with open(metrics_filename, 'w') as f:
            f.write(log_message)
        
        print(f"测试指标已保存到: {metrics_filename}")
        
        return


    def update_perturbation(self, perturbation_file, outfile):
        """更新当前处理的扰动文件和输出文件"""
        self.filename = outfile
        self.current_pert_file = os.path.basename(perturbation_file)
        
        # 更新可视化相关的设置
        if self.visualize_pert is not None and self.current_pert_file in self.visualize_pert:
            vis_dir = os.path.join(os.path.dirname(self.filename), 'visualize')
            os.makedirs(vis_dir, exist_ok=True)
            
            # 为当前扰动文件创建子文件夹
            self.vis_subdir = os.path.join(vis_dir, os.path.splitext(self.current_pert_file)[0])
            os.makedirs(self.vis_subdir, exist_ok=True)
            
            # 创建日志文件
            self.vis_log_file = os.path.join(self.vis_subdir, 'visualization_log.txt')
            with open(self.vis_log_file, 'w') as f:
                f.write("# PointNetLK 配准可视化日志\n")
                f.write("# 扰动文件: {}\n".format(self.current_pert_file))
                f.write("# 创建时间: {}\n\n".format(time.strftime("%Y-%m-%d %H:%M:%S")))
                f.write("点云对,扰动文件,预测值(w1,w2,w3,v1,v2,v3),真实扰动(w1,w2,w3,v1,v2,v3)\n")
                f.write("--------------------------------------------------------------------\n")


def get_datasets(args):
    cinfo = None
    if args.categoryfile and os.path.exists(args.categoryfile):
        try:
            categories = [line.rstrip('\n') for line in open(args.categoryfile)]
            categories.sort()
            c_to_idx = {categories[i]: i for i in range(len(categories))}
            cinfo = (categories, c_to_idx)
        except Exception as e:
            LOGGER.warning(f"Failed to load category file: {e}")
            # 如果是C3VD数据集，即使没有类别文件也可以继续
            if args.dataset_type != 'c3vd':
                raise

    perturbations = None
    fmt_trans = False
    # 检测是否为gt_poses.csv文件以启用随机选择模式
    # Detect if it's gt_poses.csv file to enable random sampling mode
    is_gt_poses_mode = False
    if args.perturbations:
        if not os.path.exists(args.perturbations):
            raise FileNotFoundError(f"{args.perturbations} not found.")
        perturbations = numpy.loadtxt(args.perturbations, delimiter=',')
        
        # 硬编码检测gt_poses.csv文件
        # Hard-coded detection of gt_poses.csv file
        perturbation_filename = os.path.basename(args.perturbations)
        if perturbation_filename == 'gt_poses.csv' or 'gt_poses' in perturbation_filename:
            is_gt_poses_mode = True
            print(f"\n🎯 GT_POSES模式已激活！")
            print(f"扰动文件: {args.perturbations}")
            print(f"扰动数量: {len(perturbations)}")
            print(f"测试模式: 每个扰动随机选择一个测试样本")
            print(f"总测试次数: {len(perturbations)} (等于扰动数量)")
        else:
            print(f"\n📋 标准测试模式")
            print(f"扰动文件: {args.perturbations}")
            print(f"扰动数量: {len(perturbations)}")
            print(f"测试模式: 遍历所有测试样本，每个样本使用一个扰动")
            
    if args.format == 'wt':
        fmt_trans = True

    if args.dataset_type == 'modelnet':
        transform = torchvision.transforms.Compose([
            ptlk.data.transforms.Mesh2Points(),
            ptlk.data.transforms.OnUnitCube(),
            ptlk.data.transforms.Resampler(args.num_points)  # 使用args中的参数
        ])

        testdata = ptlk.data.datasets.ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo)

        # 根据是否为gt_poses模式选择不同的数据集类
        # Choose different dataset class based on whether it's gt_poses mode
        if is_gt_poses_mode:
            print(f"使用ModelNet随机选择模式...")
            testset = ptlk.data.datasets.CADset4tracking_fixed_perturbation_random_sample(
                testdata, perturbations, fmt_trans=fmt_trans, random_seed=42)
        else:
            testset = ptlk.data.datasets.CADset4tracking_fixed_perturbation(
                testdata, perturbations, fmt_trans=fmt_trans)
    
    elif args.dataset_type == 'c3vd':
        # 修改：移除重采样，只保留基础变换
        # Modified: Remove resampling, keep only basic transformations
        transform = torchvision.transforms.Compose([
            # 不再包含任何点云处理，将在C3VDset4tracking中处理
            # No longer includes any point cloud processing, will be handled in C3VDset4tracking
        ])
        
        # 配置体素化参数（参考train_pointlk.py）
        # Configure voxelization parameters (reference train_pointlk.py)
        use_voxelization = getattr(args, 'use_voxelization', True)
        voxel_config = None
        if use_voxelization:
            # 创建体素化配置
            # Create voxelization configuration
            voxel_config = ptlk.data.datasets.VoxelizationConfig(
                voxel_size=getattr(args, 'voxel_size', 0.05),
                voxel_grid_size=getattr(args, 'voxel_grid_size', 32),
                max_voxel_points=getattr(args, 'max_voxel_points', 100),
                max_voxels=getattr(args, 'max_voxels', 20000),
                min_voxel_points_ratio=getattr(args, 'min_voxel_points_ratio', 0.1)
            )
            print(f"\n====== Voxelization Configuration ======")
            print(f"体素化配置: 体素大小={voxel_config.voxel_size}, 网格尺寸={voxel_config.voxel_grid_size}")
            print(f"每个体素最大点数={voxel_config.max_voxel_points}, 最大体素数量={voxel_config.max_voxels}")
            print(f"最小体素点数比例={voxel_config.min_voxel_points_ratio}")
        else:
            print(f"\n====== Sampling Configuration ======")
            print("使用简单重采样方法")
        
        # 创建C3VD数据集 - 配对模式支持
        # Create C3VD dataset - pairing mode support
        # 打印配对模式信息
        # Print pairing mode information
        print(f"\n====== C3VD Dataset Configuration ======")
        print(f"Pairing mode: {args.pair_mode}")
        
        # 设置源点云路径
        # Set source point cloud path
        source_root = os.path.join(args.dataset_path, 'C3VD_ply_source')
        
        # 根据配对模式设置目标点云路径
        # Set target point cloud path according to pairing mode
        if args.pair_mode == 'scene_reference':
            if args.reference_name:
                print(f"Reference point cloud name: {args.reference_name}")
            else:
                print(f"Reference point cloud: First point cloud in each scene")
            target_path = os.path.join(args.dataset_path, 'C3VD_ref')
            print(f"Target point cloud directory: {target_path}")
        else:  # one_to_one 模式 # one_to_one mode
            target_path = os.path.join(args.dataset_path, 'visible_point_cloud_ply_depth')
            print(f"Target point cloud directory: {target_path}")
            print(f"Pairing method: Each source point cloud matches target point cloud with corresponding frame number")
        
        # 创建C3VD数据集
        # Create C3VD dataset
        c3vd_dataset = ptlk.data.datasets.C3VDDataset(
            source_root=source_root,
            transform=transform,
            pair_mode=args.pair_mode,
            reference_name=args.reference_name
        )
        
        # 检查数据集是否为空
        # Check if dataset is empty
        if len(c3vd_dataset.pairs) == 0:
            print(f"Error: No paired point clouds found, please check pairing mode and data paths")
            # 输出详细的调试信息
            # Output detailed debug information
            source_root = os.path.join(args.dataset_path, 'C3VD_ply_source')
            print(f"Source point cloud directory: {source_root}")
            print(f"Directory exists: {os.path.exists(source_root)}")
            if os.path.exists(source_root):
                scenes = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
                print(f"Found scenes: {scenes}")
                if scenes:
                    for scene in scenes[:2]:  # 只显示前两个场景的信息 # Only show info for first two scenes
                        scene_dir = os.path.join(source_root, scene)
                        files = os.listdir(scene_dir)
                        print(f"Files in scene {scene}: {files[:5]}{'...' if len(files) > 5 else ''}")
            
            target_root = target_path
            print(f"Target point cloud directory: {target_root}")
            print(f"Directory exists: {os.path.exists(target_root)}")
            if os.path.exists(target_root):
                scenes = [d for d in os.listdir(target_root) if os.path.isdir(os.path.join(target_root, d))]
                print(f"Found scenes: {scenes}")
                if scenes:
                    for scene in scenes[:2]:  # 只显示前两个场景的信息 # Only show info for first two scenes
                        scene_dir = os.path.join(target_root, scene)
                        files = os.listdir(scene_dir)
                        print(f"Files in scene {scene}: {files[:5]}{'...' if len(files) > 5 else ''}")
            
            raise RuntimeError("Cannot find paired point clouds, please check dataset structure and pairing mode settings")
        
        # 应用与训练脚本相同的场景级别数据划分逻辑
        # Apply the same scene-level data splitting logic as training script
        print(f"\n====== Dataset Splitting ======")
        
        # 获取所有场景（与训练脚本完全相同的逻辑）
        # Get all scenes (exactly same logic as training script)
        all_scenes = []
        source_root_for_split = os.path.join(args.dataset_path, 'C3VD_ply_source')
        for scene_dir in glob.glob(os.path.join(source_root_for_split, "*")):
            if os.path.isdir(scene_dir):
                all_scenes.append(os.path.basename(scene_dir))
        
        # 使用固定随机种子42随机选择4个场景作为测试集（与训练脚本保持一致）
        # Use fixed random seed 42 to randomly select 4 scenes as test set (consistent with training script)
        import random
        random.seed(42)
        test_scenes = random.sample(all_scenes, 4)
        train_scenes = [scene for scene in all_scenes if scene not in test_scenes]
        
        print(f"All scenes ({len(all_scenes)}): {sorted(all_scenes)}")
        print(f"Training scenes ({len(train_scenes)}): {sorted(train_scenes)}")
        print(f"Test scenes ({len(test_scenes)}): {sorted(test_scenes)}")
        
        # 只使用测试场景的数据，过滤出测试集的索引
        # Only use test scene data, filter out test set indices
        test_indices = []
        
        for idx, (source_file, target_file) in enumerate(c3vd_dataset.pairs):
            # 从源文件路径提取场景名称
            # Extract scene name from source file path
            scene_name = None
            for scene in all_scenes:
                if f"/{scene}/" in source_file:
                    scene_name = scene
                    break
            
            # 只保留测试场景的数据
            # Only keep test scene data
            if scene_name in test_scenes:
                test_indices.append(idx)
        
        # 创建测试数据子集（只包含测试场景的数据）
        # Create test data subset (only includes test scene data)
        testdata = torch.utils.data.Subset(c3vd_dataset, test_indices)
        
        print(f"Total paired point clouds: {len(c3vd_dataset.pairs)}")
        print(f"Test set point cloud pairs (test scenes only): {len(testdata)}")
        
        # 验证测试集只包含测试场景的数据
        # Verify that test set only contains test scene data
        if len(testdata) > 0:
            # 检查前几个样本确认场景划分正确
            # Check first few samples to confirm correct scene splitting
            print(f"\nTest set scene verification:")
            sample_scenes = set()
            for i in range(min(10, len(testdata))):
                idx = testdata.indices[i]
                source_file, target_file = c3vd_dataset.pairs[idx]
                for scene in all_scenes:
                    if f"/{scene}/" in source_file:
                        sample_scenes.add(scene)
                        break
            print(f"Scenes in test set samples: {sorted(sample_scenes)}")
            print(f"Expected test scenes: {sorted(test_scenes)}")
            if sample_scenes.issubset(set(test_scenes)):
                print("✅ Scene splitting verification passed")
            else:
                print("❌ Warning: Scene splitting verification failed")
        else:
            print("❌ Error: No test samples found after scene splitting")
            raise RuntimeError("No test samples found after applying scene-based splitting")
        
        # 创建简单的变换包装类（用于C3VD数据集的扰动测试）
        # Create simple transformation wrapper class (for C3VD dataset perturbation testing)
        class SimpleRigidTransform:
            def __init__(self, perturbations_data, fmt_trans=False):
                self.perturbations = perturbations_data
                self.fmt_trans = fmt_trans
                self.igt = None  # 当前变换矩阵
                self.current_perturbation_index = 0  # 当前扰动索引
            
            def __call__(self, tensor):
                """应用刚性变换到点云
                Args:
                    tensor: 输入点云 [N, 3]
                Returns:
                    transformed_tensor: 变换后的点云 [N, 3]
                """
                try:
                    # 检查是否有扰动数据
                    if self.perturbations is None or len(self.perturbations) == 0:
                        print("警告: 没有扰动数据，返回原始点云")
                        return tensor
                    
                    # 获取当前扰动
                    if self.current_perturbation_index >= len(self.perturbations):
                        self.current_perturbation_index = 0  # 重置索引
                    
                    twist = torch.from_numpy(numpy.array(self.perturbations[self.current_perturbation_index])).contiguous().view(1, 6)
                    self.current_perturbation_index += 1  # 更新索引
                    
                    x = twist.to(tensor)
                    
                    if not self.fmt_trans:
                        # x: twist-vector
                        g = ptlk.se3.exp(x).to(tensor) # [1, 4, 4]
                        p1 = ptlk.se3.transform(g, tensor)
                        self.igt = g.squeeze(0) # igt: p0 -> p1
                    else:
                        # x: rotation and translation
                        w = x[:, 0:3]
                        q = x[:, 3:6]
                        R = ptlk.so3.exp(w).to(tensor) # [1, 3, 3]
                        g = torch.zeros(1, 4, 4)
                        g[:, 3, 3] = 1
                        g[:, 0:3, 0:3] = R # rotation
                        g[:, 0:3, 3] = q   # translation
                        p1 = ptlk.se3.transform(g, tensor)
                        self.igt = g.squeeze(0) # igt: p0 -> p1
                    
                    return p1
                except Exception as e:
                    print(f"Error during rigid transform: {e}")
                    # 在变换失败时返回原始张量
                    return tensor
        
        rigid_transform = SimpleRigidTransform(perturbations, fmt_trans)

        # 根据是否为gt_poses模式选择不同的数据集类
        if is_gt_poses_mode:
            print(f"使用C3VD随机选择模式...")
            testset = ptlk.data.datasets.C3VDset4tracking_test_random_sample(
                testdata, rigid_transform, num_points=args.num_points,
                use_voxelization=use_voxelization, voxel_config=voxel_config, random_seed=42)
        else:
            # 这是标准测试模式 - 使用 C3VDset4tracking_test
            print(f"使用C3D标准测试模式...")
            testset = ptlk.data.datasets.C3VDset4tracking_test(
                testdata, rigid_transform, num_points=args.num_points,
                use_voxelization=use_voxelization, voxel_config=voxel_config)

    else:
        raise ValueError('Unsupported dataset type: {}'.format(args.dataset_type))

    # 应用最大样本数限制
    if hasattr(args, 'max_samples') and args.max_samples is not None and args.max_samples > 0:
        if len(testset) > args.max_samples:
            print(f"限制测试集样本数到 {args.max_samples} 个（原样本数：{len(testset)}）")
            testset = torch.utils.data.Subset(testset, range(args.max_samples))
            print(f"限制后样本数: {len(testset)}")
        else:
            print(f"测试集样本数 {len(testset)} 小于或等于最大样本数限制 {args.max_samples}，使用全部样本")
    else:
        print(f"未设置最大样本数限制，或限制为0，使用全部测试集样本：{len(testset)} 个")

    return testset


def process_single_pair(args):
    """处理单对点云输入模式
    Process single pair point cloud input mode
    """
    import time
    from ptlk.data.datasets import SinglePairTrackingDataset, VoxelizationConfig
    import ptlk.se3 as se3
    
    try:
        print(f"\n========== 开始单对点云处理 Starting Single Pair Processing ==========")
        
        # 解析扰动值
        perturbation_values = [float(x.strip()) for x in args.single_perturbation.split(',')]
        if len(perturbation_values) != 6:
            raise ValueError(f"扰动值必须是6个数字 (rx,ry,rz,tx,ty,tz)，当前提供了{len(perturbation_values)}个")
        
        print(f"扰动值: {perturbation_values}")
        
        # 创建体素化配置
        voxel_config = VoxelizationConfig(
            voxel_size=args.voxel_size,
            voxel_grid_size=args.voxel_grid_size,
            max_voxel_points=args.max_voxel_points,
            max_voxels=args.max_voxels,
            min_voxel_points_ratio=args.min_voxel_points_ratio
        )
        
        # 创建单对点云跟踪数据集
        print(f"\n创建单对点云跟踪数据集...")
        testset = SinglePairTrackingDataset(
            source_cloud_path=args.source_cloud,
            target_cloud_path=args.target_cloud,
            perturbation=perturbation_values,
            num_points=args.num_points,
            use_voxelization=args.use_voxelization,
            voxel_config=voxel_config,
            fmt_trans=(args.format == 'wt')
        )
        
        # 创建数据加载器
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=1, shuffle=False, num_workers=1
        )
        
        # 创建动作执行器
        act = Action(args)
        
        # 加载模型
        print(f"\n加载模型...")
        model = act.create_model()
        device = torch.device(args.device)
        model = model.to(device)
        model.eval()
        
        print(f"模型类型: {args.model_type}")
        print(f"设备: {device}")
        
        # 进行预测
        print(f"\n开始预测...")
        with torch.no_grad():
            for i, (template, source, igt) in enumerate(testloader):
                template = template.to(device)
                source = source.to(device)
                igt = igt.to(device)
                
                print(f"\n处理批次 {i+1}/1:")
                print(f"  模板形状: {template.shape}")
                print(f"  源点云形状: {source.shape}")
                print(f"  真实变换矩阵形状: {igt.shape}")
                
                # 进行配准预测
                g_hat = act.do_estimate(template, source, model, device)
                
                print(f"  预测变换矩阵形状: {g_hat.shape}")
                
                # 计算误差
                print(f"\n========== 配准结果 Registration Results ==========")
                
                # 输出传入的扰动
                print(f"传入的扰动 Input Perturbation:")
                print(f"  向量形式: [{', '.join([f'{x:.6f}' for x in perturbation_values])}]")
                print(f"  旋转部分 (rx,ry,rz): [{', '.join([f'{x:.6f}' for x in perturbation_values[:3]])}]")
                print(f"  平移部分 (tx,ty,tz): [{', '.join([f'{x:.6f}' for x in perturbation_values[3:]])}]")
                
                # 输出预测的变换
                print(f"\n预测的变换 Predicted Transformation:")
                g_hat_np = g_hat.cpu().numpy().squeeze()
                print(f"  变换矩阵:")
                for row in range(4):
                    print(f"    [{', '.join([f'{g_hat_np[row, col]:8.6f}' for col in range(4)])}]")
                
                # 提取旋转和平移
                predicted_twist = se3.log(g_hat).cpu().numpy().squeeze()
                print(f"  扭转向量形式: [{', '.join([f'{x:.6f}' for x in predicted_twist])}]")
                print(f"  旋转部分: [{', '.join([f'{x:.6f}' for x in predicted_twist[:3]])}]")
                print(f"  平移部分: [{', '.join([f'{x:.6f}' for x in predicted_twist[3:]])}]")
                
                # 计算配准误差
                igt_np = igt.cpu().numpy().squeeze()
                g_hat_np = g_hat.cpu().numpy().squeeze()
                
                # 计算相对误差变换
                g_rel = np.linalg.inv(igt_np) @ g_hat_np
                
                # 计算旋转误差（角度）
                R_rel = g_rel[:3, :3]
                trace_R = np.trace(R_rel)
                # 避免数值误差导致的域错误
                cos_angle = (trace_R - 1) / 2
                cos_angle = np.clip(cos_angle, -1, 1)
                rotation_error_rad = np.arccos(cos_angle)
                rotation_error_deg = np.degrees(rotation_error_rad)
                
                # 计算平移误差（欧氏距离）
                t_rel = g_rel[:3, 3]
                translation_error = np.linalg.norm(t_rel)
                
                print(f"\n配准误差 Registration Error:")
                print(f"  旋转误差: {rotation_error_rad:.6f} 弧度 = {rotation_error_deg:.6f} 度")
                print(f"  平移误差: {translation_error:.6f}")
                
                # 如果启用增强输出，保存到文件
                if args.enhanced_output:
                    print(f"\n保存增强输出到文件: {args.outfile}")
                    
                    # 确保输出目录存在
                    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
                    
                    with open(args.outfile, 'w') as f:
                        f.write("# 单对点云配准结果 Single Pair Point Cloud Registration Results\n")
                        f.write(f"# 源点云: {args.source_cloud}\n")
                        f.write(f"# 目标点云: {args.target_cloud}\n")
                        f.write(f"# 模型类型: {args.model_type}\n")
                        f.write(f"# 处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("\n")
                        
                        f.write("# 传入的扰动 Input Perturbation\n")
                        f.write(f"input_perturbation_vector,{','.join([f'{x:.6f}' for x in perturbation_values])}\n")
                        f.write(f"input_rotation_part,{','.join([f'{x:.6f}' for x in perturbation_values[:3]])}\n")
                        f.write(f"input_translation_part,{','.join([f'{x:.6f}' for x in perturbation_values[3:]])}\n")
                        f.write("\n")
                        
                        f.write("# 预测的变换 Predicted Transformation\n")
                        f.write(f"predicted_twist_vector,{','.join([f'{x:.6f}' for x in predicted_twist])}\n")
                        f.write(f"predicted_rotation_part,{','.join([f'{x:.6f}' for x in predicted_twist[:3]])}\n")
                        f.write(f"predicted_translation_part,{','.join([f'{x:.6f}' for x in predicted_twist[3:]])}\n")
                        f.write("\n")
                        
                        f.write("# 预测变换矩阵 Predicted Transformation Matrix\n")
                        for row in range(4):
                            f.write(f"transformation_matrix_row_{row},{','.join([f'{g_hat_np[row, col]:.6f}' for col in range(4)])}\n")
                        f.write("\n")
                        
                        f.write("# 配准误差 Registration Error\n")
                        f.write(f"rotation_error_rad,{rotation_error_rad:.6f}\n")
                        f.write(f"rotation_error_deg,{rotation_error_deg:.6f}\n")
                        f.write(f"translation_error,{translation_error:.6f}\n")
                
                print(f"\n========== 单对点云处理完成 Single Pair Processing Completed ==========")
                break  # 只处理一个批次
                
    except Exception as e:
        print(f"单对点云处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    ARGS = options()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
        filename=ARGS.logfile)
    LOGGER.debug('Testing (PID=%d), %s', os.getpid(), ARGS)

    main(ARGS)
    LOGGER.debug('done (PID=%d)', os.getpid())

#EOF