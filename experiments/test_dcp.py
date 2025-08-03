#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DCP 测试脚本: 支持 C3VD 和 ModelNet 数据集
"""
import os
import sys
import argparse
import logging
import numpy
import torch
import torch.utils.data
import time
import random
import traceback
import math
import glob

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
# 将 experiments 目录加入 Python 路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import ptlk
from ptlk.dcp import DCP
from ptlk.dcp.util import transform_point_cloud
# 现在直接从ptlk.dcp.data导入修改后的ModelNet40
# Now directly import the modified ModelNet40 from ptlk.dcp.data
from ptlk.dcp.data import ModelNet40

# C3VD 加载依赖
import torchvision
from ptlk.data.transforms import OnUnitCube, Resampler, RandomTransformSE3
from ptlk.data.datasets import C3VDDataset, C3VDset4tracking_test, VoxelizationConfig

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

# 2. 从 train_pointlk.py 移植 RigidTransform 定义
# 2. Port RigidTransform definition from train_pointlk.py
class RigidTransform(object):
    """
    应用固定的刚性变换
    Apply fixed rigid transformation
    """
    def __init__(self, perturbations, fmt_trans=False):
        self.perturbations = perturbations
        self.fmt_trans = fmt_trans
        self.igt = None

    def __call__(self, pcloud):
        # pcloud: [N, 3]
        # 随机选择一个扰动
        # Randomly select a perturbation
        idx = random.randrange(self.perturbations.shape[0])
        x = torch.from_numpy(self.perturbations[idx, :]).contiguous().view(1, 6) # [1, 6]
        x = x.to(pcloud)
        
        if not self.fmt_trans:
            # x: twist-vector
            g = ptlk.se3.exp(x).to(pcloud) # [1, 4, 4]
            p1 = ptlk.se3.transform(g, pcloud)
            self.igt = g.squeeze(0) # igt: p0 -> p1
        else:
            # x: rotation and translation
            w = x[:, 0:3]
            q = x[:, 3:6]
            R = ptlk.so3.exp(w).to(pcloud) # [1, 3, 3]
            g = torch.zeros(1, 4, 4)
            g[:, 3, 3] = 1
            g[:, 0:3, 0:3] = R # rotation
            g[:, 0:3, 3] = q   # translation
            p1 = ptlk.se3.transform(g, pcloud)
            self.igt = g.squeeze(0) # igt: p0 -> p1
        
        return p1 #, self.igt


def options(argv=None):
    parser = argparse.ArgumentParser(description='DCP Testing')

    # required.
    parser.add_argument('-o', '--outfile', required=True, type=str,
                        metavar='FILENAME', help='output filename (.csv)')
    parser.add_argument('-i', '--dataset-path', required=True, type=str,
                        metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
    parser.add_argument('--model-path', required=True, type=str,
                        metavar='PATH', help='path to trained DCP model file')

    # settings for input data
    parser.add_argument('--dataset-type', default='modelnet', choices=['modelnet', 'c3vd'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--categoryfile', required=False, type=str,
                        metavar='PATH', help='path to the categories to be tested') # eg. './sampledata/modelnet40_half1.txt'
    parser.add_argument('--num-points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')
    parser.add_argument('--pair-mode', default='one_to_one', choices=['one_to_one', 'scene_reference'],
                        help='Point cloud pairing mode for C3VD.')
    parser.add_argument('--reference-name', default=None, type=str,
                        help='Target cloud name used in C3VD scene reference mode')

    # settings for DCP model
    parser.add_argument('--emb-dims', type=int, default=512, help='feature dimension')
    parser.add_argument('--emb-nn', type=str, default='dgcnn', choices=['pointnet', 'dgcnn'], help='embedding network')
    parser.add_argument('--pointer', type=str, default='transformer', choices=['identity', 'transformer'], help='pointer network')
    parser.add_argument('--head', type=str, default='mlp', choices=['mlp', 'svd'], help='head network')
    parser.add_argument('--n-blocks', type=int, default=1, help='transformer layers')
    parser.add_argument('--n-heads', type=int, default=4, help='transformer heads')
    parser.add_argument('--ff-dims', type=int, default=1024, help='transformer feed-forward dims')
    parser.add_argument('--dropout', type=float, default=0.0, help='transformer dropout')
    
    # 1. 添加 'cycle' 参数以修复 AttributeError
    # 1. Add 'cycle' parameter to fix AttributeError
    parser.add_argument('--cycle', action='store_true',
                        help='use cycle consistency loss (for DCP)')

    # settings for on testing
    parser.add_argument('-l', '--logfile', default='', type=str,
                        metavar='LOGNAME', help='path to logfile (default: null (no logging))')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--device', default='cpu', type=str,
                        metavar='DEVICE', help='use CUDA if available (default: cpu)')
    parser.add_argument('--max-samples', default=0, type=int,
                        metavar='N', help='Maximum number of test samples (0 for no limit)')
    
    # Perturbation settings from test_pointlk.py
    parser.add_argument('--perturbation-dir', default=None, type=str,
                        metavar='PATH', help='Perturbation directory path, will process all perturbation files in the directory')
    parser.add_argument('--perturbation-file', default=None, type=str,
                        metavar='PATH', help='Single perturbation file path (e.g., gt_poses.csv)')
    parser.add_argument('--format', default='wv', choices=['wv', 'wt'],
                        help='perturbation format (default: wv (twist)) (wt: rotation and translation)')

    args = parser.parse_args(argv)
    # Add dummy args that might be required by reused functions
    args.pretrained = None
    args.transfer_from = None
    args.use_voxelization = False # DCP doesn't use it, but C3VD loader needs it.
    args.voxel_size = 0.05
    args.voxel_grid_size = 32
    args.max_voxel_points = 100
    args.max_voxels = 20000
    args.min_voxel_points_ratio = 0.1
    return args


def main(args):
    # This main function is adapted from test_pointlk.py
    perturbation_files = []
    
    if args.perturbation_dir and os.path.exists(args.perturbation_dir):
        print(f"\n====== Perturbation Directory ======")
        print(f"Scanning perturbation directory: {args.perturbation_dir}")
        for filename in sorted(os.listdir(args.perturbation_dir)):
            if filename.endswith('.csv'):
                full_path = os.path.join(args.perturbation_dir, filename)
                # 避免重复添加 gt_poses.csv
                # Avoid adding gt_poses.csv twice
                if 'gt_poses.csv' not in os.path.basename(full_path):
                    perturbation_files.append(full_path)
                    print(f"Found perturbation file: {filename}")
    
    # 修复：确保 --perturbation-file 参数被正确处理，即使 --perturbation-dir 也被指定
    # Fix: Ensure --perturbation-file is handled correctly, even if --perturbation-dir is also specified
    if args.perturbation_file and os.path.exists(args.perturbation_file):
        if args.perturbation_file not in perturbation_files:
            perturbation_files.append(args.perturbation_file)
            print(f"Added single perturbation file: {os.path.basename(args.perturbation_file)}")
    
    if not perturbation_files:
        print("Error: No perturbation files found. Please use --perturbation-dir or --perturbation-file.")
        return
    
    print(f"Total found {len(perturbation_files)} perturbation files to process")
    
    act = Action(args)
    
    for i, pert_file in enumerate(perturbation_files):
        filename = os.path.basename(pert_file)
        print(f"\n====== Processing perturbation file [{i+1}/{len(perturbation_files)}]: {filename} ======")
        
        original_outfile = args.outfile
        original_logfile = args.logfile
        
        # Create output name for the current perturbation file
        angle_str = ""
        if filename.startswith("pert_") and "_" in filename:
            parts = filename.split("_")
            if len(parts) >= 2:
                angle_str = parts[1].split(".")[0]
        
        if angle_str:
            output_dir = os.path.join(os.path.dirname(args.outfile), f"angle_{angle_str}")
            os.makedirs(output_dir, exist_ok=True)
            base_filename = os.path.splitext(filename)[0]
            current_outfile = os.path.join(output_dir, f"results_{base_filename}.log")
            log_file = os.path.join(output_dir, f"log_{base_filename}.log")
        else:
            output_dir = os.path.dirname(args.outfile)
            os.makedirs(output_dir, exist_ok=True)
            base_filename = os.path.splitext(filename)[0]
            current_outfile = os.path.join(output_dir, f"results_{base_filename}.log")
            log_file = os.path.join(output_dir, f"log_{base_filename}.log")
        
        # This is a bit of a hack to reuse the get_datasets logic from test_pointlk.
        # It expects args.perturbations to be set.
        args.perturbations = pert_file
        args.outfile = current_outfile
        args.logfile = log_file
        
        print(f"Output file: {args.outfile}")
        print(f"Log file: {args.logfile}")
        
        testset = get_datasets(args)
        
        # Update action executor's filenames
        act.update_filenames(current_outfile, log_file, pert_file)
        
        run(args, testset, act)
        
        args.outfile = original_outfile
        args.logfile = original_logfile
        
        del testset
        torch.cuda.empty_cache()
        import gc
        gc.collect()


def run(args, testset, action):
    # This run function is adapted from test_pointlk.py
    # 强制进行CUDA检查，如果驱动不可用则退回到CPU
    # Force CUDA check, fall back to CPU if driver is unavailable
    if args.device == 'cuda:0' and not torch.cuda.is_available():
        print("警告: 指定使用cuda:0，但未找到NVIDIA驱动或CUDA不可用。将强制使用CPU。")
        args.device = 'cpu'
    
    args.device = torch.device(args.device)
    LOGGER.debug('Testing (PID=%d), %s', os.getpid(), args)

    model = action.create_model()
    model.to(args.device)

    if len(testset) == 0:
        print("Error: No valid samples in dataset, cannot continue testing.")
        return

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=1, shuffle=False, num_workers=min(args.workers, 1))
    
    LOGGER.debug('tests, begin')
    os.makedirs(os.path.dirname(action.filename), exist_ok=True)
    
    action.eval_1(model, testloader, args.device)
    LOGGER.debug('tests, end')


class Action:
    # This Action class is adapted from test_pointlk.py for DCP
    def __init__(self, args):
        self.args = args
        self.filename = args.outfile

    def create_model(self):
        model = DCP(self.args)
        if self.args.model_path and os.path.isfile(self.args.model_path):
            model.load_state_dict(torch.load(self.args.model_path, map_location='cpu'))
            print(f"Loaded DCP model from: {self.args.model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {self.args.model_path}")
        return model

    def do_estimate(self, p0, p1, model, device):
        # p0: template, [B, N, 3] or [B, 3, N]
        # p1: source, [B, N, 3] or [B, 3, N]
        # 确保输入张量在正确的设备上
        p0 = p0.to(device)
        p1 = p1.to(device)
        
        # DCP 模型期望输入为 [B, 3, N]
        # ModelNet 加载器返回 [B, 3, N], C3VD 加载器返回 [B, N, 3].
        # 这里的逻辑通过检查维度来处理这两种情况
        if p1.size(1) == 3:
            src = p1
            tgt = p0
        else:
            src = p1.transpose(1, 2).contiguous()
            tgt = p0.transpose(1, 2).contiguous()

        # model(src, tgt) 预测从 src 到 tgt 的变换, 即 p1->p0
        rot_ab_pred, trans_ab_pred, _, _ = model(src, tgt)
        
        g_hat = torch.eye(4, device=device).unsqueeze(0).repeat(p0.size(0), 1, 1)
        g_hat[:, :3, :3] = rot_ab_pred
        g_hat[:, :3, 3] = trans_ab_pred.view(-1, 3)
        return g_hat # Estimated transform from p1 to p0

    def eval_1__header(self, fout):
        cols = ['sample_id', 'scene_name', 'sequence', 'rotation_error', 'translation_error', 'total_error']
        print(','.join(map(str, cols)), file=fout)
        fout.flush()

    def eval_1__write(self, fout, ig_gt, g_hat, sample_info=None):
        dg = g_hat.bmm(ig_gt)
        dx = ptlk.se3.log(dg)
        
        rot_error = dx[:, :3]
        trans_error = dx[:, 3:]
        
        rot_norm = torch.rad2deg(rot_error.norm(p=2, dim=1)) # Report in degrees
        trans_norm = trans_error.norm(p=2, dim=1)
        total_norm = dx.norm(p=2, dim=1)
        
        for i in range(g_hat.size(0)):
            sample_id = sample_info.get('identifier', f'sample_{i}') if sample_info else f'sample_{i}'
            scene_name = sample_info.get('scene', 'unknown') if sample_info else 'unknown'
            sequence = sample_info.get('sequence', f'{i:04d}') if sample_info else f'{i:04d}'

            vals = [sample_id, scene_name, sequence, 
                   rot_norm[i].item(), trans_norm[i].item(), total_norm[i].item()]
            print(','.join(map(str, vals)), file=fout)
        fout.flush()

    def eval_1(self, model, testloader, device):
        model.eval()
        success_count = 0
        total_count = 0
        
        with open(self.filename, 'w') as fout:
            print(f"# DCP Registration Test Results", file=fout)
            print(f"# Perturbation file: {os.path.basename(self.args.perturbations)}", file=fout)
            self.eval_1__header(fout)
            with torch.no_grad():
                for i, data in enumerate(testloader):
                    total_count += 1
                    try:
                        # 对于DCP的ModelNet, data直接是(p0,p1,R_ab,t_ab,R_ba,t_ba,...)
                        if self.args.dataset_type == 'modelnet':
                            p0, p1, R_ab, t_ab, R_ba, t_ba, _, _ = data
                            B = p0.size(0)
                            # igt 是从 p1 到 p0 的地面真值变换，即 (R_ba, t_ba)
                            igt = torch.eye(4, device=p0.device).unsqueeze(0).repeat(B, 1, 1)
                            igt[:, :3, :3] = R_ba
                            igt[:, :3, 3] = t_ba
                        else: # C3VD
                            p0, p1, igt = data # p0: template, p1: source, igt: p1->p0
                        
                        g_hat = self.do_estimate(p0, p1, model, device)
                        ig_gt = igt.cpu().contiguous().view(-1, 4, 4)
                        g_hat = g_hat.cpu().contiguous().view(-1, 4, 4)

                        sample_info = {}
                        if self.args.dataset_type == 'c3vd' and hasattr(testloader.dataset, 'get_cloud_info'):
                             sample_info = testloader.dataset.get_cloud_info(i)
                        else:
                             sample_info = {'identifier': f'modelnet_sample_{i}', 'scene': 'modelnet', 'sequence': f'{i:04d}'}


                        self.eval_1__write(fout, ig_gt, g_hat, sample_info)
                        success_count += 1
                    except Exception as e:
                        print(f"Error processing sample {i}: {e}")
                        traceback.print_exc()

        print(f"\n====== Test Completed for {os.path.basename(self.args.perturbations)} ======")
        print(f"Successfully processed samples: {success_count}/{total_count}")
        print(f"Results saved to: {self.filename}")

    def update_filenames(self, outfile, logfile, pert_file):
        self.filename = outfile
        self.args.logfile = logfile
        self.args.perturbations = pert_file


# 移除不再需要的 ModelNet40_DCP_fixed_perturbation 包装类
# Remove the no longer needed ModelNet40_DCP_fixed_perturbation wrapper class

def get_datasets(args):
    perturbations = numpy.loadtxt(args.perturbations, delimiter=',')
    fmt_trans = (args.format == 'wt')

    if args.dataset_type == 'modelnet':
        if not fmt_trans:
            print("INFO: Twist vector format ('wv') detected. Converting to Euler angles and translation for ModelNet loader.")
            from scipy.spatial.transform import Rotation as sciR
            
            twists = torch.from_numpy(perturbations).float()
            # g = se3.exp(twist) gives the transformation from p0 to p1
            g = ptlk.se3.exp(twists)
            R = g[:, :3, :3]
            t = g[:, :3, 3]
            
            # The data loader ptlk.dcp.data.ModelNet40 uses rotation order Z, then Y, then X (extrinsic),
            # which is what from_euler('zyx', [z,y,x]) produces.
            # To get the angles for this from a matrix, we use as_euler('zyx').
            # This returns angles for Z, Y, X axes respectively.
            eulers_zyx_rad = sciR.from_matrix(R.numpy()).as_euler('zyx')
            
            # 数据加载器期望以度为单位的角度，但 as_euler 返回弧度。需要转换。
            # The data loader expects angles in degrees, but as_euler returns radians. Conversion is needed.
            eulers_zyx_deg = numpy.rad2deg(eulers_zyx_rad)

            # The data loader expects the perturbation columns to be [angle_x, angle_y, angle_z].
            # So we need to reverse the columns from [z, y, x] to [x, y, z].
            eulers_xyz_deg = eulers_zyx_deg[:, ::-1]
            
            # Combine to create the new perturbation array
            perturbations = numpy.concatenate([eulers_xyz_deg, t.numpy()], axis=1)
            print("INFO: Converted perturbations to Euler angles (degrees) and translation.")

        # Directly use the modified ModelNet40 loader
        testset = ModelNet40(
            num_points=args.num_points,
            partition='test',
            perturbations=perturbations)

    elif args.dataset_type == 'c3vd':
        # Reusing the logic from test_pointlk.py for C3VD
        transform = torchvision.transforms.Compose([]) # No specific transform needed here
        source_root = os.path.join(args.dataset_path, 'C3VD_ply_source')
        
        # 3. 修复C3VD数据集的路径和配对模式逻辑
        # 3. Fix path and pairing mode logic for C3VD dataset
        if args.pair_mode == 'scene_reference':
            target_root = os.path.join(args.dataset_path, 'C3VD_ref')
        else:
            target_root = os.path.join(args.dataset_path, 'visible_point_cloud_ply_depth')

        base_dataset = C3VDDataset(
            source_root=source_root,
            target_root=target_root,
            transform=transform,
            pair_mode=args.pair_mode,
            reference_name=args.reference_name
        )
        # Using a simple rigid transform wrapper that applies perturbations
        rigid_transform = RigidTransform(perturbations, fmt_trans)
        
        # We assume testing on the whole dataset as C3VD doesn't have a standard split in this context.
        # If a split is needed, logic from test_pointlk.py can be added here.
        testset = C3VDset4tracking_test(
            base_dataset, rigid_transform, num_points=args.num_points,
            use_voxelization=False, voxel_config=None) # DCP doesn't use voxelization

    else:
        raise ValueError('Unsupported dataset type: {}'.format(args.dataset_type))

    if args.max_samples and args.max_samples > 0:
        if len(testset) > args.max_samples:
            print(f"Limiting test set from {len(testset)} to {args.max_samples} samples.")
            # 使用随机索引而不是顺序索引
            # Use random indices instead of sequential indices
            indices = random.sample(range(len(testset)), args.max_samples)
            testset = torch.utils.data.Subset(testset, indices)

    return testset


if __name__ == '__main__':
    ARGS = options()
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
        filename=ARGS.logfile if ARGS.logfile else None)
    LOGGER.info('Testing (PID=%d), %s', os.getpid(), ARGS)
    main(ARGS)
    LOGGER.info('done (PID=%d)', os.getpid())
