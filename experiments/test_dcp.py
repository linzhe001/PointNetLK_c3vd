#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DCP 测试脚本: 支持 C3VD 和 ModelNet 数据集
"""
import os
import sys
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
# 将 experiments 目录加入 Python 路径
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import ptlk
from ptlk.dcp import DCP
from ptlk.dcp.util import transform_point_cloud
from train_pointlk import get_datasets


def parse_args():
    parser = argparse.ArgumentParser(description='Test DCP on C3VD and ModelNet')
    parser.add_argument('--dataset-type', type=str, required=True, choices=['modelnet','c3vd'],
                        help='数据集类型: modelnet 或 c3vd')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='数据集根路径')
    parser.add_argument('--categoryfile', type=str, default='',
                        help='ModelNet 分类文件路径，仅在 modelnet 时使用')
    parser.add_argument('--pair-mode', type=str, default='one_to_one',
                        help='C3VD 配对模式')
    parser.add_argument('--reference-name', type=str, default=None,
                        help='C3VD 场景参考点云名称')
    parser.add_argument('--num-points', type=int, default=1024,
                        help='每个点云点数')
    parser.add_argument('--use-voxelization', action='store_true', default=False,
                        help='启用体素化预处理')
    parser.add_argument('--model-path', type=str, required=True,
                        help='DCP 模型权重路径')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='批大小')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备')
    args = parser.parse_args()
    return args


def test(model, loader, device):
    model.eval()
    total_mse = total_mae = 0.0
    with torch.no_grad():
        for p0, p1, igt in loader:
            p0 = p0.to(device)
            p1 = p1.to(device)
            # 转换到 [B,3,N]
            src = p0.transpose(1,2).contiguous()
            tgt = p1.transpose(1,2).contiguous()
            rot_ab_pred, trans_ab_pred, rot_ba_pred, trans_ba_pred = model(src, tgt)
            # 计算对齐后的点云
            aligned = transform_point_cloud(src, rot_ab_pred, trans_ab_pred).transpose(1,2)
            mse = F.mse_loss(aligned, p1, reduction='mean')
            mae = torch.mean(torch.abs(aligned - p1))
            B = p0.size(0)
            total_mse += mse.item() * B
            total_mae += mae.item() * B
    n = len(loader.dataset)
    print(f'Test MSE: {total_mse/n:.4f} | MAE: {total_mae/n:.4f}')


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # 加载数据集
    trainset, testset = get_datasets(args)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    # 加载模型
    model = DCP(args)
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    model.to(device)
    # 运行测试
    test(model, testloader, device)

if __name__ == '__main__':
    main() 