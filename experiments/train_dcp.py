#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DCP 训练脚本: 支持 C3VD 和 ModelNet 数据集
"""
import os
import sys
import argparse
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
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
    parser = argparse.ArgumentParser(description='Train DCP on C3VD and ModelNet')
    parser.add_argument('--dataset-type', type=str, required=True, choices=['modelnet','c3vd'],
                        help='数据集类型: modelnet 或 c3vd')
    parser.add_argument('--dataset-path', type=str, required=True,
                        help='数据集根路径')
    parser.add_argument('--categoryfile', type=str, default='',
                        help='ModelNet 分类文件路径，仅在 modelnet 时使用')
    parser.add_argument('--pair-mode', type=str, default='one_to_one',
                        choices=['one_to_one','scene_reference','source_to_source','target_to_target','all'],
                        help='C3VD 配对模式')
    parser.add_argument('--reference-name', type=str, default=None,
                        help='C3VD 场景参考点云名称')
    parser.add_argument('--num-points', type=int, default=1024,
                        help='每个点云点数')
    parser.add_argument('--use-voxelization', action='store_true', default=False,
                        help='启用体素化预处理')

    # DCP 模型参数
    parser.add_argument('--emb-dims', type=int, default=512,
                        help='特征维度 (默认: 512)')
    parser.add_argument('--emb-nn', type=str, default='dgcnn', choices=['pointnet','dgcnn'],
                        help='嵌入网络: pointnet 或 dgcnn')
    parser.add_argument('--pointer', type=str, default='transformer', choices=['identity','transformer'],
                        help='指针网络类型')
    parser.add_argument('--head', type=str, default='mlp', choices=['mlp','svd'],
                        help='头部类型')
    parser.add_argument('--cycle', action='store_true',
                        help='使用 cycle 约束')

    # 训练参数
    parser.add_argument('--batch-size', type=int, default=8,
                        help='批大小')
    parser.add_argument('--epochs', type=int, default=200,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0,
                        help='权重衰减')
    parser.add_argument('--save-dir', type=str, default='./checkpoints/dcp',
                        help='模型保存目录')
    parser.add_argument('--resume', type=str, default='',
                        help='断点恢复模型路径')
    parser.add_argument('--pretrained', type=str, default='',
                        help='预训练模型路径')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备')

    args = parser.parse_args()
    return args


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for p0, p1, igt in loader:
        p0 = p0.to(device)  # 模板点云
        p1 = p1.to(device)  # 源点云
        igt = igt.to(device)  # 真实变换矩阵
        B = p0.size(0)
        # 转换输入形状到 [B,3,N]
        src = p0.transpose(1,2).contiguous()
        tgt = p1.transpose(1,2).contiguous()
        optimizer.zero_grad()
        rot_ab_pred, trans_ab_pred, rot_ba_pred, trans_ba_pred = model(src, tgt)
        # 提取 GT
        rot_ab_gt = igt[:, :3, :3]
        trans_ab_gt = igt[:, :3, 3]
        # 计算损失
        loss_rot = F.mse_loss(rot_ab_pred, rot_ab_gt)
        loss_trans = F.mse_loss(trans_ab_pred, trans_ab_gt)
        loss = loss_rot + loss_trans
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * B
    return total_loss / len(loader.dataset)


def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = total_mse = total_mae = 0.0
    with torch.no_grad():
        for p0, p1, igt in loader:
            p0 = p0.to(device)
            p1 = p1.to(device)
            igt = igt.to(device)
            B = p0.size(0)
            src = p0.transpose(1,2).contiguous()
            tgt = p1.transpose(1,2).contiguous()
            rot_ab_pred, trans_ab_pred, rot_ba_pred, trans_ba_pred = model(src, tgt)
            rot_ab_gt = igt[:, :3, :3]
            trans_ab_gt = igt[:, :3, 3]
            loss = F.mse_loss(rot_ab_pred, rot_ab_gt) + F.mse_loss(trans_ab_pred, trans_ab_gt)
            # 计算对齐误差
            aligned = transform_point_cloud(src, rot_ab_pred, trans_ab_pred).transpose(1,2)
            mse = F.mse_loss(aligned, p1)
            mae = torch.mean(torch.abs(aligned - p1))
            total_loss += loss.item() * B
            total_mse += mse.item() * B
            total_mae += mae.item() * B
    n = len(loader.dataset)
    return total_loss/n, total_mse/n, total_mae/n


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # 加载数据集
    trainset, testset = get_datasets(args)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             num_workers=4, pin_memory=True)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    # 初始化模型
    model = DCP(args)
    # 加载预训练或断点
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location='cpu'))
    elif args.pretrained:
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float('inf')
    # 训练循环
    for epoch in range(1, args.epochs+1):
        start = time.time()
        train_loss = train_one_epoch(model, trainloader, optimizer, device)
        val_loss, val_mse, val_mae = eval_one_epoch(model, testloader, device)
        elapsed = time.time() - start
        print(f'Epoch {epoch}/{args.epochs} | Time {elapsed:.1f}s | '
              f'Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} '
              f'| MSE {val_mse:.4f} | MAE {val_mae:.4f}')
        # 保存最优模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best.pth'))
        # 保存最新模型
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'last.pth'))

if __name__ == '__main__':
    main() 