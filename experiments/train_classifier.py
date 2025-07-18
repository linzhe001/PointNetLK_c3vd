"""
    Example for training a classifier.

    It is better that before training PointNet-LK,
    train a classifier with same dataset
    so that we can use 'transfer-learning.'
"""

import argparse
import os
import sys
import logging
import numpy
import torch
import torch.utils.data
import torchvision
import glob
from plyfile import PlyData
import numpy as np
import time
import gc
import random
import math

# addpath('../')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import ptlk
from ptlk import attention_v1
from ptlk import mamba3d_v1  # 导入Mamba3D模块
from ptlk import fast_point_attention  # 导入快速点注意力模块
from ptlk import cformer  # 导入Cformer模块

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def options(argv=None):
    parser = argparse.ArgumentParser(description='PointNet classifier')

    # required.
    parser.add_argument('-o', '--outfile', required=True, type=str,
                        metavar='BASENAME', help='output filename (prefix)') # result: ${BASENAME}_feat_best.pth
    parser.add_argument('-i', '--dataset-path', required=True, type=str,
                        metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
    parser.add_argument('-c', '--categoryfile', required=True, type=str,
                        metavar='PATH', help='path to the categories to be trained') # eg. './sampledata/modelnet40_half1.txt'

    # settings for input data
    parser.add_argument('--dataset-type', default='modelnet', choices=['modelnet', 'shapenet2', 'c3vd'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--num-points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')

    # settings for PointNet
    parser.add_argument('--use-tnet', dest='use_tnet', action='store_true',
                        help='flag for setting up PointNet with TNet')
    parser.add_argument('--dim-k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg', 'selective'],
                        help='symmetric function (default: max)')

    # 添加模型选择参数
    parser.add_argument('--model-type', default='pointnet', choices=['pointnet', 'attention', 'mamba3d', 'fast_attention', 'cformer'],
                        help='选择模型类型: pointnet、attention、mamba3d、fast_attention或cformer (默认: pointnet)')
    
    # 添加attention模型特定参数
    parser.add_argument('--num-attention-blocks', default=3, type=int,
                        metavar='N', help='attention模块中的注意力块数量 (默认: 3)')
    parser.add_argument('--num-heads', default=8, type=int,
                        metavar='N', help='多头注意力的头数 (默认: 8)')
                        
    # 添加Mamba3D模型特定参数
    parser.add_argument('--num-mamba-blocks', default=3, type=int,
                        metavar='N', help='Mamba3D模块中的Mamba块数量 (默认: 3)')
    parser.add_argument('--d-state', default=16, type=int,
                        metavar='N', help='Mamba状态空间维度 (默认: 16)')
    parser.add_argument('--expand', default=2, type=float,
                        metavar='N', help='Mamba扩展因子 (默认: 2)')

    # 添加快速点注意力模型特定参数
    parser.add_argument('--num-fast-attention-blocks', default=2, type=int,
                        metavar='N', help='快速点注意力模块中的注意力块数量 (默认: 2)')
    parser.add_argument('--fast-attention-scale', default=1, type=int,
                        metavar='N', help='快速点注意力模型的规模缩放因子 (默认: 1, 更大值表示更轻量的模型)')

    # 添加Cformer模型特定参数
    parser.add_argument('--num-proxy-points', default=8, type=int,
                        metavar='N', help='Cformer模型中的代理点数量 (默认: 8)')
    parser.add_argument('--num-blocks', default=2, type=int,
                        metavar='N', help='Cformer模型中的块数量 (默认: 2)')

    # settings for on training
    parser.add_argument('-l', '--logfile', default='', type=str,
                        metavar='LOGNAME', help='path to logfile (default: null (no logging))')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--epochs', default=200, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        metavar='METHOD', help='name of an optimizer (default: Adam)')
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')
    parser.add_argument('--verbose', action='store_true',
                        help='Display detailed logs')
    parser.add_argument('--drop-last', action='store_true',
                        help='Drop the last incomplete batch')
    
    # 在参数解析器部分添加新的学习率调度参数
    parser.add_argument('--base-lr', default=None, type=float,
                        help='基础学习率，自动设置为优化器初始学习率')
    parser.add_argument('--warmup-epochs', default=5, type=int,
                        metavar='N', help='学习率预热轮次数 (默认: 5)')
    parser.add_argument('--cosine-annealing', action='store_true',
                        help='使用余弦退火学习率策略')

    args = parser.parse_args(argv)
    return args

def main(args):
    # dataset
    trainset, testset = get_datasets(args)
    num_classes = len(trainset.classes)

    # training
    act = Action(args, num_classes)
    run(args, trainset, testset, act)


def run(args, trainset, testset, action):
    # CUDA availability check
    print(f"\n====== CUDA Availability Check ======")
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes")
        print(f"Number of devices: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print(f"CUDA Available: No (training will run on CPU, which will be slow)")
        args.device = 'cpu'
    
    args.device = torch.device(args.device)
    print(f"Using device: {args.device}")
    
    # Dataset information
    print(f"\n====== Dataset Information ======")
    print(f"Training set: {len(trainset)} samples, Test set: {len(testset)} samples")
    print(f"Batch size: {args.batch_size}, Points per cloud: {args.num_points}, Drop last: {args.drop_last}")
    
    # Model initialization and loading
    model = action.create_model()
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    model.to(args.device)
    
    # Confirm model is on correct device
    print(f"\n====== Model Information ======")
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Model parameters on CUDA: {next(model.parameters()).is_cuda}")
    if str(args.device) != 'cpu':
        print(f"Current GPU memory usage: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        print(f"Current GPU memory cached: {torch.cuda.memory_reserved()/1024**2:.1f}MB")

    checkpoint = None
    if args.resume:
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])

    # Data loaders
    print(f"\n====== Data Loaders ======")
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size, shuffle=True, 
        num_workers=min(args.workers, 4),  # Reduce worker count
        drop_last=args.drop_last,
        pin_memory=(str(args.device) != 'cpu'))  # Use pin_memory for acceleration
    
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size, shuffle=False, 
        num_workers=min(args.workers, 2),  # Reduce worker count
        drop_last=args.drop_last,
        pin_memory=(str(args.device) != 'cpu'))  # Use pin_memory for acceleration
    
    print(f"Training batches: {len(trainloader)}, Test batches: {len(testloader)}")
    
    # Optimizer
    min_loss = float('inf')
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(learnable_params, lr=0.001, weight_decay=1e-6)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=0.01, momentum=0.9, weight_decay=1e-5)

    if checkpoint is not None:
        min_loss = checkpoint['min_loss']
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # 定义学习率调度函数
    def adjust_learning_rate(epoch, optimizer, base_lr, warmup_epochs=5, cosine_annealing=True, max_epochs=200):
        """
        高级学习率调度函数，包含预热期和余弦退火
        
        Args:
            epoch: 当前训练轮次
            optimizer: 优化器实例
            base_lr: 基础学习率
            warmup_epochs: 预热期轮次数
            cosine_annealing: 是否使用余弦退火
            max_epochs: 总训练轮次
        
        Returns:
            当前学习率
        """
        # 最后50个epoch使用最低学习率
        if epoch >= max_epochs - 50:
            lr = 1e-7
        elif epoch < warmup_epochs:
            # 预热阶段：从很小的学习率线性增加到基础学习率
            lr = base_lr * (epoch + 1) / warmup_epochs
        elif cosine_annealing:
            # 余弦退火：在预热后逐渐减小学习率
            lr = base_lr * 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs - 50)))
        else:
            # 确保学习率不会太小
            lr = max(base_lr, 1e-7)
        
        # 更新所有参数组的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    # 基础学习率 - 用于计算学习率调度
    base_lr = args.base_lr if args.base_lr is not None else (0.001 if args.optimizer == 'Adam' else 0.01)
    
    # 学习率预热和余弦退火参数
    warmup_epochs = args.warmup_epochs
    cosine_annealing = args.cosine_annealing
    
    print(f"\n====== lr_scheduler ======")
    print(f"base_lr: {base_lr}")
    print(f"warmup_epochs: {warmup_epochs}")
    print(f"cosine_annealing: {cosine_annealing}")

    # Training
    print("\n====== Starting Training ======")
    LOGGER.debug('train, begin')
    
    total_start_time = time.time()
    best_epoch = 0  # 初始化best_epoch变量
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        
        # 在每个epoch开始前更新学习率
        lr = adjust_learning_rate(epoch, optimizer, base_lr, 
                                 warmup_epochs=warmup_epochs, 
                                 cosine_annealing=cosine_annealing, 
                                 max_epochs=args.epochs)
        
        running_loss, running_info = action.train_1(model, trainloader, optimizer, args.device)
        print("\n====== Starting Validation ======")
        val_loss, val_info = action.eval_1(model, testloader, args.device)
        
        epoch_time = time.time() - epoch_start
        
        is_best = val_loss < min_loss
        min_loss = min(val_loss, min_loss)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[Time] Epoch {epoch+1}: {epoch_time:.2f} sec | Loss: {running_loss:.4f} | Validation: {val_loss:.4f} | Accuracy: {running_info:.2f} | LR: {current_lr:.8f}")
        print(f"[Info] Best epoch: {best_epoch}")
        
        LOGGER.info('epoch, %04d, %f, %f, %f, %f, %04d, %f', epoch + 1, running_loss, val_loss, running_info, val_info, best_epoch, current_lr)
        snap = {'epoch': epoch + 1,
                'model': model.state_dict(),
                'min_loss': min_loss,
                'best_epoch': best_epoch,  # 保存最佳epoch信息
                'optimizer' : optimizer.state_dict(),}
        
        if is_best:
            save_checkpoint(snap, args.outfile, 'snap_best')
            save_checkpoint(model.state_dict(), args.outfile, 'model_best')
            save_checkpoint(model.features.state_dict(), args.outfile, 'feat_best')
            print(f"[Save] Best model saved")
            best_epoch = epoch + 1

        # Display estimated remaining time
        elapsed = time.time() - total_start_time
        estimated_total = elapsed / (epoch + 1 - args.start_epoch) * (args.epochs - args.start_epoch)
        remaining = estimated_total - elapsed
        print(f"[Progress] {epoch+1}/{args.epochs} epochs | Elapsed: {elapsed/60:.1f} min | Remaining: {remaining/60:.1f} min")

    total_time = time.time() - total_start_time
    print(f"\n====== Training Complete ======")
    print(f"Total training time: {total_time/60:.2f} minutes ({total_time:.2f} seconds)")
    print(f"Average time per epoch: {total_time/(args.epochs-args.start_epoch):.2f} seconds")
    LOGGER.debug('train, end')

def save_checkpoint(state, filename, suffix):
    torch.save(state, '{}_{}.pth'.format(filename, suffix))


class Action:
    def __init__(self, args, num_classes):
        self.num_classes = num_classes
        self.dim_k = args.dim_k
        self.use_tnet = args.use_tnet
        # 添加新的属性
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

    def create_model(self):
        if self.model_type == 'attention':
            # 创建attention模型
            feat = attention_v1.AttentionNet_features(
                dim_k=self.dim_k, 
                sym_fn=self.sym_fn,
                scale=1,
                num_attention_blocks=self.num_attention_blocks,
                num_heads=self.num_heads
            )
            return attention_v1.AttentionNet_classifier(self.num_classes, feat, self.dim_k)
        elif self.model_type == 'mamba3d':
            # 创建Mamba3D模型
            feat = mamba3d_v1.Mamba3D_features(
                dim_k=self.dim_k,
                sym_fn=self.sym_fn,
                scale=1,
                num_mamba_blocks=self.num_mamba_blocks,
                d_state=self.d_state,
                expand=self.expand
            )
            return mamba3d_v1.Mamba3D_classifier(self.num_classes, feat, self.dim_k)
        elif self.model_type == 'fast_attention':
            # 创建快速点注意力模型
            feat = fast_point_attention.FastPointAttention_features(
                dim_k=self.dim_k,
                sym_fn=self.sym_fn,
                scale=self.fast_attention_scale,
                num_attention_blocks=self.num_fast_attention_blocks
            )
            return fast_point_attention.FastPointAttention_classifier(self.num_classes, feat, self.dim_k)
        elif self.model_type == 'cformer':
            # 创建Cformer模型
            feat = cformer.CFormer_features(
                dim_k=self.dim_k,
                sym_fn=self.sym_fn,
                scale=1,
                num_proxy_points=self.num_proxy_points,
                num_blocks=self.num_blocks
            )
            return cformer.CFormer_classifier(self.num_classes, feat, self.dim_k)
        else:
            # 创建原始pointnet模型
            feat = ptlk.pointnet.PointNet_features(self.dim_k, self.sym_fn)
            return ptlk.pointnet.PointNet_classifier(self.num_classes, feat, self.dim_k)

    def train_1(self, model, trainloader, optimizer, device):
        model.train()
        vloss = 0.0
        pred  = 0.0
        count = 0  # 用于计算准确率的样本数
        batch_count = 0  # 用于计算平均loss的batch数
        nan_count = 0  # Count of NaN batches
        
        batch_times = []
        data_times = []
        forward_times = []
        backward_times = []
        
        batch_start = time.time()
        
        for i, data in enumerate(trainloader):
            data_time = time.time() - batch_start
            data_times.append(data_time)
            
            # Forward pass
            forward_start = time.time()
            target, output, loss = self.compute_loss(model, data, device)
            
            # If loss is NaN, skip this batch
            if not torch.isfinite(loss):
                print(f"Warning: Batch {i} has non-finite loss value {loss.item()}, skipping")
                nan_count += 1
                batch_start = time.time()
                continue
                
            if str(device) != 'cpu':
                torch.cuda.synchronize()
            forward_time = time.time() - forward_start
            forward_times.append(forward_time)
            
            # Backward pass
            backward_start = time.time()
            optimizer.zero_grad()
            loss.backward()
            
            # Add gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Check if gradients contain NaN or Inf values
            do_step = True
            for param in model.parameters():
                if param.grad is not None:
                    if (1 - torch.isfinite(param.grad).long()).sum() > 0:
                        do_step = False
                        print(f"Warning: Found non-finite gradient values, skipping parameter update")
                        break
            if do_step:
                optimizer.step()
            if str(device) != 'cpu':
                torch.cuda.synchronize()
            backward_time = time.time() - backward_start
            backward_times.append(backward_time)

            vloss += loss.item()
            batch_count += 1  # 增加batch计数
            count += output.size(0)

            _, pred1 = output.max(dim=1)
            ag = (pred1 == target)
            am = ag.sum()
            pred += am.item()
            
            # Record total batch time
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Display progress every 5 batches
            if i % 5 == 0:
                if str(device) != 'cpu':
                    mem_used = torch.cuda.memory_allocated()/1024**2
                    mem_total = torch.cuda.get_device_properties(0).total_memory/1024**2
                    print(f"Batch {i}/{len(trainloader)} | Loss: {loss.item():.4f} | GPU Memory: {mem_used:.1f}/{mem_total:.1f}MB | Time: {batch_time:.4f} sec")
                else:
                    print(f"Batch {i}/{len(trainloader)} | Loss: {loss.item():.4f} | Time: {batch_time:.4f} sec")
            
            batch_start = time.time()

        # Display NaN batch statistics
        if nan_count > 0:
            print(f"Total of {nan_count} batches were skipped due to NaN loss ({nan_count/len(trainloader)*100:.2f}%)")
            
        # 修改：使用batch_count计算平均loss
        running_loss = float(vloss)/batch_count if batch_count > 0 else float('nan')
        accuracy = float(pred)/count if count > 0 else 0.0
        
        # Calculate average times
        avg_batch = sum(batch_times) / len(batch_times) if batch_times else 0
        avg_data = sum(data_times) / len(data_times) if data_times else 0
        avg_forward = sum(forward_times) / len(forward_times) if forward_times else 0
        avg_backward = sum(backward_times) / len(backward_times) if backward_times else 0
        
        print(f"\nPerformance statistics:")
        print(f"Valid batches: {batch_count}/{len(trainloader)} ({batch_count/len(trainloader)*100:.2f}%)")
        print(f"Average batch time: {avg_batch:.4f} sec = Data loading: {avg_data:.4f} sec + Forward pass: {avg_forward:.4f} sec + Backward pass: {avg_backward:.4f} sec")
        print(f"Training results: Loss={running_loss:.4f}, Accuracy={accuracy:.4f}")
        
        return running_loss, accuracy

    def eval_1(self, model, testloader, device):
        model.eval()
        vloss = 0.0
        pred = 0.0
        count = 0  # 用于计算准确率的样本数
        batch_count = 0  # 用于计算平均loss的batch数
        nan_count = 0  # Count of NaN batches
        
        # Add time tracking
        batch_times = []
        data_times = []
        forward_times = []
        
        batch_start = time.time()
        
        print("\n====== Starting Validation ======")
        
        with torch.no_grad():
            for i, data in enumerate(testloader):
                # Record data loading time
                data_time = time.time() - batch_start
                data_times.append(data_time)
                
                try:
                    # Forward pass time measurement
                    forward_start = time.time()
                    target, output, loss = self.compute_loss(model, data, device)
                    
                    if str(device) != 'cpu':
                        torch.cuda.synchronize()
                    forward_time = time.time() - forward_start
                    forward_times.append(forward_time)
                    
                    # Check if loss is valid
                    if not torch.isfinite(loss) or loss.item() > 1e6:
                        print(f"Warning: Batch {i} has non-finite validation loss value {loss.item()}, skipping")
                        nan_count += 1
                        batch_start = time.time()
                        continue
                        
                    loss_val = loss.item()
                    vloss += loss_val
                    batch_count += 1  # 增加batch计数
                    count += output.size(0)

                    # Calculate accuracy
                    _, pred1 = output.max(dim=1)
                    ag = (pred1 == target)
                    am = ag.sum()
                    batch_acc = am.item() / target.size(0)  # Current batch accuracy
                    pred += am.item()
                    
                    # Record total batch time
                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)
                    
                    # Display progress every 5 batches or at the last batch
                    if i % 5 == 0 or i == len(testloader) - 1:
                        if str(device) != 'cpu':
                            mem_used = torch.cuda.memory_allocated()/1024**2
                            mem_total = torch.cuda.get_device_properties(0).total_memory/1024**2
                            print(f"Validation batch {i}/{len(testloader)} | Loss: {loss_val:.4f} | Accuracy: {batch_acc:.4f} | GPU Memory: {mem_used:.1f}/{mem_total:.1f}MB | Time: {batch_time:.4f} sec")
                        else:
                            print(f"Validation batch {i}/{len(testloader)} | Loss: {loss_val:.4f} | Accuracy: {batch_acc:.4f} | Time: {batch_time:.4f} sec")
                    
                    batch_start = time.time()
                    
                except Exception as e:
                    print(f"Error processing validation batch {i}: {e}")
                    nan_count += 1
                    batch_start = time.time()
                    continue

        # Display NaN batch statistics
        if nan_count > 0:
            print(f"Total of {nan_count} validation batches were skipped due to errors or non-finite loss ({nan_count/len(testloader)*100:.2f}%)")
            
        # 修改：使用batch_count计算平均loss
        ave_loss = float(vloss)/batch_count if batch_count > 0 else float('inf')
        accuracy = float(pred)/count if count > 0 else 0.0
        
        # Calculate average times
        avg_batch = sum(batch_times) / len(batch_times) if batch_times else 0
        avg_data = sum(data_times) / len(data_times) if data_times else 0
        avg_forward = sum(forward_times) / len(forward_times) if forward_times else 0
        
        print(f"\nValidation performance statistics:")
        print(f"Valid batches: {batch_count}/{len(testloader)} ({batch_count/len(testloader)*100:.2f}%)")
        print(f"Average batch time: {avg_batch:.4f} sec = Data loading: {avg_data:.4f} sec + Forward pass: {avg_forward:.4f} sec")
        print(f"Validation results: Loss={ave_loss:.4f}, Accuracy={accuracy:.4f}")
        
        return ave_loss, accuracy

    def compute_loss(self, model, data, device):
        points, target = data
        points = points.to(device)
        target = target.to(device)
        
        # Check for NaN inputs
        if not torch.isfinite(points).all():
            print("Warning: Input point cloud contains NaN values")
            # Replace NaN and Inf with 0
            points = points.clone()
            points[torch.isnan(points)] = 0.0
            points[torch.isinf(points)] = 0.0
        
        output = model(points)
        loss = model.loss(output, target)
        return target, output, loss


class ShapeNet2_transform_coordinate:
    def __init__(self):
        pass
    def __call__(self, mesh):
        return mesh.clone().rot_x()

def plyread(file_path):
    """Read point cloud from PLY file"""
    ply_data = PlyData.read(file_path)
    pc = np.vstack([
        ply_data['vertex']['x'],
        ply_data['vertex']['y'],
        ply_data['vertex']['z']
    ]).T
    return torch.from_numpy(pc.astype(np.float32))

class C3VDDatasetForClassification(torch.utils.data.Dataset):
    """C3VD数据集用于分类任务"""
    def __init__(self, dataset_path, transform=None, classinfo=None):
        self.transform = transform
        
        # 添加调试信息
        print(f"\n====== C3VDDatasetForClassification调试信息 ======")
        print(f"数据集路径: {dataset_path}")
        
        # 确定源点云和目标点云目录
        self.source_root = os.path.join(dataset_path, 'C3VD_ply_source')
        self.target_root = os.path.join(dataset_path, 'visible_point_cloud_ply_depth')
        
        # 检查目录是否存在
        print(f"源点云目录 {self.source_root} 存在: {os.path.exists(self.source_root)}")
        print(f"目标点云目录 {self.target_root} 存在: {os.path.exists(self.target_root)}")
        
        # 如果目录不存在，打印当前目录内容
        if not os.path.exists(self.source_root) or not os.path.exists(self.target_root):
            print(f"数据集根目录内容: {os.listdir(dataset_path) if os.path.exists(dataset_path) else '无法访问'}")
            
        # 使用classinfo如果提供了，否则从文件夹名提取类别
        if classinfo is not None:
            self.classes, self.c_to_idx = classinfo
            print(f"从classinfo加载类别: {self.classes}")
        else:
            # 获取场景并提取前缀作为类别
            scene_prefixes = set()
            scene_dirs = glob.glob(os.path.join(self.source_root, "*"))
            print(f"找到场景目录数量: {len(scene_dirs)}")
            
            for scene_dir in scene_dirs:
                if os.path.isdir(scene_dir):
                    scene_name = os.path.basename(scene_dir)
                    # 提取第一个下划线前的部分作为类别
                    prefix = scene_name.split('_')[0]
                    scene_prefixes.add(prefix)
                    print(f"场景: {scene_name}, 提取类别: {prefix}")
            
            # 排序并创建类别索引映射
            self.classes = sorted(list(scene_prefixes))
            self.c_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
            
            print(f"从文件夹名称提取了 {len(self.classes)} 个类别: {self.classes}")
        
        # 收集点云文件和它们的类别
        self.points_files = []
        self.point_classes = []
        self.scene_to_prefix = {}  # 映射场景名到类别前缀
        self.pair_scenes = []  # 记录每个点云所属的场景，用于场景分割
        
        # 创建场景到前缀的映射
        scene_dirs = glob.glob(os.path.join(self.source_root, "*"))
        
        for scene_dir in scene_dirs:
            if os.path.isdir(scene_dir):
                scene_name = os.path.basename(scene_dir)
                prefix = scene_name.split('_')[0]
                if prefix in self.c_to_idx:
                    self.scene_to_prefix[scene_name] = prefix
        
        # 处理源点云
        for scene in self.scene_to_prefix:
            source_path = os.path.join(self.source_root, scene)
            if os.path.isdir(source_path):
                prefix = self.scene_to_prefix[scene]
                # 获取场景中的所有源点云
                source_files = glob.glob(os.path.join(source_path, "????_depth_pcd.ply"))
                for ply_file in source_files:
                    self.points_files.append(ply_file)
                    self.point_classes.append(self.c_to_idx[prefix])
                    self.pair_scenes.append(scene)  # 记录点云所属场景
        
        # 处理目标点云
        for scene in self.scene_to_prefix:
            target_path = os.path.join(self.target_root, scene)
            if os.path.isdir(target_path):
                prefix = self.scene_to_prefix[scene]
                # 获取场景中的所有目标点云
                target_files = glob.glob(os.path.join(target_path, "*.ply"))
                
                for ply_file in target_files:
                    self.points_files.append(ply_file)
                    self.point_classes.append(self.c_to_idx[prefix])
                    self.pair_scenes.append(scene)  # 记录点云所属场景
        
        # 打印每个类别的样本数量
        class_counts = {}
        for class_idx in self.point_classes:
            class_name = self.classes[class_idx]
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
        
        print("class_counts:")
        for class_name, count in class_counts.items():
            print(f"  - {class_name}: {count}")
            
        if len(self.points_files) == 0:
            print(f"错误：没有找到有效的点云文件！请检查数据集路径和结构。")
            print(f"源点云路径: {self.source_root}")
            print(f"目标点云路径: {self.target_root}")
            
            # 尝试列出路径下的文件和目录
            print("\n检查数据集目录结构:")
            if os.path.exists(dataset_path):
                print(f"数据集根目录 {dataset_path} 内容: {os.listdir(dataset_path)}")
                
                # 检查C3VD_ply_source目录
                if os.path.exists(self.source_root):
                    source_dirs = os.listdir(self.source_root)
                    print(f"源点云目录 {self.source_root} 内容: {source_dirs}")
                    # 如果有场景目录，检查第一个场景目录
                    if source_dirs:
                        first_scene = os.path.join(self.source_root, source_dirs[0])
                        if os.path.isdir(first_scene):
                            print(f"第一个场景目录 {first_scene} 内容: {os.listdir(first_scene)}")
                
                # 检查目标点云目录
                if os.path.exists(self.target_root):
                    target_dirs = os.listdir(self.target_root)
                    print(f"目标点云目录 {self.target_root} 内容: {target_dirs}")
                    # 如果有场景目录，检查第一个场景目录
                    if target_dirs:
                        first_scene = os.path.join(self.target_root, target_dirs[0])
                        if os.path.isdir(first_scene):
                            print(f"第一个场景目录 {first_scene} 内容: {os.listdir(first_scene)}")
                            
            # 检查类别文件
            if classinfo is not None and hasattr(self, 'classes'):
                print(f"\n类别信息: {self.classes}")
 
    def __len__(self):
        return len(self.points_files)
    
    def __getitem__(self, idx):
        ply_file = self.points_files[idx]
        class_idx = self.point_classes[idx]
        
        # 读取点云
        points = plyread(ply_file)
        
        # 应用变换
        if self.transform:
            points = self.transform(points)
        
        return points, class_idx

    def get_scene_indices(self, scene_names):
        """获取指定场景的样本索引"""
        indices = []
        for i, scene in enumerate(self.pair_scenes):
            if scene in scene_names:
                indices.append(i)
        
        return indices

class C3VDClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, num_points=1024):
        self.dataset = dataset
        self.num_points = num_points
        
        # 使用来自transforms.py的标准变换类
        self.resampler = ptlk.data.transforms.Resampler(num_points)
        self.normalizer = ptlk.data.transforms.OnUnitCube()
        
        # 数据增强变换
        self.rotator_z = ptlk.data.transforms.RandomRotatorZ()
        self.jitter = ptlk.data.transforms.RandomJitter(scale=0.01, clip=0.05)
        
        # 保留原始数据集的属性
        if hasattr(dataset, 'classes'):
            self.classes = dataset.classes
            self.c_to_idx = dataset.c_to_idx
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'classes'):
            # 如果是Subset对象，访问其底层dataset
            self.classes = dataset.dataset.classes
            self.c_to_idx = dataset.dataset.c_to_idx
        else:
            print("警告: 无法找到类别信息！")
            self.classes = ["unknown"]
            self.c_to_idx = {"unknown": 0}
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # 获取原始点云和类别
        points, class_idx = self.dataset[idx]
        
        # 检查点云有效性
        if not torch.isfinite(points).all():
            print(f"警告: 索引 {idx} 的点云包含无效值，尝试修复")
            points = torch.nan_to_num(points, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 重采样确保点云有相同数量的点
        if points.shape[0] > self.num_points:
            points = self.resampler(points)
        
        # 标准化到单位立方体
        points = self.normalizer(points)
        
        # 应用数据增强
        # 1. 应用Z轴旋转
        points = self.rotator_z(points)
        
        # 2. 添加随机抖动
        if torch.rand(1).item() > 0.5:  # 50%几率添加抖动
            points = self.jitter(points)
        
        return points, class_idx

def get_datasets(args):
    # 检查数据集路径是否存在
    if not os.path.exists(args.dataset_path):
        print(f"错误: 数据集路径 {args.dataset_path} 不存在!")
        # 尝试列出父目录内容
        parent_dir = os.path.dirname(args.dataset_path)
        if os.path.exists(parent_dir):
            print(f"父目录 {parent_dir} 内容: {os.listdir(parent_dir)}")
    else:
        print(f"数据集路径存在，内容: {os.listdir(args.dataset_path)}")
    
    # 检查类别文件是否存在
    if args.categoryfile and not os.path.exists(args.categoryfile):
        print(f"错误: 类别文件 {args.categoryfile} 不存在!")
    elif args.categoryfile:
        print(f"类别文件存在，内容:")
        with open(args.categoryfile, 'r') as f:
            print(f.read())

    cinfo = None
    if args.categoryfile:
        categories = [line.rstrip('\n') for line in open(args.categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)
        print(f"加载的类别: {categories}")

    if args.dataset_type == 'modelnet':
        transform = torchvision.transforms.Compose([\
                ptlk.data.transforms.Mesh2Points(),\
                ptlk.data.transforms.OnUnitCube(),\
                ptlk.data.transforms.Resampler(args.num_points),\
                ptlk.data.transforms.RandomRotatorZ(),\
                ptlk.data.transforms.RandomJitter()\
            ])

        trainset = ptlk.data.datasets.ModelNet(args.dataset_path, train=1, transform=transform, classinfo=cinfo)
        testset = ptlk.data.datasets.ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo)

    elif args.dataset_type == 'shapenet2':
        transform = torchvision.transforms.Compose([\
                ShapeNet2_transform_coordinate(),\
                ptlk.data.transforms.Mesh2Points(),\
                ptlk.data.transforms.OnUnitCube(),\
                ptlk.data.transforms.Resampler(args.num_points),\
                ptlk.data.transforms.RandomRotatorZ(),\
                ptlk.data.transforms.RandomJitter()\
            ])

        dataset = ptlk.data.datasets.ShapeNet2(args.dataset_path, transform=transform, classinfo=cinfo)
        trainset, testset = dataset.split(0.8)

    elif args.dataset_type == 'c3vd':
        print("\n====== 正在创建C3VD数据集 ======")
        # 创建C3VD分类数据集
        c3vd_dataset = C3VDDatasetForClassification(
            args.dataset_path,
            transform=None,
            classinfo=cinfo
        )
        
        # 检查数据集是否为空
        if hasattr(c3vd_dataset, 'points_files'):
            print(f"数据集点云文件数量: {len(c3vd_dataset.points_files)}")
            if len(c3vd_dataset.points_files) == 0:
                print("警告: 数据集中没有点云文件!")
        else:
            print("警告: 数据集没有points_files属性!")
        
        # 随机分割
        print("\n====== random_split ======")
        indices = list(range(len(c3vd_dataset)))
        print(f"total number: {len(indices)}")
        
        if len(indices) == 0:
            raise ValueError("数据集为空！请检查数据集路径和配置。")
            
        split = int(numpy.floor(0.2 * len(c3vd_dataset)))
        numpy.random.seed(42)
        numpy.random.shuffle(indices)
        train_indices, test_indices = indices[split:], indices[:split]
        
        
        try:
            # 创建子集
            train_subset = torch.utils.data.Subset(c3vd_dataset, train_indices)
            test_subset = torch.utils.data.Subset(c3vd_dataset, test_indices)
            
            # 创建用于分类的数据集
            trainset = C3VDClassifierDataset(
                dataset=train_subset,
                num_points=args.num_points
            )
            
            testset = C3VDClassifierDataset(
                dataset=test_subset,
                num_points=args.num_points
            )
        except Exception as e:
            print(f"创建数据集时出错: {str(e)}")
            raise
    
    # 添加安全检查
    if len(trainset) == 0:
        raise ValueError("最终训练集为空！请检查数据处理逻辑。")
    if len(testset) == 0:
        raise ValueError("最终测试集为空！请检查数据处理逻辑。")
        
    print(f"train size: {len(trainset)}, test size: {len(testset)}")
    
    return trainset, testset


if __name__ == '__main__':
    ARGS = options()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
        filename=ARGS.logfile)
    LOGGER.debug('Training (PID=%d), %s', os.getpid(), ARGS)

    main(ARGS)
    LOGGER.debug('done (PID=%d)', os.getpid())

#EOF