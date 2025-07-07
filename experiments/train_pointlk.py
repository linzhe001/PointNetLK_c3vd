"""
    Example for training a tracker (PointNet-LK).

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
import gc
import copy
import glob
import random
import math
import traceback

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
    parser = argparse.ArgumentParser(description='PointNet-LK')

    # required.
    parser.add_argument('-o', '--outfile', required=True, type=str,
                        metavar='BASENAME', help='output filename (prefix)') # the result: ${BASENAME}_model_best.pth
    parser.add_argument('-i', '--dataset-path', required=True, type=str,
                        metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
    parser.add_argument('-c', '--categoryfile', required=True, type=str,
                        metavar='PATH', help='path to the categories to be trained') # eg. './sampledata/modelnet40_half1.txt'

    # settings for input data
    parser.add_argument('--dataset-type', default='modelnet', choices=['modelnet', 'shapenet2', 'c3vd'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--num-points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')
    parser.add_argument('--mag', default=0.8, type=float,
                        metavar='T', help='max. mag. of twist-vectors (perturbations) on training (default: 0.8)')
    # C3VD 配对模式设置
    parser.add_argument('--pair-mode', default='one_to_one', 
                        choices=['one_to_one', 'scene_reference', 'source_to_source', 'target_to_target', 'all'],
                        help='点云配对模式: one_to_one (每个源点云对应特定目标点云), scene_reference (每个场景使用一个共享目标点云), '
                             'source_to_source (源点云和源点云配对), target_to_target (目标点云和目标点云配对), '
                             'all (包含所有配对方式)')
    parser.add_argument('--reference-name', default=None, type=str,
                        help='场景参考模式下使用的目标点云名称，默认使用场景中的第一个点云')

    # settings for PointNet
    parser.add_argument('--pointnet', default='tune', type=str, choices=['fixed', 'tune'],
                        help='train pointnet (default: tune)')
    parser.add_argument('--use-tnet', dest='use_tnet', action='store_true',
                        help='flag for setting up PointNet with TNet')
    parser.add_argument('--dim-k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg', 'selective'],
                        help='symmetric function (default: max)')
    
    # 添加模型选择参数 (与train_classifier.py保持一致)
    parser.add_argument('--model-type', default='pointnet', choices=['pointnet', 'attention', 'mamba3d', 'fast_attention', 'cformer'],
                        help='选择模型类型: pointnet、attention、mamba3d、fast_attention或cformer (默认: pointnet)')
    
    # 添加attention模型特定参数 (与train_classifier.py保持一致)
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
    
    parser.add_argument('--transfer-from', default='', type=str,
                        metavar='PATH', help='path to pointnet features file')

    # settings for LK
    parser.add_argument('--max-iter', default=10, type=int,
                        metavar='N', help='max-iter on LK. (default: 10)')
    parser.add_argument('--delta', default=1.0e-2, type=float,
                        metavar='D', help='step size for approx. Jacobian (default: 1.0e-2)')
    parser.add_argument('--learn-delta', dest='learn_delta', action='store_true',
                        help='flag for training step size delta')

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

    # Additional parameters
    parser.add_argument('--verbose', action='store_true',
                        help='Display detailed logs')
    parser.add_argument('--drop-last', action='store_true',
                        help='Drop the last incomplete batch')

    # Scene split parameter
    parser.add_argument('--scene-split', action='store_true',
                        help='Split train and validation sets by scene')
    
    # 体素化相关参数
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
    
    # 添加学习率调度参数 (与train_classifier.py保持一致)
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

    # 重置日志配置，确保日志正确写入到指定文件
    if args.logfile:
        # 完全重置日志系统
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        # 配置根日志记录器
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(name)s, %(asctime)s, %(message)s',
            filename=args.logfile,
            filemode='w'  # 使用'w'模式覆盖任何已存在的日志文件
        )
        
        # 配置模块特定的日志记录器
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        print(f"日志将写入: {args.logfile}")
    
    # training
    act = Action(args)
    run(args, trainset, testset, act)


def run(args, trainset, testset, action):
    # Custom dataset wrapper that handles exceptions
    class DatasetWrapper(torch.utils.data.Dataset):
        """Wrapper for safely loading dataset samples that might cause exceptions.
        
        This wrapper catches exceptions during sample loading and returns None instead,
        which will be filtered out by the custom collate function.
        """
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            try:
                return self.dataset[idx]
            except Exception as e:
                print(f"Warning: Error loading sample at index {idx}: {str(e)}")
                return None

    # 自定义collate函数，处理None值
    def custom_collate_fn(batch):
        """自定义collate函数，过滤掉None值并检查批次大小"""
        # 移除None值
        batch = list(filter(lambda x: x is not None, batch))
        
        # 检查批次是否为空
        if len(batch) == 0:
            raise ValueError("All samples in the batch are invalid")
            
        # 检查每个元素是否包含None值
        for item in batch:
            if None in item:
                print(f"Warning: Found None in batch item: {item}")
        
        # 使用默认的collate函数处理剩余样本
        return torch.utils.data.dataloader.default_collate(batch)
    
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
    

    # Dataset statistics
    print("\n====== Detailed Dataset Statistics ======")
    
    # Get original datasets (before wrapping)
    original_trainset = trainset.dataset if isinstance(trainset, DatasetWrapper) else trainset
    original_testset = testset.dataset if isinstance(testset, DatasetWrapper) else testset
    
    if hasattr(original_trainset, 'pairs') and hasattr(original_trainset, 'scenes'):
        print(f"Training set scenes: {len(original_trainset.scenes)}")
        print(f"Training set point cloud pairs: {len(original_trainset.pairs)}")
        print(f"Point cloud pairs distribution per scene:")
        scene_counts = {}
        for scene in original_trainset.pair_scenes:
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
        for scene, count in scene_counts.items():
            print(f"  - {scene}: {count} point cloud pairs")
    
    # Validation set statistics
    if hasattr(original_testset, 'pairs') and hasattr(original_testset, 'scenes'):
        print(f"\nValidation set scenes: {len(original_testset.scenes)}")
        print(f"Validation set point cloud pairs: {len(original_testset.pairs)}")
    
    # Calculate expected batches
    total_samples = len(trainset)
    expected_batches = total_samples // args.batch_size
    if not args.drop_last and total_samples % args.batch_size != 0:
        expected_batches += 1
    
    print(f"\nBatch statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Drop last batch: {args.drop_last}")
    print(f"Expected number of batches: {expected_batches}")
    
    # Basic dataset information
    print(f"\n====== Dataset Information ======")
    print(f"Training set: {len(trainset)} samples, Test set: {len(testset)} samples")
    print(f"Batch size: {args.batch_size}, Points per cloud: {args.num_points}, Drop last batch: {args.drop_last}")
    
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
        print(f"🔄 从检查点恢复训练: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')  # 先加载到CPU，避免设备不匹配问题
        
        # 检查checkpoint的格式
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            # 完整的checkpoint格式 (包含训练状态)
            args.start_epoch = checkpoint.get('epoch', 0)
            model.load_state_dict(checkpoint['model'])
            print(f"   - 检查点类型: 完整训练状态")
            print(f"   - 恢复到epoch: {checkpoint.get('epoch', 0)}")
            print(f"   - 之前最佳损失: {checkpoint.get('min_loss', 'N/A')}")
            if 'best_epoch' in checkpoint:
                print(f"   - 之前最佳epoch: {checkpoint['best_epoch']}")
        else:
            # 只有模型权重的格式
            model.load_state_dict(checkpoint)
            args.start_epoch = 0  # 从epoch 0开始，但使用预训练权重
            checkpoint = None  # 设置为None，这样后面就不会尝试加载优化器状态
            print(f"   - 检查点类型: 仅模型权重")
            print(f"   - 将从epoch 0开始训练（使用预训练模型权重）")
            print(f"   - 注意: 优化器状态将重新初始化")

    # Wrap datasets
    trainset = DatasetWrapper(trainset)
    testset = DatasetWrapper(testset)
    
    # Data loaders
    print(f"\n====== Data Loaders ======")
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=1,  # 减少worker数量
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=min(args.workers, 2),
        drop_last=args.drop_last,
        pin_memory=(str(args.device) != 'cpu'),
        collate_fn=custom_collate_fn
    )
    
    print(f"Training batches: {len(trainloader)}, Test batches: {len(testloader)}")
    
    
    # Optimizer
    min_loss = float('inf')
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(learnable_params, lr=0.0001, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=0.001, momentum=0.9, weight_decay=1e-4)

    # 恢复训练状态
    best_epoch = 0
    if checkpoint is not None:
        min_loss = checkpoint.get('min_loss', float('inf'))
        best_epoch = checkpoint.get('best_epoch', 0)
        # 只有当checkpoint包含优化器状态时才恢复
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"✅ 成功恢复训练状态:")
            print(f"   - 当前最佳损失: {min_loss}")
            print(f"   - 当前最佳epoch: {best_epoch}")
            print(f"   - 优化器状态已恢复")
        else:
            print(f"⚠️  部分恢复训练状态:")
            print(f"   - 当前最佳损失: {min_loss}")
            print(f"   - 当前最佳epoch: {best_epoch}")
            print(f"   - 优化器状态将重新初始化（使用默认学习率）")
    
    # 使用更强的学习率调度策略
    if args.epochs > 50:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, 
            threshold=0.01, min_lr=1e-6)

    # ========================
    # 调试模式：尝试处理单个批次
    # ========================
    print("\n====== 调试模式：单批次测试 ======")
    try:
        print("获取单个批次数据...")
        debug_batch = next(iter(trainloader))
        print(f"批次数据形状: {[x.shape for x in debug_batch]}")
        
        print("\n测试前向传播...")
        model.train()  # 设置为训练模式
        
        try:
            # 在测试单个批次时捕获任何错误
            with torch.autograd.detect_anomaly():
                loss, loss_g = action.compute_loss(model, debug_batch, args.device)
                print(f"前向传播成功! loss={loss.item():.4f}, loss_g={loss_g.item():.4f}")
                
                print("\n测试反向传播...")
                optimizer.zero_grad()
                loss.backward()
                print("反向传播成功!")
                
                print("\n测试参数更新...")
                optimizer.step()
                print("参数更新成功!")
                
                print("\n单批次测试全部成功!")
        except Exception as e:
            print(f"单批次测试失败: {e}")
            traceback.print_exc()
    except Exception as e:
        print(f"无法获取测试批次: {e}")
        traceback.print_exc()
    
    print("\n是否继续完整训练？(10秒后自动继续，按Ctrl+C可中断)")
    try:
        # 设置10秒暂停，用户可以检查输出并决定是否继续
        time.sleep(10)
    except KeyboardInterrupt:
        print("用户中断训练")
        return
        
    # Training
    print("\n====== Starting Training ======")
    LOGGER.debug('train, begin')
    
    # 添加数据加载器测试
    print("测试数据加载器...")
    try:
        print("尝试获取第一个训练batch...")
        test_iter = iter(trainloader)
        first_batch = next(test_iter)
        print(f"第一个batch加载成功，形状: {[x.shape for x in first_batch]}")
        del test_iter, first_batch  # 清理内存
    except Exception as e:
        print(f"数据加载器测试失败: {e}")
        traceback.print_exc()
        return
    
    total_start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        
        consecutive_nan = 0
        max_consecutive_nan = 20
        last_valid_state = None
        
        # Save state at the beginning of each epoch
        if epoch_start:
            last_valid_state = {
                'model': copy.deepcopy(model.state_dict()),
                'optimizer': copy.deepcopy(optimizer.state_dict())
            }
        
        running_loss, running_info = action.train_1(model, trainloader, optimizer, args.device)
        val_loss, val_info = action.eval_1(model, testloader, args.device)
        
        # Detect consecutive NaNs and recover
        if not isinstance(val_loss, torch.Tensor):
            # If val_loss is a Python float and not a tensor
            if not (isinstance(val_loss, float) and math.isfinite(val_loss)):
                consecutive_nan += 1
                if consecutive_nan >= max_consecutive_nan and last_valid_state is not None:
                    print(f"Warning: Detected {consecutive_nan} consecutive NaN batches, recovering to last valid state")
                    model.load_state_dict(last_valid_state['model'])
                    optimizer.load_state_dict(last_valid_state['optimizer'])
                    consecutive_nan = 0
        else:
            # If val_loss is a tensor
            if not torch.isfinite(val_loss).all():
                consecutive_nan += 1
                if consecutive_nan >= max_consecutive_nan and last_valid_state is not None:
                    print(f"Warning: Detected {consecutive_nan} consecutive NaN batches, recovering to last valid state")
                    model.load_state_dict(last_valid_state['model'])
                    optimizer.load_state_dict(last_valid_state['optimizer'])
                    consecutive_nan = 0
        
        # Update learning rate
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # 首先定义一个变量来跟踪best epoch的数值
        is_best = val_loss < min_loss
        if is_best:
            best_epoch = epoch + 1  # 更新最佳epoch
        min_loss = min(val_loss, min_loss)

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[Time] Epoch {epoch+1}: {epoch_time:.2f} sec | Loss: {running_loss:.4f} | Validation: {val_loss:.4f}")
        print(f"[Info] Best epoch: {best_epoch}, Current LR: {current_lr:.6f}")
        
        # 修改日志输出，增加best_epoch和current_lr
        LOGGER.info('epoch, %04d, %f, %f, %f, %f, %04d, %f', 
                   epoch + 1, running_loss, val_loss, running_info, val_info, best_epoch, current_lr)
        snap = {'epoch': epoch + 1,
                'model': model.state_dict(),
                'min_loss': min_loss,
                'best_epoch': best_epoch,  # 保存最佳epoch信息
                'optimizer' : optimizer.state_dict(),}
        
        if is_best:
            save_checkpoint(snap, args.outfile, 'snap_best')
            save_checkpoint(model.state_dict(), args.outfile, 'model_best')
            print(f"[Save] Best model saved")

        # Clear cache after each epoch
        if str(args.device) != 'cpu':
            torch.cuda.empty_cache()
            gc.collect()
        
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
    def __init__(self, args):
        # PointNet相关参数
        self.pointnet = args.pointnet # tune or fixed
        self.transfer_from = args.transfer_from
        self.dim_k = args.dim_k
        self.use_tnet = args.use_tnet
        
        # 添加新的属性 (与train_classifier.py保持一致)
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
        
        # 聚合函数设置 (与train_classifier.py保持一致)
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
                
        # LK参数
        self.delta = args.delta
        self.learn_delta = args.learn_delta
        self.max_iter = args.max_iter
        self.xtol = 1.0e-7
        self.p0_zero_mean = True
        self.p1_zero_mean = True

        self._loss_type = 1 # see. self.compute_loss()

    def create_model(self):
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
                    ptnet.load_state_dict(pretrained_dict)
                    print(f"成功加载PointNet预训练权重: {self.transfer_from}")
                except Exception as e:
                    print(f"加载PointNet预训练权重失败: {e}")
                    print("继续使用随机初始化权重")
        
        if self.pointnet == 'tune':
            pass
        elif self.pointnet == 'fixed':
            for param in ptnet.parameters():
                param.requires_grad_(False)
                
        return ptlk.pointlk.PointLK(ptnet, self.delta, self.learn_delta)

    def train_1(self, model, trainloader, optimizer, device):
        model.train()
        vloss = 0.0
        gloss = 0.0
        count = 0
        nan_batch_count = 0
        nan_loss_count = 0
        nan_grad_count = 0
        
        batch_times = []
        data_times = []
        forward_times = []
        backward_times = []
        
        print("=========== 训练循环开始 ===========")
        print("总批次数: {}".format(len(trainloader)))
        print("设备: {}".format(device))
        print("当前CUDA内存: {:.1f}MB".format(torch.cuda.memory_allocated()/1024**2 if str(device) != 'cpu' else 0))
        
        batch_start = time.time()
        
        for i, data in enumerate(trainloader):
            print("\n----- 开始处理批次 {}/{} -----".format(i+1, len(trainloader)))
            data_time = time.time() - batch_start
            data_times.append(data_time)
            print("数据加载时间: {:.4f}秒".format(data_time))
            
            # 检查数据完整性
            if len(data) != 3:
                print("警告: 批次数据不完整，应有3个元素，实际有{}个".format(len(data)))
                batch_start = time.time()
                continue
                
            print("数据形状: {}".format([x.shape for x in data]))
            print("检查数据是否包含NaN: p0={}, p1={}, igt={}".format(
                torch.isnan(data[0]).any(), torch.isnan(data[1]).any(), torch.isnan(data[2]).any()))
            
            # Forward pass
            print("开始前向传播...")
            forward_start = time.time()
            
            try:
                loss, loss_g = self.compute_loss(model, data, device)
                print("损失计算完成: loss={:.4f}, loss_g={:.4f}".format(loss.item(), loss_g.item()))
            except Exception as e:
                print("前向传播或损失计算出错: {}".format(e))
                traceback.print_exc()
                nan_batch_count += 1
                batch_start = time.time()
                continue
            
            # Check if loss is NaN, if so skip this batch
            if not torch.isfinite(loss) or not torch.isfinite(loss_g):
                print(f"警告: 批次 {i} 损失值非有限 {loss.item() if torch.isfinite(loss) else 'NaN'}/{loss_g.item() if torch.isfinite(loss_g) else 'NaN'}, 跳过")
                nan_loss_count += 1
                nan_batch_count += 1
                batch_start = time.time()
                continue
                
            if str(device) != 'cpu':
                torch.cuda.synchronize()
            forward_time = time.time() - forward_start
            forward_times.append(forward_time)
            print("前向传播时间: {:.4f}秒".format(forward_time))
            
            # Backward pass
            print("开始反向传播...")
            backward_start = time.time()
            
            try:
                optimizer.zero_grad()
                print("梯度已清零")
                
                loss.backward()
                print("反向传播完成")
                
                # Stronger gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                print("梯度裁剪完成")
            except Exception as e:
                print("反向传播出错: {}".format(e))
                traceback.print_exc()
                nan_batch_count += 1
                batch_start = time.time()
                continue
            
            # Check if gradients contain NaN or Inf values
            do_step = True
            grad_check_start = time.time()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if (1 - torch.isfinite(param.grad).long()).sum() > 0:
                        do_step = False
                        print(f"警告: 参数 {name} 的梯度包含NaN/Inf值")
                        nan_grad_count += 1
                        break
            print("梯度检查耗时: {:.4f}秒".format(time.time() - grad_check_start))
            
            # Only update parameters when gradients are normal
            if do_step:
                print("更新参数...")
                try:
                    optimizer.step()
                    print("参数更新完成")
                except Exception as e:
                    print("参数更新出错: {}".format(e))
                    traceback.print_exc()
                    nan_batch_count += 1
                    batch_start = time.time()
                    continue
            else:
                # If gradients are abnormal, don't include in average loss
                print("由于梯度问题跳过参数更新")
                nan_batch_count += 1
                batch_start = time.time()
                continue
                
            if str(device) != 'cpu':
                torch.cuda.synchronize()
                print("当前CUDA内存: {:.1f}MB".format(torch.cuda.memory_allocated()/1024**2))
                
            backward_time = time.time() - backward_start
            backward_times.append(backward_time)
            print("反向传播时间: {:.4f}秒".format(backward_time))

            # Only normal batches count toward total loss and count
            vloss += loss.item()
            gloss += loss_g.item()
            count += 1
            
            # Record total batch time
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            print(f"----- 批次 {i+1}/{len(trainloader)} 完成 | 损失: {loss.item():.4f} | 用时: {batch_time:.4f}秒 -----")
            
            # Display progress every 5 batches
            if i % 5 == 0:
                if str(device) != 'cpu':
                    mem_used = torch.cuda.memory_allocated()/1024**2
                    mem_total = torch.cuda.get_device_properties(0).total_memory/1024**2
                    print(f"批次 {i+1}/{len(trainloader)} | 损失: {loss.item():.4f} | GPU内存: {mem_used:.1f}/{mem_total:.1f}MB | 用时: {batch_time:.4f} 秒")
                else:
                    print(f"批次 {i+1}/{len(trainloader)} | 损失: {loss.item():.4f} | 用时: {batch_time:.4f} 秒")
            
            batch_start = time.time()
            
            # 增加检查点，每10个批次保存一次训练状态，便于排查问题
            if i > 0 and i % 10 == 0:
                print(f"保存中间检查点 (批次 {i+1}/{len(trainloader)})")
                temp_checkpoint = {
                    'epoch': 0,  # 首个epoch
                    'batch': i,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(temp_checkpoint, '/SAN/medic/MRpcr/results/mamba3d_c3vd/mamba3d_pointlk_temp_checkpoint.pth')

        # Display NaN batch statistics
        if nan_batch_count > 0:
            print(f"\n警告: {nan_batch_count} 个批次被跳过 ({nan_batch_count/len(trainloader)*100:.2f}%)")
            print(f"- NaN损失批次: {nan_loss_count}")
            print(f"- NaN梯度批次: {nan_grad_count}")
            
        # Safely calculate averages
        ave_vloss = float(vloss)/count if count > 0 else float('inf')
        ave_gloss = float(gloss)/count if count > 0 else float('inf')
        
        # Calculate average times
        avg_batch = sum(batch_times) / len(batch_times) if batch_times else 0
        avg_data = sum(data_times) / len(data_times) if data_times else 0
        avg_forward = sum(forward_times) / len(forward_times) if forward_times else 0
        avg_backward = sum(backward_times) / len(backward_times) if backward_times else 0
        
        print(f"\n性能统计:")
        print(f"有效批次: {count}/{len(trainloader)} ({count/len(trainloader)*100:.2f}%)")
        print(f"平均批次时间: {avg_batch:.4f} 秒 = 数据加载: {avg_data:.4f} 秒 + 前向传播: {avg_forward:.4f} 秒 + 反向传播: {avg_backward:.4f} 秒")
        print(f"训练结果: 损失={ave_vloss:.4f}, 特征损失={ave_gloss:.4f}")
        
        return ave_vloss, ave_gloss

    def eval_1(self, model, testloader, device):
        model.eval()
        vloss = 0.0
        gloss = 0.0
        count = 0
        nan_count = 0
        
        print("\n====== Starting Validation ======")
        
        with torch.no_grad():
            for i, data in enumerate(testloader):
                try:
                    loss, loss_g = self.compute_loss(model, data, device)
                    
                    # Skip NaN losses
                    if not torch.isfinite(loss) or not torch.isfinite(loss_g):
                        print(f"Validation batch {i}: Loss has non-finite values, skipping")
                        nan_count += 1
                        continue

                    vloss += loss.item()
                    gloss += loss_g.item()
                    count += 1
                    
                    # Display progress every 10 batches
                    if i % 10 == 0:
                        print(f"Validation batch {i}/{len(testloader)} | Loss: {loss.item():.4f}")
                        
                except Exception as e:
                    print(f"Error processing validation batch {i}: {e}")
                    nan_count += 1
                    continue

        # Safely calculate averages
        if count > 0:
            ave_vloss = float(vloss)/count
            ave_gloss = float(gloss)/count
        else:
            print("Warning: All validation batches failed!")
            ave_vloss = 1e5  # Use a large value instead of inf
            ave_gloss = 1e5
        
        print(f"\nValidation statistics:")
        print(f"Valid batches: {count}/{len(testloader)} ({count/len(testloader)*100:.2f}%)")
        print(f"Validation results: Loss={ave_vloss:.4f}, Feature loss={ave_gloss:.4f}")
        
        if nan_count > 0:
            print(f"Evaluation: {nan_count} batches had NaN values ({nan_count/len(testloader)*100:.2f}%)")
            
        return ave_vloss, ave_gloss

    def compute_loss(self, model, data, device):
        print("====== 开始计算损失 ======")
        p0, p1, igt = data
        # 克隆张量以避免内存操作问题
        print("将数据移动到设备: {}".format(device))
        p0 = p0.clone().to(device) # template
        p1 = p1.clone().to(device) # source
        igt = igt.clone().to(device) # igt: p0 -> p1
        
        print("点云形状: p0={}, p1={}, igt={}".format(p0.shape, p1.shape, igt.shape))
        
        # Check input point clouds
        if not torch.isfinite(p0).all() or not torch.isfinite(p1).all():
            print("警告: 输入点云包含NaN或Inf值，尝试修复")
            # Replace NaN and Inf with 0
            p0 = torch.nan_to_num(p0, nan=0.0, posinf=1.0, neginf=-1.0)
            p1 = torch.nan_to_num(p1, nan=0.0, posinf=1.0, neginf=-1.0)
            print("修复后检查: p0 finite={}, p1 finite={}".format(
                torch.isfinite(p0).all(), torch.isfinite(p1).all()))
        
        try:
            # 确保在调用前正确克隆和数据迁移
            print("准备调用do_forward...")
            p0_clone = p0.clone().detach().to(device)
            p1_clone = p1.clone().detach().to(device)
            print("参数检查: max_iter={}, xtol={}, p0_zero_mean={}, p1_zero_mean={}".format(
                self.max_iter, self.xtol, self.p0_zero_mean, self.p1_zero_mean))
            
            # 打印点云的统计信息
            print("点云统计: p0_mean={:.4f}, p0_std={:.4f}, p1_mean={:.4f}, p1_std={:.4f}".format(
                p0_clone.mean().item(), p0_clone.std().item(), 
                p1_clone.mean().item(), p1_clone.std().item()))
            
            print("开始调用do_forward...")
            r_start_time = time.time()
            r = ptlk.pointlk.PointLK.do_forward(model, p0_clone, p1_clone, self.max_iter, self.xtol,
                                             self.p0_zero_mean, self.p1_zero_mean)
            r_time = time.time() - r_start_time
            print("do_forward完成，耗时: {:.4f}秒".format(r_time))
            print("r形状: {}，r统计: mean={:.4f}, std={:.4f}, min={:.4f}, max={:.4f}, 包含NaN={}".format(
                r.shape, r.mean().item(), r.std().item(), r.min().item(), r.max().item(), torch.isnan(r).any()))
            
            print("获取变换矩阵est_g...")
            est_g = model.g
            print("est_g形状: {}，包含NaN={}".format(est_g.shape, torch.isnan(est_g).any()))

            # Add numerical check
            if not torch.isfinite(est_g).all():
                print("警告: 变换矩阵包含非有限值")
                print("est_g详细信息:")
                print(est_g)
                return torch.tensor(float('nan'), device=device), torch.tensor(float('nan'), device=device)
            
            print("计算loss_g...")
            loss_g_start = time.time()
            loss_g = ptlk.pointlk.PointLK.comp(est_g, igt)
            print("loss_g计算完成: {:.4f}，耗时: {:.4f}秒".format(loss_g.item(), time.time() - loss_g_start))
            
            # Add loss value limiting
            if loss_g > 1e5:
                print(f"警告: Loss g 过大 ({loss_g.item()})，限制为 1e5")
                loss_g = torch.clamp(loss_g, max=1e5)

            print("计算最终损失...")
            if self._loss_type == 0:
                loss_r = ptlk.pointlk.PointLK.rsq(r)
                loss = loss_r
                print("损失类型 0: loss_r={:.4f}, 最终loss={:.4f}".format(loss_r.item(), loss.item()))
            elif self._loss_type == 1:
                loss_r = ptlk.pointlk.PointLK.rsq(r)
                # Set loss scale to prevent numerical issues
                loss = loss_r + loss_g * 0.1  # Reduce loss_g weight
                print("损失类型 1: loss_r={:.4f}, loss_g={:.4f}, 最终loss={:.4f}".format(
                    loss_r.item(), loss_g.item()*0.1, loss.item()))
            elif self._loss_type == 2:
                pr = model.prev_r
                if pr is not None:
                    loss_r = ptlk.pointlk.PointLK.rsq(r - pr)
                    print("使用上一个r: pr形状={}".format(pr.shape))
                else:
                    loss_r = ptlk.pointlk.PointLK.rsq(r)
                    print("没有上一个r")
                loss = loss_r + loss_g * 0.1  # Reduce loss_g weight
                print("损失类型 2: loss_r={:.4f}, loss_g={:.4f}, 最终loss={:.4f}".format(
                    loss_r.item(), loss_g.item()*0.1, loss.item()))
            else:
                loss = loss_g
                print("损失类型 其他: 直接使用loss_g={:.4f}".format(loss_g.item()))
            
            # Final check
            if not torch.isfinite(loss):
                print(f"警告: 最终损失有非有限值 {loss}")
                return torch.tensor(1e5, device=device), torch.tensor(1e5, device=device)
            
            print("====== 损失计算完成 ======")
            return loss, loss_g
        
        except Exception as e:
            print(f"损失计算错误: {e}")
            traceback.print_exc()
            return torch.tensor(float('nan'), device=device), torch.tensor(float('nan'), device=device)


class ShapeNet2_transform_coordinate:
    def __init__(self):
        pass
    def __call__(self, mesh):
        return mesh.clone().rot_x()

def get_datasets(args):

    cinfo = None
    if args.categoryfile:
        #categories = numpy.loadtxt(args.categoryfile, dtype=str, delimiter="\n").tolist()
        categories = [line.rstrip('\n') for line in open(args.categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)

    if args.dataset_type == 'modelnet':
        transform = torchvision.transforms.Compose([\
                ptlk.data.transforms.Mesh2Points(),\
                ptlk.data.transforms.OnUnitCube(),\
                ptlk.data.transforms.Resampler(args.num_points),\
            ])

        traindata = ptlk.data.datasets.ModelNet(args.dataset_path, train=1, transform=transform, classinfo=cinfo)
        testdata = ptlk.data.datasets.ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo)

        mag_randomly = True
        trainset = ptlk.data.datasets.CADset4tracking(traindata,\
                        ptlk.data.transforms.RandomTransformSE3(args.mag, mag_randomly))
        testset = ptlk.data.datasets.CADset4tracking(testdata,\
                        ptlk.data.transforms.RandomTransformSE3(args.mag, mag_randomly))

    elif args.dataset_type == 'shapenet2':
        transform = torchvision.transforms.Compose([\
                ShapeNet2_transform_coordinate(),\
                ptlk.data.transforms.Mesh2Points(),\
                ptlk.data.transforms.OnUnitCube(),\
                ptlk.data.transforms.Resampler(args.num_points),\
            ])

        dataset = ptlk.data.datasets.ShapeNet2(args.dataset_path, transform=transform, classinfo=cinfo)
        traindata, testdata = dataset.split(0.8)

        mag_randomly = True
        trainset = ptlk.data.datasets.CADset4tracking(traindata,\
                        ptlk.data.transforms.RandomTransformSE3(args.mag, mag_randomly))
        testset = ptlk.data.datasets.CADset4tracking(testdata,\
                        ptlk.data.transforms.RandomTransformSE3(args.mag, mag_randomly))

    elif args.dataset_type == 'c3vd':
        # 移除transform，因为C3VD不再需要
        # transform = torchvision.transforms.Compose([
        #     # 移除归一化，因为C3VD不再需要
        #     # ptlk.data.transforms.OnUnitCube(),
        # ])
        
        # 配置体素化参数
        use_voxelization = getattr(args, 'use_voxelization', True)
        voxel_config = None
        if use_voxelization:
            # 创建体素化配置
            voxel_config = ptlk.data.datasets.VoxelizationConfig(
                voxel_size=getattr(args, 'voxel_size', 1),
                voxel_grid_size=getattr(args, 'voxel_grid_size', 32),
                max_voxel_points=getattr(args, 'max_voxel_points', 100),
                max_voxels=getattr(args, 'max_voxels', 20000),
                min_voxel_points_ratio=getattr(args, 'min_voxel_points_ratio', 0.1)
            )
            print(f"体素化配置: 体素大小={voxel_config.voxel_size}, 网格尺寸={voxel_config.voxel_grid_size}")
        else:
            print("使用简单重采样方法")
        
        # Create C3VD dataset
        c3vd_dataset = ptlk.data.datasets.C3VDDataset(
            source_root=os.path.join(args.dataset_path, 'C3VD_ply_source'),
            target_root=os.path.join(args.dataset_path, 'visible_point_cloud_ply_depth'),
            transform=None, # 移除transform
            pair_mode=getattr(args, 'pair_mode', 'one_to_one'),
            reference_name=getattr(args, 'reference_name', None)
        )
        
        # Split based on scene or randomly
        if args.scene_split:
            # Get all scenes
            all_scenes = []
            source_root = os.path.join(args.dataset_path, 'C3VD_ply_source')
            for scene_dir in glob.glob(os.path.join(source_root, "*")):
                if os.path.isdir(scene_dir):
                    all_scenes.append(os.path.basename(scene_dir))
            
            # Randomly select 4 scenes for validation (fixed random seed to ensure consistency with classifier)
            random.seed(42)
            test_scenes = random.sample(all_scenes, 4)
            train_scenes = [scene for scene in all_scenes if scene not in test_scenes]
            
            print(f"Training scenes ({len(train_scenes)}): {train_scenes}")
            print(f"Validation scenes ({len(test_scenes)}): {test_scenes}")
            
            # Split data by scene
            train_indices = []
            test_indices = []
            
            for idx, (source_file, target_file) in enumerate(c3vd_dataset.pairs):
                # Extract scene name
                scene_name = None
                for scene in all_scenes:
                    if f"/{scene}/" in source_file:
                        scene_name = scene
                        break
                
                if scene_name in train_scenes:
                    train_indices.append(idx)
                elif scene_name in test_scenes:
                    test_indices.append(idx)
            
            # Create subsets
            traindata = torch.utils.data.Subset(c3vd_dataset, train_indices)
            testdata = torch.utils.data.Subset(c3vd_dataset, test_indices)
            
            print(f"Scene-based split: Training samples: {len(traindata)}, Validation samples: {len(testdata)}")
        else:
            # Original random split method
            dataset_size = len(c3vd_dataset)
            train_size = int(dataset_size * 0.8)
            test_size = dataset_size - train_size
            traindata, testdata = torch.utils.data.random_split(c3vd_dataset, [train_size, test_size])
            print(f"Random split: Training samples: {len(traindata)}, Validation samples: {len(testdata)}")
        
        # Create tracking datasets for training and testing with voxelization support
        mag_randomly = True
        trainset = ptlk.data.datasets.C3VDset4tracking(
            traindata, 
            ptlk.data.transforms.RandomTransformSE3(args.mag, mag_randomly),
            num_points=args.num_points,
            use_voxelization=use_voxelization,
            voxel_config=voxel_config
        )
        testset = ptlk.data.datasets.C3VDset4tracking(
            testdata, 
            ptlk.data.transforms.RandomTransformSE3(args.mag, mag_randomly),
            num_points=args.num_points,
            use_voxelization=use_voxelization,
            voxel_config=voxel_config
        )
    
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