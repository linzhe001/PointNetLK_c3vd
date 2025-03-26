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

# addpath('../')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import ptlk

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
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')

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
                        help='显示详细日志')
    parser.add_argument('--drop-last', action='store_true',
                        help='丢弃不完整的最后一个批次')

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
    # CUDA检测部分
    print(f"\n====== CUDA 可用性检查 ======")
    if torch.cuda.is_available():
        print(f"CUDA可用: 是")
        print(f"设备数量: {torch.cuda.device_count()}")
        print(f"当前设备: {torch.cuda.current_device()}")
        print(f"设备名称: {torch.cuda.get_device_name(0)}")
    else:
        print(f"CUDA可用: 否 (训练将在CPU上进行，速度会很慢)")
        args.device = 'cpu'
    
    args.device = torch.device(args.device)
    print(f"使用设备: {args.device}")
    
    # 测试CUDA速度
    if str(args.device) != 'cpu':
        print("\n====== CUDA 速度测试 ======")
        # 在GPU上测试矩阵乘法速度
        test_size = 1000
        cpu_tensor = torch.randn(test_size, test_size)
        start = time.time()
        cpu_result = cpu_tensor @ cpu_tensor
        cpu_time = time.time() - start
        
        gpu_tensor = torch.randn(test_size, test_size, device=args.device)
        # 预热GPU
        for _ in range(5):
            _ = gpu_tensor @ gpu_tensor
        torch.cuda.synchronize()
        
        start = time.time()
        gpu_result = gpu_tensor @ gpu_tensor
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"CPU矩阵乘法({test_size}x{test_size})用时: {cpu_time:.4f}秒")
        print(f"GPU矩阵乘法({test_size}x{test_size})用时: {gpu_time:.4f}秒")
        print(f"加速比: {cpu_time/gpu_time:.1f}倍")
        
        if cpu_time/gpu_time < 5:
            print("警告: GPU加速比不足5倍，可能存在CUDA配置问题")

    # 基本信息
    print(f"\n====== 数据集信息 ======")
    print(f"训练集: {len(trainset)}样本, 测试集: {len(testset)}样本")
    print(f"批次大小: {args.batch_size}, 点云数: {args.num_points}, drop_last: {args.drop_last}")
    
    # 模型初始化和加载
    model = action.create_model()
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    model.to(args.device)
    
    # 确认模型在正确设备上
    print(f"\n====== 模型信息 ======")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f"模型参数是否在CUDA上: {next(model.parameters()).is_cuda}")
    if str(args.device) != 'cpu':
        print(f"当前GPU内存使用: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        print(f"当前GPU内存缓存: {torch.cuda.memory_reserved()/1024**2:.1f}MB")

    checkpoint = None
    if args.resume:
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])

    # dataloader
    print(f"\n====== 数据加载器 ======")
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size, shuffle=True, 
        num_workers=min(args.workers, 2),  # 减少worker数量
        drop_last=args.drop_last,
        pin_memory=(str(args.device) != 'cpu'))  # 使用pin_memory加速
    
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size, shuffle=False, 
        num_workers=min(args.workers, 2),  # 减少worker数量
        drop_last=args.drop_last,
        pin_memory=(str(args.device) != 'cpu'))  # 使用pin_memory加速
    
    print(f"训练批次数: {len(trainloader)}, 测试批次数: {len(testloader)}")
    
    # 检查第一个批次并测量加载时间
    print("\n====== 测试数据加载性能 ======")
    data_load_start = time.time()
    for data in trainloader:
        points, target = data
        data_load_time = time.time() - data_load_start
        print(f"第一批次加载时间: {data_load_time:.4f}秒")
        print(f"第一批次形状: points={points.shape}, target={target.shape}")
        
        # 测试批次处理时间
        if str(args.device) != 'cpu':
            points = points.to(args.device)
            target = target.to(args.device)
            torch.cuda.synchronize()
            forward_start = time.time()
            output = model(points)
            torch.cuda.synchronize()
            forward_time = time.time() - forward_start
            print(f"前向传播时间: {forward_time:.4f}秒")
            print(f"估计每批次总时间: {data_load_time + forward_time:.4f}秒")
        break
    
    # 优化器
    min_loss = float('inf')
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(learnable_params)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=0.1)

    if checkpoint is not None:
        min_loss = checkpoint['min_loss']
        optimizer.load_state_dict(checkpoint['optimizer'])

    # 训练
    print("\n====== 开始训练 ======")
    LOGGER.debug('train, begin')
    
    total_start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        
        running_loss, running_info = action.train_1(model, trainloader, optimizer, args.device)
        val_loss, val_info = action.eval_1(model, testloader, args.device)
        
        epoch_time = time.time() - epoch_start
        
        is_best = val_loss < min_loss
        min_loss = min(val_loss, min_loss)

        print(f"[时间] 第{epoch+1}轮训练: {epoch_time:.2f}秒 | 损失: {running_loss:.4f} | 验证: {val_loss:.4f} | 准确率: {running_info:.2f}")
        
        LOGGER.info('epoch, %04d, %f, %f, %f, %f', epoch + 1, running_loss, val_loss, running_info, val_info)
        snap = {'epoch': epoch + 1,
                'model': model.state_dict(),
                'min_loss': min_loss,
                'optimizer' : optimizer.state_dict(),}
        
        if is_best:
            save_checkpoint(snap, args.outfile, 'snap_best')
            save_checkpoint(model.state_dict(), args.outfile, 'model_best')
            save_checkpoint(model.features.state_dict(), args.outfile, 'feat_best')
            print(f"[保存] 最佳模型已保存")

        # 每轮清理缓存
        if str(args.device) != 'cpu':
            torch.cuda.empty_cache()
            gc.collect()
        
        # 显示剩余时间估计
        elapsed = time.time() - total_start_time
        estimated_total = elapsed / (epoch + 1 - args.start_epoch) * (args.epochs - args.start_epoch)
        remaining = estimated_total - elapsed
        print(f"[进度] {epoch+1}/{args.epochs} 轮 | 已用时间: {elapsed/60:.1f}分钟 | 预计剩余: {remaining/60:.1f}分钟")

    total_time = time.time() - total_start_time
    print(f"\n====== 训练完成 ======")
    print(f"总训练时间: {total_time/60:.2f}分钟 ({total_time:.2f}秒)")
    print(f"平均每轮时间: {total_time/(args.epochs-args.start_epoch):.2f}秒")
    LOGGER.debug('train, end')

def save_checkpoint(state, filename, suffix):
    torch.save(state, '{}_{}.pth'.format(filename, suffix))


class Action:
    def __init__(self, args, num_classes):
        self.num_classes = num_classes
        self.dim_k = args.dim_k
        self.use_tnet = args.use_tnet
        self.sym_fn = None
        if args.symfn == 'max':
            self.sym_fn = ptlk.pointnet.symfn_max
        elif args.symfn == 'avg':
            self.sym_fn = ptlk.pointnet.symfn_avg

    def create_model(self):
        feat = ptlk.pointnet.PointNet_features(self.dim_k, self.use_tnet, self.sym_fn)
        return ptlk.pointnet.PointNet_classifier(self.num_classes, feat, self.dim_k)

    def train_1(self, model, trainloader, optimizer, device):
        model.train()
        vloss = 0.0
        pred  = 0.0
        count = 0
        
        batch_times = []
        data_times = []
        forward_times = []
        backward_times = []
        
        batch_start = time.time()
        
        for i, data in enumerate(trainloader):
            data_time = time.time() - batch_start
            data_times.append(data_time)
            
            # 前向传播
            forward_start = time.time()
            target, output, loss = self.compute_loss(model, data, device)
            if str(device) != 'cpu':
                torch.cuda.synchronize()
            forward_time = time.time() - forward_start
            forward_times.append(forward_time)
            
            # 反向传播
            backward_start = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if str(device) != 'cpu':
                torch.cuda.synchronize()
            backward_time = time.time() - backward_start
            backward_times.append(backward_time)

            vloss += loss.item()
            count += output.size(0)

            _, pred1 = output.max(dim=1)
            ag = (pred1 == target)
            am = ag.sum()
            pred += am.item()
            
            # 记录总批次时间
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # 每5个批次显示一次进度
            if i % 5 == 0:
                if str(device) != 'cpu':
                    mem_used = torch.cuda.memory_allocated()/1024**2
                    mem_total = torch.cuda.get_device_properties(0).total_memory/1024**2
                    print(f"批次 {i}/{len(trainloader)} | 损失: {loss.item():.4f} | GPU内存: {mem_used:.1f}/{mem_total:.1f}MB | 时间: {batch_time:.4f}秒")
                else:
                    print(f"批次 {i}/{len(trainloader)} | 损失: {loss.item():.4f} | 时间: {batch_time:.4f}秒")
            
            batch_start = time.time()

        running_loss = float(vloss)/count
        accuracy = float(pred)/count
        
        # 计算平均时间
        avg_batch = sum(batch_times) / len(batch_times) if batch_times else 0
        avg_data = sum(data_times) / len(data_times) if data_times else 0
        avg_forward = sum(forward_times) / len(forward_times) if forward_times else 0
        avg_backward = sum(backward_times) / len(backward_times) if backward_times else 0
        
        print(f"\n性能统计:")
        print(f"平均批次时间: {avg_batch:.4f}秒 = 数据加载: {avg_data:.4f}秒 + 前向传播: {avg_forward:.4f}秒 + 反向传播: {avg_backward:.4f}秒")
        print(f"训练结果: 损失={running_loss:.4f}, 准确率={accuracy:.4f}")
        
        return running_loss, accuracy

    def eval_1(self, model, testloader, device):
        model.eval()
        vloss = 0.0
        pred  = 0.0
        count = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                target, output, loss = self.compute_loss(model, data, device)

                loss1 = loss.item()
                vloss += loss1
                count += output.size(0)

                _, pred1 = output.max(dim=1)
                ag = (pred1 == target)
                am = ag.sum()
                pred += am.item()

        ave_loss = float(vloss)/count
        accuracy = float(pred)/count
        return ave_loss, accuracy

    def compute_loss(self, model, data, device):
        points, target = data

        points = points.to(device)
        target = target.to(device)

        output = model(points)
        loss = model.loss(output, target)

        return target, output, loss


class ShapeNet2_transform_coordinate:
    def __init__(self):
        pass
    def __call__(self, mesh):
        return mesh.clone().rot_x()

def plyread(file_path):
    """从PLY文件读取点云"""
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
        
        # 使用场景名称作为类别
        if classinfo is not None:
            self.classes, self.c_to_idx = classinfo
        else:
            # 获取场景名称作为类别
            scenes = []
            source_root = os.path.join(dataset_path, 'C3VD_ply_rot_scale_trans')
            for scene_dir in glob.glob(os.path.join(source_root, "*")):
                if os.path.isdir(scene_dir):
                    scenes.append(os.path.basename(scene_dir))
            
            scenes.sort()
            self.classes = scenes
            self.c_to_idx = {scenes[i]: i for i in range(len(scenes))}
        
        # 收集所有点云文件和它们的类别
        self.points_files = []
        self.point_classes = []
        
        # 对源点云和目标点云都进行处理
        for dir_name in ['C3VD_ply_rot_scale_trans', 'visible_point_cloud_ply']:
            for scene in self.classes:
                scene_path = os.path.join(dataset_path, dir_name, scene)
                if os.path.isdir(scene_path):
                    # 获取该场景下的所有点云
                    for ply_file in glob.glob(os.path.join(scene_path, "*.ply")):
                        self.points_files.append(ply_file)
                        self.point_classes.append(self.c_to_idx[scene])
    
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
        transform = torchvision.transforms.Compose([
            ptlk.data.transforms.OnUnitCube(),
            ptlk.data.transforms.Resampler(args.num_points),
            ptlk.data.transforms.RandomRotatorZ(),
            ptlk.data.transforms.RandomJitter()
        ])
        
        # 创建C3VD分类数据集
        dataset = C3VDDatasetForClassification(args.dataset_path, transform=transform, classinfo=cinfo)
        
        # 分割训练集和测试集
        dataset_size = len(dataset)
        train_size = int(dataset_size * 0.8)
        test_size = dataset_size - train_size
        trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
        
        # 为了保持接口一致，需要将classes和c_to_idx属性传递给分割后的数据集
        trainset.classes = dataset.classes
        testset.classes = dataset.classes
        trainset.c_to_idx = dataset.c_to_idx
        testset.c_to_idx = dataset.c_to_idx

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