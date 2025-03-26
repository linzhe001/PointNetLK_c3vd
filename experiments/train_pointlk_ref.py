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

# addpath('../')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import ptlk

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

    # settings for PointNet
    parser.add_argument('--pointnet', default='tune', type=str, choices=['fixed', 'tune'],
                        help='train pointnet (default: tune)')
    parser.add_argument('--transfer-from', default='', type=str,
                        metavar='PATH', help='path to pointnet features file')
    parser.add_argument('--dim-k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')

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

    # 添加新参数
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细日志')
    parser.add_argument('--drop-last', action='store_true',
                        help='丢弃不完整的最后一个批次')

    args = parser.parse_args(argv)
    return args

def main(args):
    # dataset
    trainset, testset = get_datasets(args)

    # training
    act = Action(args)
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
        p0, p1, igt = data
        data_load_time = time.time() - data_load_start
        print(f"第一批次加载时间: {data_load_time:.4f}秒")
        print(f"第一批次形状: p0={p0.shape}, p1={p1.shape}, igt={igt.shape}")
        
        # 测试批次处理时间
        if str(args.device) != 'cpu':
            p0 = p0.to(args.device)
            p1 = p1.to(args.device)
            igt = igt.to(args.device)
            torch.cuda.synchronize()
            forward_start = time.time()
            r = ptlk.pointlk.PointLK.do_forward(model, p0, p1, action.max_iter, action.xtol,
                                              action.p0_zero_mean, action.p1_zero_mean)
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

        print(f"[时间] 第{epoch+1}轮训练: {epoch_time:.2f}秒 | 损失: {running_loss:.4f} | 验证: {val_loss:.4f}")
        
        LOGGER.info('epoch, %04d, %f, %f, %f, %f', epoch + 1, running_loss, val_loss, running_info, val_info)
        snap = {'epoch': epoch + 1,
                'model': model.state_dict(),
                'min_loss': min_loss,
                'optimizer' : optimizer.state_dict(),}
        
        if is_best:
            save_checkpoint(snap, args.outfile, 'snap_best')
            save_checkpoint(model.state_dict(), args.outfile, 'model_best')
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
    def __init__(self, args):
        # PointNet
        self.pointnet = args.pointnet # tune or fixed
        self.transfer_from = args.transfer_from
        self.dim_k = args.dim_k
        self.sym_fn = None
        if args.symfn == 'max':
            self.sym_fn = ptlk.pointnet.symfn_max
        elif args.symfn == 'avg':
            self.sym_fn = ptlk.pointnet.symfn_avg
        # LK
        self.delta = args.delta
        self.learn_delta = args.learn_delta
        self.max_iter = args.max_iter
        self.xtol = 1.0e-7
        self.p0_zero_mean = True
        self.p1_zero_mean = True

        self._loss_type = 1 # see. self.compute_loss()

    def create_model(self):
        ptnet = self.create_pointnet_features()
        return self.create_from_pointnet_features(ptnet)

    def create_pointnet_features(self):
        ptnet = ptlk.pointnet.PointNet_features(self.dim_k, use_tnet=False, sym_fn=self.sym_fn)
        if self.transfer_from and os.path.isfile(self.transfer_from):
            ptnet.load_state_dict(torch.load(self.transfer_from, map_location='cpu'))
        if self.pointnet == 'tune':
            pass
        elif self.pointnet == 'fixed':
            for param in ptnet.parameters():
                param.requires_grad_(False)
        return ptnet

    def create_from_pointnet_features(self, ptnet):
        return ptlk.pointlk.PointLK(ptnet, self.delta, self.learn_delta)

    def train_1(self, model, trainloader, optimizer, device):
        model.train()
        vloss = 0.0
        gloss = 0.0
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
            loss, loss_g = self.compute_loss(model, data, device)
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
            gloss += loss_g.item()
            count += 1
            
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

        ave_vloss = float(vloss)/count
        ave_gloss = float(gloss)/count
        
        # 计算平均时间
        avg_batch = sum(batch_times) / len(batch_times) if batch_times else 0
        avg_data = sum(data_times) / len(data_times) if data_times else 0
        avg_forward = sum(forward_times) / len(forward_times) if forward_times else 0
        avg_backward = sum(backward_times) / len(backward_times) if backward_times else 0
        
        print(f"\n性能统计:")
        print(f"平均批次时间: {avg_batch:.4f}秒 = 数据加载: {avg_data:.4f}秒 + 前向传播: {avg_forward:.4f}秒 + 反向传播: {avg_backward:.4f}秒")
        print(f"训练结果: 损失={ave_vloss:.4f}, 特征损失={ave_gloss:.4f}")
        
        return ave_vloss, ave_gloss

    def eval_1(self, model, testloader, device):
        model.eval()
        vloss = 0.0
        gloss = 0.0
        count = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                loss, loss_g = self.compute_loss(model, data, device)

                vloss1 = loss.item()
                vloss += vloss1
                gloss1 = loss_g.item()
                gloss += gloss1
                count += 1

        ave_vloss = float(vloss)/count
        ave_gloss = float(gloss)/count
        return ave_vloss, ave_gloss

    def compute_loss(self, model, data, device):
        p0, p1, igt = data
        p0 = p0.to(device) # template
        p1 = p1.to(device) # source
        igt = igt.to(device) # igt: p0 -> p1
        r = ptlk.pointlk.PointLK.do_forward(model, p0, p1, self.max_iter, self.xtol,\
                                            self.p0_zero_mean, self.p1_zero_mean)
        #r = model(p0, p1, self.max_iter)
        est_g = model.g

        loss_g = ptlk.pointlk.PointLK.comp(est_g, igt)

        if self._loss_type == 0:
            loss_r = ptlk.pointlk.PointLK.rsq(r)
            loss = loss_r
        elif self._loss_type == 1:
            loss_r = ptlk.pointlk.PointLK.rsq(r)
            loss = loss_r + loss_g
        elif self._loss_type == 2:
            pr = model.prev_r
            if pr is not None:
                loss_r = ptlk.pointlk.PointLK.rsq(r - pr)
            else:
                loss_r = ptlk.pointlk.PointLK.rsq(r)
            loss = loss_r + loss_g
        else:
            loss = loss_g

        return loss, loss_g


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
        transform = torchvision.transforms.Compose([
            ptlk.data.transforms.OnUnitCube(),
            ptlk.data.transforms.Resampler(args.num_points),
        ])
        
        # 创建C3VD数据集
        c3vd_dataset = ptlk.data.datasets.C3VDDataset(
            source_root=os.path.join(args.dataset_path, 'C3VD_ply_rot_scale_trans'),
            target_root=os.path.join(args.dataset_path, 'visible_point_cloud_ply'),
            transform=transform
        )
        
        # 把数据集分成训练集和测试集
        dataset_size = len(c3vd_dataset)
        train_size = int(dataset_size * 0.8)
        test_size = dataset_size - train_size
        traindata, testdata = torch.utils.data.random_split(c3vd_dataset, [train_size, test_size])
        
        # 为训练和测试创建跟踪数据集
        mag_randomly = True
        trainset = ptlk.data.datasets.C3VDset4tracking(
            traindata, ptlk.data.transforms.RandomTransformSE3(args.mag, mag_randomly))
        testset = ptlk.data.datasets.C3VDset4tracking(
            testdata, ptlk.data.transforms.RandomTransformSE3(args.mag, mag_randomly))
    
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