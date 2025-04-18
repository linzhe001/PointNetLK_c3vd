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
                        help='Display detailed logs')
    parser.add_argument('--drop-last', action='store_true',
                        help='Drop the last incomplete batch')
    
    # Scene split parameter
    parser.add_argument('--scene-split', action='store_true',
                        help='Split train and validation sets by scene')

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
    
    # Test CUDA speed
    if str(args.device) != 'cpu':
        print("\n====== CUDA Speed Test ======")
        # Test matrix multiplication speed on GPU
        test_size = 1000
        cpu_tensor = torch.randn(test_size, test_size)
        start = time.time()
        cpu_result = cpu_tensor @ cpu_tensor
        cpu_time = time.time() - start
        
        gpu_tensor = torch.randn(test_size, test_size, device=args.device)
        # Warm up GPU
        for _ in range(5):
            _ = gpu_tensor @ gpu_tensor
        torch.cuda.synchronize()
        
        start = time.time()
        gpu_result = gpu_tensor @ gpu_tensor
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        
        print(f"CPU matrix multiplication ({test_size}x{test_size}) time: {cpu_time:.4f} sec")
        print(f"GPU matrix multiplication ({test_size}x{test_size}) time: {gpu_time:.4f} sec")
        print(f"Speed-up factor: {cpu_time/gpu_time:.1f}x")
        
        if cpu_time/gpu_time < 5:
            print("Warning: GPU acceleration is less than 5x, possible CUDA configuration issue")

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
        num_workers=min(args.workers, 2),  # Reduce worker count
        drop_last=args.drop_last,
        pin_memory=(str(args.device) != 'cpu'))  # Use pin_memory for acceleration
    
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size, shuffle=False, 
        num_workers=min(args.workers, 2),  # Reduce worker count
        drop_last=args.drop_last,
        pin_memory=(str(args.device) != 'cpu'))  # Use pin_memory for acceleration
    
    print(f"Training batches: {len(trainloader)}, Test batches: {len(testloader)}")
    
    # Check first batch and measure loading time
    print("\n====== Testing Data Loading Performance ======")
    data_load_start = time.time()
    for data in trainloader:
        points, target = data
        data_load_time = time.time() - data_load_start
        print(f"First batch loading time: {data_load_time:.4f} sec")
        print(f"First batch shapes: points={points.shape}, target={target.shape}")
        
        # Test batch processing time
        if str(args.device) != 'cpu':
            points = points.to(args.device)
            target = target.to(args.device)
            torch.cuda.synchronize()
            forward_start = time.time()
            output = model(points)
            torch.cuda.synchronize()
            forward_time = time.time() - forward_start
            print(f"Forward pass time: {forward_time:.4f} sec")
            print(f"Estimated total time per batch: {data_load_time + forward_time:.4f} sec")
        break
    
    # Optimizer
    min_loss = float('inf')
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(learnable_params, lr=0.0001, weight_decay=1e-5)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=0.001, momentum=0.9, weight_decay=1e-4)

    if checkpoint is not None:
        min_loss = checkpoint['min_loss']
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training
    print("\n====== Starting Training ======")
    LOGGER.debug('train, begin')
    
    total_start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        
        running_loss, running_info = action.train_1(model, trainloader, optimizer, args.device)
        print("\n====== Starting Validation ======")
        val_loss, val_info = action.eval_1(model, testloader, args.device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        is_best = val_loss < min_loss
        min_loss = min(val_loss, min_loss)

        print(f"[Time] Epoch {epoch+1}: {epoch_time:.2f} sec | Loss: {running_loss:.4f} | Validation: {val_loss:.4f} | Accuracy: {running_info:.2f}")
        
        LOGGER.info('epoch, %04d, %f, %f, %f, %f', epoch + 1, running_loss, val_loss, running_info, val_info)
        snap = {'epoch': epoch + 1,
                'model': model.state_dict(),
                'min_loss': min_loss,
                'optimizer' : optimizer.state_dict(),}
        
        if is_best:
            save_checkpoint(snap, args.outfile, 'snap_best')
            save_checkpoint(model.state_dict(), args.outfile, 'model_best')
            save_checkpoint(model.features.state_dict(), args.outfile, 'feat_best')
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
            
        running_loss = float(vloss)/count if count > 0 else float('nan')
        accuracy = float(pred)/count if count > 0 else 0.0
        
        # Calculate average times
        avg_batch = sum(batch_times) / len(batch_times) if batch_times else 0
        avg_data = sum(data_times) / len(data_times) if data_times else 0
        avg_forward = sum(forward_times) / len(forward_times) if forward_times else 0
        avg_backward = sum(backward_times) / len(backward_times) if backward_times else 0
        
        print(f"\nPerformance statistics:")
        print(f"Average batch time: {avg_batch:.4f} sec = Data loading: {avg_data:.4f} sec + Forward pass: {avg_forward:.4f} sec + Backward pass: {avg_backward:.4f} sec")
        print(f"Training results: Loss={running_loss:.4f}, Accuracy={accuracy:.4f}")
        
        return running_loss, accuracy

    def eval_1(self, model, testloader, device):
        model.eval()
        vloss = 0.0
        pred = 0.0
        count = 0
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
            
        # Safely calculate averages
        ave_loss = float(vloss)/count if count > 0 else float('inf')
        accuracy = float(pred)/count if count > 0 else 0.0
        
        # Calculate average times
        avg_batch = sum(batch_times) / len(batch_times) if batch_times else 0
        avg_data = sum(data_times) / len(data_times) if data_times else 0
        avg_forward = sum(forward_times) / len(forward_times) if forward_times else 0
        
        print(f"\nValidation performance statistics:")
        print(f"Average batch time: {avg_batch:.4f} sec = Data loading: {avg_data:.4f} sec + Forward pass: {avg_forward:.4f} sec")
        print(f"Validation results: Loss={ave_loss:.4f}, Accuracy={accuracy:.4f}")
        
        # Check if validation result is NaN
        if not numpy.isfinite(ave_loss):
            print(f"Warning: Validation loss is non-finite ({ave_loss}), which may indicate model stability issues")
            # Can use a large finite value instead of NaN to continue training
            ave_loss = 1e6
        
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
    """C3VD dataset for classification tasks"""
    def __init__(self, dataset_path, transform=None, classinfo=None):
        self.transform = transform
        
        # Use scene names as classes
        if classinfo is not None:
            self.classes, self.c_to_idx = classinfo
        else:
            # Get scene names as classes
            scenes = []
            source_root = os.path.join(dataset_path, 'C3VD_ply_rot_scale_trans')
            for scene_dir in glob.glob(os.path.join(source_root, "*")):
                if os.path.isdir(scene_dir):
                    scenes.append(os.path.basename(scene_dir))
            
            scenes.sort()
            self.classes = scenes
            self.c_to_idx = {scenes[i]: i for i in range(len(scenes))}
        
        # Collect all point cloud files and their classes
        self.points_files = []
        self.point_classes = []
        
        # Process both source and target point clouds
        for dir_name in ['C3VD_ply_rot_scale_trans', 'C3VD_ref']:
            for scene in self.classes:
                scene_path = os.path.join(dataset_path, dir_name, scene)
                if os.path.isdir(scene_path):
                    # Get all point clouds in the scene
                    for ply_file in glob.glob(os.path.join(scene_path, "*.ply")):
                        self.points_files.append(ply_file)
                        self.point_classes.append(self.c_to_idx[scene])
    
    def __len__(self):
        return len(self.points_files)
    
    def __getitem__(self, idx):
        ply_file = self.points_files[idx]
        class_idx = self.point_classes[idx]
        
        # Read point cloud
        points = plyread(ply_file)
        
        # Apply transformation
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
        
        # Create C3VD classification dataset
        dataset = C3VDDatasetForClassification(args.dataset_path, transform=transform, classinfo=cinfo)
        
        # Depending on whether to use scene split, decide split method
        if hasattr(args, 'scene_split') and args.scene_split:
            # Split by scene
            # Get all scene names
            all_scenes = dataset.classes
            print(f"Total scene count: {len(all_scenes)}")
            
            # Randomly select 4 scenes as validation set
            random.seed(42)  # Fixed random seed for reproducibility
            test_scenes = random.sample(all_scenes, 4)
            train_scenes = [scene for scene in all_scenes if scene not in test_scenes]
            
            print(f"Training scenes ({len(train_scenes)} scenes): {train_scenes}")
            print(f"Validation scenes ({len(test_scenes)} scenes): {test_scenes}")
            
            # Build indices by scene
            train_indices = []
            test_indices = []
            
            for idx, file_path in enumerate(dataset.points_files):
                # Extract scene name
                scene_name = None
                for scene in all_scenes:
                    if f"/{scene}/" in file_path:
                        scene_name = scene
                        break
                
                if scene_name in train_scenes:
                    train_indices.append(idx)
                elif scene_name in test_scenes:
                    test_indices.append(idx)
            
            # Create subsets
            trainset = torch.utils.data.Subset(dataset, train_indices)
            testset = torch.utils.data.Subset(dataset, test_indices)
            
            print(f"Training sample count: {len(trainset)}, Validation sample count: {len(testset)}")
            
            # To maintain interface consistency, need to pass classes and c_to_idx attributes to split datasets
            trainset.classes = dataset.classes
            testset.classes = dataset.classes
            trainset.c_to_idx = dataset.c_to_idx
            testset.c_to_idx = dataset.c_to_idx
        else:
            # Original random split method
            dataset_size = len(dataset)
            train_size = int(dataset_size * 0.8)
            test_size = dataset_size - train_size
            trainset, testset = torch.utils.data.random_split(dataset, [train_size, test_size])
            
            # To maintain interface consistency, need to pass classes and c_to_idx attributes to split datasets
            trainset.classes = dataset.classes
            testset.classes = dataset.classes
            trainset.c_to_idx = dataset.c_to_idx
            testset.c_to_idx = dataset.c_to_idx
            
            print(f"Random split: Training sample count: {len(trainset)}, Validation sample count: {len(testset)}")

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