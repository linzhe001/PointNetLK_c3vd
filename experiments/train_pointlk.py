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

    # Additional parameters
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
                print(f"Warning: Skipping sample at index {idx}: {str(e)}")
                return None

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
    
    # Problematic point cloud file statistics
    print("\n====== Problematic Point Cloud File Statistics ======")
    invalid_samples = []
    total_tested = 0
    problem_files = {}  # For storing information about problematic files
    
    for idx in range(len(trainset)):  # Test all samples
        try:
            trainset[idx]
            total_tested += 1
        except Exception as e:
            # Get original file information
            if hasattr(original_trainset, 'pairs'):
                source_file, target_file = original_trainset.pairs[idx]
                scene = original_trainset.pair_scenes[idx]
                error_info = {
                    'source_file': source_file,
                    'target_file': target_file,
                    'scene': scene,
                    'error': str(e)
                }
                invalid_samples.append((idx, error_info))
                
                # Track problems by scene
                if scene not in problem_files:
                    problem_files[scene] = []
                problem_files[scene].append(source_file)
    
    if invalid_samples:
        print(f"\nFound {len(invalid_samples)} invalid samples:")
        print("\nProblem files by scene:")
        for scene, files in problem_files.items():
            print(f"\nScene {scene} problem files ({len(files)}):")
            for file in files:
                print(f"  - {os.path.basename(file)}")
        
        print("\nDetailed error information:")
        for idx, info in invalid_samples:
            print(f"\nIndex {idx}:")
            print(f"  Scene: {info['scene']}")
            print(f"  Source file: {os.path.basename(info['source_file'])}")
            print(f"  Target file: {os.path.basename(info['target_file'])}")
            print(f"  Error: {info['error']}")
    
    print(f"\nTest results summary:")
    print(f"Total tested samples: {total_tested}")
    print(f"Valid samples: {total_tested - len(invalid_samples)}")
    print(f"Invalid samples: {len(invalid_samples)}")
    print(f"Success rate: {(total_tested-len(invalid_samples))/total_tested*100:.2f}%")
    
    if problem_files:
        print(f"\nProblem scenes statistics:")
        for scene, files in problem_files.items():
            total_scene_files = scene_counts.get(scene, 0)
            error_rate = len(files) / total_scene_files * 100
            print(f"Scene {scene}: {len(files)}/{total_scene_files} files with errors ({error_rate:.2f}%)")
    
    # Save problem file list to log file
    log_file = os.path.join(os.path.dirname(args.outfile), 'problem_files.log')
    with open(log_file, 'w') as f:
        f.write("Problem Files Statistical Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Invalid samples: {len(invalid_samples)}\n\n")
        
        for scene, files in problem_files.items():
            f.write(f"\nScene: {scene}\n")
            f.write("-" * 30 + "\n")
            for file in files:
                f.write(f"{file}\n")
    
    print(f"\nDetailed problem file list saved to: {log_file}")
    
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
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])

    # Custom collate function to handle None values
    def custom_collate_fn(batch):
        # Filter out None values
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            raise ValueError("All samples in the batch are invalid")
        return torch.utils.data.dataloader.default_collate(batch)

    # Wrap datasets
    trainset = DatasetWrapper(trainset)
    testset = DatasetWrapper(testset)
    
    # Data loaders
    print(f"\n====== Data Loaders ======")
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=min(args.workers, 2),  # Reduce worker count
        drop_last=args.drop_last,
        pin_memory=(str(args.device) != 'cpu'),
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
    
    # Check first batch and measure loading time
    print("\n====== Testing Data Loading Performance ======")
    data_load_start = time.time()
    for data in trainloader:
        p0, p1, igt = data
        data_load_time = time.time() - data_load_start
        print(f"First batch loading time: {data_load_time:.4f} sec")
        print(f"First batch shapes: p0={p0.shape}, p1={p1.shape}, igt={igt.shape}")
        
        # Test batch processing time
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
            print(f"Forward pass time: {forward_time:.4f} sec")
            print(f"Estimated total time per batch: {data_load_time + forward_time:.4f} sec")
        break
    
    # Optimizer
    min_loss = float('inf')
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(learnable_params, lr=0.00001, weight_decay=1e-5)  # Higher learning rate
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=0.001, momentum=0.9, weight_decay=1e-5)

    if checkpoint is not None:
        min_loss = checkpoint['min_loss']
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Improved learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=3, verbose=True, 
        threshold=0.01, min_lr=1e-7)

    # Training
    print("\n====== Starting Training ======")
    LOGGER.debug('train, begin')
    
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
        scheduler.step(val_loss)
        
        epoch_time = time.time() - epoch_start
        
        is_best = val_loss < min_loss
        min_loss = min(val_loss, min_loss)

        print(f"[Time] Epoch {epoch+1}: {epoch_time:.2f} sec | Loss: {running_loss:.4f} | Validation: {val_loss:.4f}")
        
        LOGGER.info('epoch, %04d, %f, %f, %f, %f', epoch + 1, running_loss, val_loss, running_info, val_info)
        snap = {'epoch': epoch + 1,
                'model': model.state_dict(),
                'min_loss': min_loss,
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
        nan_batch_count = 0
        nan_loss_count = 0
        nan_grad_count = 0
        
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
            loss, loss_g = self.compute_loss(model, data, device)
            
            # Check if loss is NaN, if so skip this batch
            if not torch.isfinite(loss) or not torch.isfinite(loss_g):
                print(f"Warning: Batch {i} has non-finite loss values {loss.item() if torch.isfinite(loss) else 'NaN'}/{loss_g.item() if torch.isfinite(loss_g) else 'NaN'}, skipping")
                nan_loss_count += 1
                nan_batch_count += 1
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
            
            # Stronger gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            
            # Check if gradients contain NaN or Inf values
            do_step = True
            for param in model.parameters():
                if param.grad is not None:
                    if (1 - torch.isfinite(param.grad).long()).sum() > 0:
                        do_step = False
                        print(f"Warning: Batch {i} has non-finite gradient values, skipping parameter update")
                        nan_grad_count += 1
                        break
            
            # Only update parameters when gradients are normal
            if do_step:
                optimizer.step()
            else:
                # If gradients are abnormal, don't include in average loss
                nan_batch_count += 1
                batch_start = time.time()
                continue
                
            if str(device) != 'cpu':
                torch.cuda.synchronize()
            backward_time = time.time() - backward_start
            backward_times.append(backward_time)

            # Only normal batches count toward total loss and count
            vloss += loss.item()
            gloss += loss_g.item()
            count += 1
            
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
        if nan_batch_count > 0:
            print(f"\nWarning: {nan_batch_count} batches were skipped ({nan_batch_count/len(trainloader)*100:.2f}%)")
            print(f"- NaN loss batches: {nan_loss_count}")
            print(f"- NaN gradient batches: {nan_grad_count}")
            
        # Safely calculate averages
        ave_vloss = float(vloss)/count if count > 0 else float('inf')
        ave_gloss = float(gloss)/count if count > 0 else float('inf')
        
        # Calculate average times
        avg_batch = sum(batch_times) / len(batch_times) if batch_times else 0
        avg_data = sum(data_times) / len(data_times) if data_times else 0
        avg_forward = sum(forward_times) / len(forward_times) if forward_times else 0
        avg_backward = sum(backward_times) / len(backward_times) if backward_times else 0
        
        print(f"\nPerformance statistics:")
        print(f"Valid batches: {count}/{len(trainloader)} ({count/len(trainloader)*100:.2f}%)")
        print(f"Average batch time: {avg_batch:.4f} sec = Data loading: {avg_data:.4f} sec + Forward pass: {avg_forward:.4f} sec + Backward pass: {avg_backward:.4f} sec")
        print(f"Training results: Loss={ave_vloss:.4f}, Feature loss={ave_gloss:.4f}")
        
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
        p0, p1, igt = data
        p0 = p0.to(device) # template
        p1 = p1.to(device) # source
        igt = igt.to(device) # igt: p0 -> p1
        
        # Check input point clouds
        if not torch.isfinite(p0).all() or not torch.isfinite(p1).all():
            print("Warning: Input point clouds contain NaN or Inf values, attempting to fix")
            # Replace NaN and Inf with 0
            p0 = torch.nan_to_num(p0, nan=0.0, posinf=1.0, neginf=-1.0)
            p1 = torch.nan_to_num(p1, nan=0.0, posinf=1.0, neginf=-1.0)
        
        try:
            r = ptlk.pointlk.PointLK.do_forward(model, p0, p1, self.max_iter, self.xtol,
                                             self.p0_zero_mean, self.p1_zero_mean)
            est_g = model.g

            # Add numerical check
            if not torch.isfinite(est_g).all():
                print("Warning: Transformation matrix contains non-finite values")
                return torch.tensor(float('nan'), device=device), torch.tensor(float('nan'), device=device)
            
            loss_g = ptlk.pointlk.PointLK.comp(est_g, igt)
            
            # Add loss value limiting
            if loss_g > 1e5:
                print(f"Warning: Loss g is too large ({loss_g.item()}), limiting to 1e5")
                loss_g = torch.clamp(loss_g, max=1e5)

            if self._loss_type == 0:
                loss_r = ptlk.pointlk.PointLK.rsq(r)
                loss = loss_r
            elif self._loss_type == 1:
                loss_r = ptlk.pointlk.PointLK.rsq(r)
                # Set loss scale to prevent numerical issues
                loss = loss_r + loss_g * 0.1  # Reduce loss_g weight
            elif self._loss_type == 2:
                pr = model.prev_r
                if pr is not None:
                    loss_r = ptlk.pointlk.PointLK.rsq(r - pr)
                else:
                    loss_r = ptlk.pointlk.PointLK.rsq(r)
                loss = loss_r + loss_g * 0.1  # Reduce loss_g weight
            else:
                loss = loss_g
            
            # Final check
            if not torch.isfinite(loss):
                print(f"Warning: Final loss has non-finite value {loss}")
                return torch.tensor(1e5, device=device), torch.tensor(1e5, device=device)
            
            return loss, loss_g
        
        except Exception as e:
            print(f"Error computing loss: {e}")
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
        transform = torchvision.transforms.Compose([
            ptlk.data.transforms.OnUnitCube(),
            ptlk.data.transforms.Resampler(args.num_points),
        ])
        
        # Create C3VD dataset
        c3vd_dataset = ptlk.data.datasets.C3VDDataset(
            source_root=os.path.join(args.dataset_path, 'C3VD_ply_rot_scale_trans'),
            target_root=os.path.join(args.dataset_path, 'C3VD_ref'),
            transform=transform
        )
        
        # Split based on scene or randomly
        if args.scene_split:
            # Get all scenes
            all_scenes = []
            source_root = os.path.join(args.dataset_path, 'C3VD_ply_rot_scale_trans')
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
        
        # Create tracking datasets for training and testing
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