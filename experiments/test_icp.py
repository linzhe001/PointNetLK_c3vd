"""
    Example for testing ICP.

    No-noise version.

    This ICP-test is very slow. use faster ones like Matlab or C++...
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
import glob
import math
import shutil
import traceback

# addpath('../')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import ptlk

import icp

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())

def options(argv=None):
    parser = argparse.ArgumentParser(description='ICP')

    # required.
    parser.add_argument('-o', '--outfile', required=True, type=str,
                        metavar='FILENAME', help='output filename (.csv)')
    parser.add_argument('-i', '--dataset-path', required=True, type=str,
                        metavar='PATH', help='path to the input dataset')
    parser.add_argument('-c', '--categoryfile', required=True, type=str,
                        metavar='PATH', help='path to the categories to be tested')
    parser.add_argument('-p', '--perturbations', required=False, type=str,
                        metavar='PATH', help='path to the perturbations')

    # settings for input data
    parser.add_argument('--dataset-type', default='modelnet', choices=['modelnet', 'c3vd'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--format', default='wv', choices=['wv', 'wt'],
                        help='perturbation format (default: wv (twist)) (wt: rotation and translation)') # the output is always in twist format
    parser.add_argument('--num-points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')
    # C3VD 配对模式设置
    parser.add_argument('--pair-mode', default='one_to_one', choices=['one_to_one', 'scene_reference'],
                        help='点云配对模式: one_to_one (每个源点云对应特定目标点云) 或 scene_reference (每个场景使用一个共享目标点云)')
    parser.add_argument('--reference-name', default=None, type=str,
                        help='场景参考模式下使用的目标点云名称，默认使用场景中的第一个点云')
    parser.add_argument('--max-samples', default=2000, type=int,
                        metavar='N', help='最大测试样本数 (default: 2000)')

    # settings for ICP
    parser.add_argument('--max-iter', default=20, type=int,
                        metavar='N', help='max-iter on ICP. (default: 20)')

    # settings for on testing
    parser.add_argument('-l', '--logfile', default='', type=str,
                        metavar='LOGNAME', help='path to logfile (default: null (no logging))')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--device', default='cpu', type=str,
                        metavar='DEVICE', help='use CUDA if available (default: cpu)')
    
    # 新增参数：支持扰动文件夹批量处理
    parser.add_argument('--perturbation-dir', default=None, type=str,
                        metavar='PATH', help='扰动文件夹路径，将处理文件夹中的所有扰动文件')
    
    # 新增参数：联合归一化设置（用于C3VD数据集）
    parser.add_argument('--use-joint-normalization', action='store_true',
                        help='使用联合边界框对配对点云进行归一化，保持相对空间关系')

    args = parser.parse_args(argv)
    return args

def main(args):
    # 创建空列表存储所有要处理的扰动文件
    perturbation_files = []
    
    # 如果指定了扰动文件夹，先添加文件夹中的所有.csv文件
    if args.perturbation_dir and os.path.exists(args.perturbation_dir):
        print(f"\n====== 扰动文件夹 ======")
        print(f"扫描扰动文件夹: {args.perturbation_dir}")
        for filename in sorted(os.listdir(args.perturbation_dir)):
            if filename.endswith('.csv'):
                full_path = os.path.join(args.perturbation_dir, filename)
                perturbation_files.append(full_path)
                print(f"发现扰动文件: {filename}")
    
    # 如果还指定了单独的扰动文件，也添加进列表
    if args.perturbations and os.path.exists(args.perturbations):
        if args.perturbations not in perturbation_files:
            perturbation_files.append(args.perturbations)
            print(f"添加单独指定的扰动文件: {os.path.basename(args.perturbations)}")
    
    # 检查是否有扰动文件要处理
    if not perturbation_files:
        print("错误: 没有找到扰动文件。请使用 --perturbation-dir 指定扰动文件夹或使用 --perturbations 指定扰动文件。")
        return
    
    print(f"总共发现 {len(perturbation_files)} 个扰动文件需要处理")
    
    # 创建动作执行器，但不立即传入扰动文件
    act = Action(args)
    
    # 依次处理每个扰动文件
    for i, pert_file in enumerate(perturbation_files):
        filename = os.path.basename(pert_file)
        print(f"\n====== 处理扰动文件 [{i+1}/{len(perturbation_files)}]: {filename} ======")
        
        # 保存原始参数值
        original_perturbations = args.perturbations
        original_outfile = args.outfile
        original_logfile = args.logfile
        
        # 提取扰动角度信息（如果有）
        angle_str = ""
        if filename.startswith("pert_") and "_" in filename:
            parts = filename.split("_")
            if len(parts) >= 2:
                angle_str = parts[1].split(".")[0]
        
        # 为当前扰动文件创建输出文件名
        if angle_str:
            # 为每个角度创建单独目录
            output_dir = os.path.join(os.path.dirname(args.outfile), f"angle_{angle_str}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 保持与原始CSV文件相同的命名结构，只更改扩展名为.log
            base_filename = os.path.splitext(filename)[0]
            current_outfile = os.path.join(output_dir, f"icp_results_{base_filename}.log")
            log_file = os.path.join(output_dir, f"icp_log_{base_filename}.log")
        else:
            # 对于没有角度信息的扰动文件，创建基本目录
            output_dir = os.path.dirname(args.outfile)
            os.makedirs(output_dir, exist_ok=True)
            
            # 使用扰动文件名作为输出文件名
            base_filename = os.path.splitext(filename)[0]
            current_outfile = os.path.join(output_dir, f"icp_results_{base_filename}.log")
            log_file = os.path.join(output_dir, f"icp_log_{base_filename}.log")
        
        # 设置当前参数值
        args.perturbations = pert_file
        args.outfile = current_outfile
        args.logfile = log_file
        
        print(f"输出文件: {args.outfile}")
        print(f"日志文件: {args.logfile}")
        
        # 为当前扰动文件获取数据集
        testset = get_datasets(args)
        
        # 更新动作执行器的文件名
        act.update_perturbation(args.perturbations, current_outfile)
        
        # 运行测试
        run(args, testset, act)
        
        # 恢复原始参数
        args.perturbations = original_perturbations
        args.outfile = original_outfile
        args.logfile = original_logfile
        
        # 清理内存
        del testset
        torch.cuda.empty_cache()
        import gc
        gc.collect()

def run(args, testset, action):
    # Custom dataset wrapper that handles exceptions
    class DatasetWrapper(torch.utils.data.Dataset):
        """Wrapper for safely loading dataset samples that might cause exceptions."""
        def __init__(self, dataset):
            self.dataset = dataset
            self.valid_indices = []
            self.num_points = args.num_points  # 保存num_points参数
            self.test_all_indices()
            
        def test_all_indices(self):
            # 预先检查所有样本，找出有效的索引
            print("预检查数据集样本有效性...")
            for i in range(len(self.dataset)):
                try:
                    _ = self.dataset[i]
                    self.valid_indices.append(i)
                except Exception as e:
                    print(f"警告: 样本 {i} 无效: {str(e)}")
            
            print(f"数据集有效性检查完成: {len(self.valid_indices)}/{len(self.dataset)} 有效")
            
        def __len__(self):
            return len(self.valid_indices)

        def __getitem__(self, idx):
            try:
                return self.dataset[self.valid_indices[idx]]
            except Exception as e:
                print(f"警告: 无法加载样本 {self.valid_indices[idx]}: {str(e)}")
                # 返回最后一个已知有效的样本
                if idx > 0:
                    return self.dataset[self.valid_indices[idx-1]]
                # 如果没有有效样本，则创建一个空点云
                else:
                    # 创建一个随机点云和标签作为替代
                    import numpy as np
                    random_cloud = torch.from_numpy(np.random.rand(self.num_points, 3).astype(np.float32))
                    return random_cloud, random_cloud

    LOGGER.debug('Testing (PID=%d), %s', os.getpid(), args)

    sys.setrecursionlimit(20000)

    # 设备检查
    print(f"\n====== 设备配置 ======")
    device = torch.device(args.device)
    print(f"使用设备: {device}")

    # 包装数据集以处理异常
    if hasattr(testset, '__len__'):
        print(f"\n====== 数据集准备 ======")
        print(f"原始数据集大小: {len(testset)}")
        testset = DatasetWrapper(testset)
        print(f"经过筛选的数据集大小: {len(testset)}")

    # 自定义collate函数，处理None值
    def custom_collate_fn(batch):
        # 过滤掉None值
        batch = list(filter(lambda x: x is not None, batch))
        if len(batch) == 0:
            raise ValueError("批次中所有样本都无效")
        return torch.utils.data.dataloader.default_collate(batch)

    # 如果数据集为空，则不继续执行
    if len(testset) == 0:
        print("错误: 数据集中没有有效样本，无法继续测试。")
        return

    # dataloader
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=1, shuffle=False, 
        num_workers=min(args.workers, 1),  # 减少worker数量，降低出错率
        collate_fn=custom_collate_fn)
    
    print(f"\n====== 数据集信息 ======")
    print(f"测试样本数: {len(testset)}")
    if hasattr(args, 'num_points'):
        print(f"每个点云的点数: {args.num_points}")
    print(f"批次大小: 1")

    # testing
    print(f"\n====== 开始测试 ======")
    LOGGER.debug('tests, begin')

    # 创建结果目录
    os.makedirs(os.path.dirname(action.filename), exist_ok=True)

    # 执行测试
    action.eval_1(testloader)
    LOGGER.debug('tests, end')


class Action:
    def __init__(self, args):
        self.filename = args.outfile
        # ICP
        self.max_iter = args.max_iter
        # 当前扰动文件名（从参数中提取）
        if args.perturbations:
            self.current_pert_file = os.path.basename(args.perturbations)
        else:
            self.current_pert_file = "unknown"

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

    def eval_1(self, testloader):
        # 添加误差统计变量
        total_rot_error = 0.0
        total_trans_error = 0.0
        total_total_error = 0.0
        success_count = 0
        total_count = 0
        error_count = 0
        all_rot_errors = []
        all_trans_errors = []
        all_total_errors = []

        # 确保输出目录存在
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        with open(self.filename, 'w') as fout:
            # 写入文件头部信息
            # Write file header information
            print(f"# ICP Registration Test Results", file=fout)
            print(f"# Perturbation file: {self.current_pert_file}", file=fout)
            print(f"# Test time: {time.strftime('%Y-%m-%d %H:%M:%S')}", file=fout)
            print(f"# Rotation error unit: radians", file=fout)
            print(f"# Translation error unit: meters", file=fout)
            print(f"# =====================================", file=fout)
            
            self.eval_1__header(fout)
            with torch.no_grad():
                # 计算总共需要测试的样本数
                total_samples = len(testloader)
                start_time = time.time()
                
                for i, data in enumerate(testloader):
                    batch_start_time = time.time()
                    total_count += 1
                    
                    try:
                        p0, p1, igt = data
                        
                        # 检查点云有效性
                        if not torch.isfinite(p0).all() or not torch.isfinite(p1).all():
                            print(f"警告: 批次 {i} 包含NaN值，尝试修复")
                            p0 = torch.nan_to_num(p0, nan=0.0, posinf=1.0, neginf=-1.0)
                            p1 = torch.nan_to_num(p1, nan=0.0, posinf=1.0, neginf=-1.0)
                        
                        # 获取样本信息（如果可用）
                        sample_info = {}
                        if hasattr(testloader.dataset, 'get_cloud_info'):
                            cloud_info = testloader.dataset.get_cloud_info(i)
                            if cloud_info:
                                sample_info = cloud_info
                        
                        # 如果没有获取到信息，创建默认信息
                        if not sample_info:
                            sample_info = {
                                'identifier': f"sample_{i:04d}",
                                'scene': 'unknown',
                                'sequence': f'{i:04d}'
                            }
                        
                        # 执行配准
                        res = self.do_estimate(p0, p1) # --> [1, 4, 4]
                        ig_gt = igt.cpu().contiguous().view(-1, 4, 4) # --> [1, 4, 4]
                        g_hat = res.cpu().contiguous().view(-1, 4, 4) # --> [1, 4, 4]

                        # 计算配准误差
                        dg = torch.matmul(g_hat, ig_gt) # if correct, dg == identity matrix.
                        dx = ptlk.se3.log(dg) # --> [1, 6] (if corerct, dx == zero vector)
                        
                        # 分别计算旋转误差和平移误差
                        rot_error = dx[:, :3]  # 旋转误差 [1, 3]
                        trans_error = dx[:, 3:]  # 平移误差 [1, 3]
                        
                        rot_norm = rot_error.norm(p=2, dim=1)  # 旋转误差L2范数
                        trans_norm = trans_error.norm(p=2, dim=1)  # 平移误差L2范数
                        total_norm = dx.norm(p=2, dim=1)  # 总误差L2范数
                        
                        # 累加误差统计
                        total_rot_error += rot_norm.item()
                        total_trans_error += trans_norm.item()
                        total_total_error += total_norm.item()
                        all_rot_errors.append(rot_norm.item())
                        all_trans_errors.append(trans_norm.item())
                        all_total_errors.append(total_norm.item())
                        
                        # 计算总误差
                        dn = dx.norm(p=2, dim=1) # --> [1]
                        dm = dn.mean()

                        # 写入结果
                        self.eval_1__write(fout, ig_gt, g_hat, sample_info)
                        success_count += 1
                        
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
                        
                        # 计算角度表示的旋转误差（弧度转角度）
                        rot_deg = rot_norm.item() * (180.0 / math.pi)
                        
                        # 打印进度
                        print(f"测试: [{i+1}/{total_samples}] {(i+1)/total_samples*100:.1f}% | "
                              f"旋转误差: {rot_norm.item():.6f}弧度 ({rot_deg:.2f}度), 平移误差: {trans_norm.item():.6f} | "
                              f"总误差: {total_norm.item():.6f} {error_level} | "
                              f"耗时: {batch_time:.2f}秒 | "
                              f"剩余: {remaining_time/60:.1f}分钟")
                        
                        LOGGER.info('test, %d/%d, rot_error: %f, trans_error: %f, total_error: %f', 
                                   i, total_samples, rot_norm.item(), trans_norm.item(), total_norm.item())
                    
                    except Exception as e:
                        print(f"错误: 处理批次 {i} 时出错: {str(e)}")
                        # 记录错误详情到日志
                        LOGGER.error('Error in batch %d: %s', i, str(e), exc_info=True)
                        # 写入无效结果
                        dummy_vals = ['nan'] * 6
                        print(','.join(dummy_vals), file=fout)
                        fout.flush()
        
        # 结果统计
        total_time = time.time() - start_time
        
        # 添加统计信息到输出文件
        with open(self.filename, 'a') as fout:
            print(f"", file=fout)
            print(f"# =====================================", file=fout)
            print(f"# Test Statistics", file=fout)
            print(f"# =====================================", file=fout)
            print(f"# Total time: {total_time:.2f} seconds", file=fout)
            print(f"# Successfully processed samples: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)", file=fout)
            
            # 计算统计指标
            if success_count > 0:
                avg_rot_error = total_rot_error / success_count
                avg_trans_error = total_trans_error / success_count
                avg_total_error = total_total_error / success_count
                
                # 计算中位数误差
                all_rot_errors.sort()
                all_trans_errors.sort()
                all_total_errors.sort()
                
                median_rot_error = all_rot_errors[len(all_rot_errors)//2] if all_rot_errors else 0
                median_trans_error = all_trans_errors[len(all_trans_errors)//2] if all_trans_errors else 0
                median_total_error = all_total_errors[len(all_total_errors)//2] if all_total_errors else 0
                
                # 计算标准差
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
        
        print(f"\n====== 测试完成 ======")
        print(f"总耗时: {total_time:.2f}秒 (平均每样本 {total_time/total_count:.2f}秒)")
        print(f"成功处理样本: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        # 计算平均误差
        if success_count > 0:
            avg_rot_error = total_rot_error / success_count
            avg_trans_error = total_trans_error / success_count
            # 将旋转误差从弧度转为角度
            avg_rot_error_deg = avg_rot_error * (180.0 / math.pi)
            print(f"平均旋转误差: {avg_rot_error:.6f}弧度 ({avg_rot_error_deg:.2f}度)")
            print(f"平均平移误差: {avg_trans_error:.6f}")
            
        if error_count > 0:
            print(f"配准失败样本: {error_count}/{success_count} ({error_count/success_count*100:.1f}%)")
        print(f"结果已保存至: {self.filename}")

        return success_count, total_count

    def update_perturbation(self, perturbation_file, outfile):
        """更新当前处理的扰动文件和输出文件"""
        self.filename = outfile
        self.current_pert_file = os.path.basename(perturbation_file)

    def do_estimate(self, p0, p1):
        # 确保提取正确形状的点云数据
        np_p0 = p0.cpu().detach().squeeze(0).numpy()
        np_p1 = p1.cpu().detach().squeeze(0).numpy()
        
        # 确保点云形状正确为[num_points, 3]
        if np_p0.shape[1] != 3 and np_p0.shape[0] == 3:
            np_p0 = np_p0.T
        if np_p1.shape[1] != 3 and np_p1.shape[0] == 3:
            np_p1 = np_p1.T
        
        # 创建ICP实例并运行
        mod = icp.ICP(np_p0, np_p1)
        g, p, itr = mod.compute(self.max_iter)
        
        # 转换结果回PyTorch张量
        est_g = torch.from_numpy(g).float().view(1, 4, 4).to(p0.device)
        return est_g


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
    if args.perturbations:
        if not os.path.exists(args.perturbations):
            raise FileNotFoundError(f"{args.perturbations} not found.")
        perturbations = numpy.loadtxt(args.perturbations, delimiter=',')
    if args.format == 'wt':
        fmt_trans = True

    if args.dataset_type == 'modelnet':
        transform = torchvision.transforms.Compose([\
                ptlk.data.transforms.Mesh2Points(),\
                ptlk.data.transforms.OnUnitCube(),\
            ])

        testdata = ptlk.data.datasets.ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo)

        testset = ptlk.data.datasets.CADset4tracking_fixed_perturbation(testdata,\
                        perturbations, fmt_trans=fmt_trans)
    
    elif args.dataset_type == 'c3vd':
        # 修改：移除重采样，只保留基础变换
        # Modified: Remove resampling, keep only basic transformations
        transform = torchvision.transforms.Compose([
            # 不再包含任何点云处理，将在C3VDset4tracking中处理
            # No longer includes any point cloud processing, will be handled in C3VDset4tracking
        ])
        
        # 如果使用联合归一化，则在测试数据集类中处理
        # If using joint normalization, handle in test dataset class
        if args.use_joint_normalization:
            print(f"\n====== 联合归一化设置 ======")
            print(f"启用联合边界框归一化: 使用共同边界框对配对点云进行归一化")
            print(f"点云重采样: 联合归一化后重采样到 {args.num_points} 个点")
        
        # 创建C3VD数据集 - 配对模式支持
        # Create C3VD dataset - pairing mode support
        # 打印配对模式信息
        # Print pairing mode information
        print(f"\n====== C3VD数据集配置 ======")
        print(f"配对模式: {args.pair_mode}")
        
        # 设置源点云路径
        # Set source point cloud path
        source_root = os.path.join(args.dataset_path, 'C3VD_ply_source')
        
        # 根据配对模式设置目标点云路径
        # Set target point cloud path according to pairing mode
        if args.pair_mode == 'scene_reference':
            if args.reference_name:
                print(f"参考点云名称: {args.reference_name}")
            else:
                print(f"参考点云: 每个场景的第一个点云")
            target_path = os.path.join(args.dataset_path, 'C3VD_ref')
            print(f"目标点云目录: {target_path}")
        else:  # one_to_one 模式 # one_to_one mode
            target_path = os.path.join(args.dataset_path, 'visible_point_cloud_ply_depth')
            print(f"目标点云目录: {target_path}")
            print(f"配对方式: 每个源点云匹配对应帧号的目标点云")
        
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
            print(f"错误: 没有找到任何配对点云，请检查配对模式和数据路径")
            # 输出详细的调试信息
            # Output detailed debug information
            source_root = os.path.join(args.dataset_path, 'C3VD_ply_source')
            print(f"源点云目录: {source_root}")
            print(f"目录是否存在: {os.path.exists(source_root)}")
            if os.path.exists(source_root):
                scenes = [d for d in os.listdir(source_root) if os.path.isdir(os.path.join(source_root, d))]
                print(f"发现场景: {scenes}")
                if scenes:
                    for scene in scenes[:2]:  # 只显示前两个场景的信息 # Only show info for first two scenes
                        scene_dir = os.path.join(source_root, scene)
                        files = os.listdir(scene_dir)
                        print(f"场景 {scene} 中的文件: {files[:5]}{'...' if len(files) > 5 else ''}")
            
            target_root = target_path
            print(f"目标点云目录: {target_root}")
            print(f"目录是否存在: {os.path.exists(target_root)}")
            if os.path.exists(target_root):
                scenes = [d for d in os.listdir(target_root) if os.path.isdir(os.path.join(target_root, d))]
                print(f"发现场景: {scenes}")
                if scenes:
                    for scene in scenes[:2]:  # 只显示前两个场景的信息 # Only show info for first two scenes
                        scene_dir = os.path.join(target_root, scene)
                        files = os.listdir(scene_dir)
                        print(f"场景 {scene} 中的文件: {files[:5]}{'...' if len(files) > 5 else ''}")
            
            raise RuntimeError("找不到配对点云，请检查数据集结构和配对模式设置")
        
        # 不进行数据分割，直接使用整个数据集作为测试集
        # Don't split data, use entire dataset as test set
        testdata = c3vd_dataset
        
        # 创建一个自定义的RandomTransformSE3类，使用固定的扰动而不是随机生成
        # Create a custom RandomTransformSE3 class that uses fixed perturbations instead of random generation
        class FixedTransformSE3(ptlk.data.transforms.RandomTransformSE3):
            def __init__(self, perturbations, fmt_trans=False):
                super().__init__(0.8, False)  # 随机参数在这里不重要，我们会覆盖__call__ # Random parameters don't matter here, we will override __call__
                self.perturbations = perturbations
                self.fmt_trans = fmt_trans
                self.igt = None  # 初始化igt属性 # Initialize igt attribute
                self.idx = 0  # 当前使用的扰动索引 # Current perturbation index being used
                
            def __call__(self, points):
                # 从扰动文件获取一个扰动向量，每次调用使用下一个扰动
                # Get a perturbation vector from perturbation file, use next perturbation each call
                if self.idx >= len(self.perturbations):
                    self.idx = 0  # 如果已经用完所有扰动，重新从第一个开始 # If all perturbations used, start from first again
                
                # 获取当前扰动
                # Get current perturbation
                twist = torch.from_numpy(self.perturbations[self.idx]).contiguous().view(1, 6)
                self.idx += 1  # 更新索引 # Update index
                
                x = twist.to(points)
                
                if not self.fmt_trans:
                    # 按照扭曲向量方式处理
                    # Process according to twist vector method
                    g = ptlk.se3.exp(x).to(points)  # [1, 4, 4]
                    p1 = ptlk.se3.transform(g, points)
                    self.igt = g.squeeze(0)  # igt: points -> p1
                else:
                    # 按照旋转和平移方式处理
                    # Process according to rotation and translation method
                    w = x[:, 0:3]
                    q = x[:, 3:6]
                    g = torch.zeros(1, 4, 4).to(points)
                    g[:, 3, 3] = 1
                    g[:, 0:3, 0:3] = ptlk.so3.exp(w).to(points)  # 旋转 # Rotation
                    g[:, 0:3, 3] = q  # 平移 # Translation
                    p1 = ptlk.se3.transform(g, points)
                    self.igt = g.squeeze(0)  # igt: points -> p1
                    
                return p1
        
        # 创建固定变换
        # Create fixed transformation
        fixed_transform = FixedTransformSE3(perturbations, fmt_trans)
        
        # 使用我们新创建的测试专用数据集
        # Use our newly created test-specific dataset
        testset = ptlk.data.datasets.C3VDset4tracking_test(
            testdata, 
            fixed_transform, 
            use_joint_normalization=args.use_joint_normalization,
            num_points=args.num_points)  # 传递点数参数 # Pass point number parameter
        
        # 打印数据集信息
        # Print dataset information
        print(f"C3VD数据集总大小: {len(c3vd_dataset)}")
        print(f"测试集大小: {len(testset)}")
        
        # 查看部分样本信息
        # View partial sample information
        print("\n样本配对信息示例:")
        for i in range(min(3, len(c3vd_dataset.pairs))):
            source_file, target_file = c3vd_dataset.pairs[i]
            source_basename = os.path.basename(source_file)
            target_basename = os.path.basename(target_file)
            print(f"样本 {i}: 源点云={source_basename}, 目标点云={target_basename}")
        
        # 随机选择指定数量的样本进行测试
        # Randomly select specified number of samples for testing
        max_samples = args.max_samples
        if len(testset) > max_samples:
            print(f"数据集太大，随机选择{max_samples}个样本进行测试...")
            # 设置随机种子以确保可复现性
            # Set random seed to ensure reproducibility
            torch.manual_seed(42)
            # 获取随机索引
            # Get random indices
            indices = torch.randperm(len(testset))[:max_samples].tolist()
            # 创建子集
            # Create subset
            testset = torch.utils.data.Subset(testset, indices)
            print(f"采样后的测试集大小: {len(testset)}")

    return testset


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