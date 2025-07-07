""" 3DMamba-based Point Cloud Feature Extractor v2.0
    专为点云配准和分类任务设计的基于Mamba模型的特征提取器
    
    使用 mamba-ssm 库的高性能CUDA算子替换了自定义的S6Layer实现，
    显著提升了计算效率。

    设计考虑:
    1. 点云的无序性 - 通过位置编码和空间感知SSM处理
    2. 点云配准任务 - 需要捕获局部和全局几何特征
    3. 与PointNet_features兼容的接口
    4. Mamba模型的线性复杂度和长序列处理能力
    
    v2.0 更新:
    - 引入 mamba-ssm 库替换自定义 S6Layer
    - 移除 S6Layer, _efficient_scan, _process_chunk 等底层实现
    - 简化 Mamba3DBlock，直接调用高性能 Mamba 层
    - 提升训练和推理速度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mamba_ssm import Mamba # [新增] 导入 Mamba 核心库


def flatten(x):
    return x.view(x.size(0), -1)


class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return flatten(x)


def mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    """ 创建多层感知机层
        [B, Cin, N] -> [B, Cout, N] 或 [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            weights = torch.nn.Linear(last, outp)
        layers.append(weights)
        layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(torch.nn.ReLU())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers


class MLPNet(torch.nn.Module):
    """ 多层感知机网络，用于mamba模块中的特征变换
        支持共享权重(Conv1d)和非共享权重(Linear)两种模式
    """
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        self.b_shared = b_shared
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        return self.layers(inp)


class Mamba3DBlock(nn.Module):
    """
    3DMamba块 - [高性能版本]
    使用 mamba-ssm 的 CUDA-accelerated Mamba layer
    """
    def __init__(self, d_model, d_state=16, d_ff=None, expand=2, dropout=0.1):
        super().__init__()
        
        # [改动] 1. 使用 mamba_ssm.Mamba 替换 S6Layer
        # 我们直接将参数传递给 Mamba 类。
        # d_conv=4 是一个常用的默认值，代表 Mamba 内部卷积核大小。
        self.mamba_layer = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=expand
        )
        
        if d_ff is None:
            d_ff = int(d_model * 2)
            
        # [保留] 前馈网络部分保持不变
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # [保留] 归一化层也保留
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        前向传播遵循 Mamba 论文中典型的 Block 结构:
        Input -> Mamba -> Norm -> FFN -> Residual
        The Mamba layer itself contains a residual connection.
        """
        # 1. 通过 Mamba 层
        # mamba_ssm.Mamba 自身就是一个完整的块，包含内部的残差连接
        x = self.mamba_layer(x) 
        
        # 2. 应用前馈网络 (FFN) 和残差连接
        # 这是标准的 Transformer Block 结构: Output = Input + FFN(Norm(Input))
        x_norm = self.norm(x)
        ffn_out = self.feed_forward(x_norm)
        x = x + ffn_out
        
        return x


def symfn_max(x):
    """最大池化聚合函数 [B, N, K] -> [B, K]"""
    return torch.max(x, dim=1)[0]


def symfn_avg(x):
    """平均池化聚合函数 [B, N, K] -> [B, K]"""
    return torch.mean(x, dim=1)


def symfn_selective(x):
    """基于选择性聚合的函数 [B, N, K] -> [B, K]"""
    # 使用softmax得到每个点的权重
    weights = torch.softmax(torch.sum(x, dim=-1), dim=-1)  # [B, N]
    weights = weights.unsqueeze(-1)  # [B, N, 1]
    
    # 加权求和
    aggregated = torch.sum(x * weights, dim=1)  # [B, K]
    return aggregated


class Mamba3D_features(torch.nn.Module):
    """基于3DMamba的点云特征提取器 - v2.0 高性能版本
    
    替换PointNet_features，专为点云配准和分类任务设计
    输入: [B, N, 3] 点云
    输出: [B, K] 全局特征向量
    
    优化特点：
    1. 使用 mamba-ssm 加速，大幅提升性能
    2. 使用更高效的位置编码
    3. 保持与PointNet_features完全API兼容性
    """
    def __init__(self, dim_k=1024, sym_fn=symfn_max, scale=1, num_mamba_blocks=3, 
                 d_state=16, expand=2):
        super().__init__()
        
        # 特征维度设置
        self.d_model = max(64, int(128 / scale))
        self.dim_k = int(dim_k / scale)
        self.num_mamba_blocks = num_mamba_blocks
        
        # 输入嵌入层：将3D坐标映射到高维特征空间
        self.input_projection = nn.Linear(3, self.d_model)
        
        # 3D位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, self.d_model) * 0.02)
        
        # [改动] 创建高性能版本的 Mamba3DBlock
        # 参数可以直接传递，不再需要手动缩减
        self.mamba_blocks = nn.ModuleList([
            Mamba3DBlock(
                d_model=self.d_model, 
                d_state=d_state,
                expand=expand
            )
            for _ in range(self.num_mamba_blocks)
        ])
        
        # 特征变换层：从mamba维度映射到最终特征维度
        self.feature_transform = MLPNet(
            self.d_model, 
            [int(256/scale), self.dim_k], 
            b_shared=True  # 使用Conv1d层以支持[B, d_model, N]格式输入
        )
        
        # 聚合函数
        self.sy = sym_fn
        
        # 保持与PointNet_features兼容的属性
        self.t_out_t2 = None
        self.t_out_h1 = None

    def forward(self, points):
        """
        前向传播
        
        Args:
            points: [B, N, 3] 输入点云
            
        Returns:
            features: [B, K] 全局特征向量
        """
        batch_size, num_points, _ = points.size()
        
        # 输入投影：3D坐标 -> 高维特征
        x = self.input_projection(points)  # [B, N, d_model]
        
        # 添加位置编码
        if num_points <= self.pos_encoding.size(1):
            pos_encoding = self.pos_encoding[:, :num_points, :]
        else:
            # 对于更长的序列，使用线性插值
            pos_encoding = F.interpolate(
                self.pos_encoding.transpose(1, 2), 
                size=num_points, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        x = x + pos_encoding
        
        # 保存中间特征（兼容性）
        self.t_out_h1 = x.transpose(1, 2)  # [B, d_model, N] 格式用于兼容
        
        # 通过多层Mamba块
        for mamba_block in self.mamba_blocks:
            x = mamba_block(x)  # [B, N, d_model]
        
        # 特征变换
        # 转换为 [B, d_model, N] 格式用于Conv1d处理
        x = x.transpose(1, 2)  # [B, d_model, N]
        x = self.feature_transform(x)  # [B, dim_k, N]
        
        # 转回 [B, N, dim_k] 格式进行聚合
        x = x.transpose(1, 2)  # [B, N, dim_k]
        
        # 全局聚合
        if self.sy == symfn_max:
            global_features = symfn_max(x)  # [B, dim_k]
        elif self.sy == symfn_avg:
            global_features = symfn_avg(x)  # [B, dim_k]
        elif self.sy == symfn_selective:
            global_features = symfn_selective(x)  # [B, dim_k]
        else:
            # 默认使用最大池化
            global_features = symfn_max(x)  # [B, dim_k]
        
        return global_features


class Mamba3D_classifier(torch.nn.Module):
    """基于3DMamba的点云分类器
    
    类似于PointNet_classifier，使用Mamba3D_features作为特征提取器
    """
    def __init__(self, num_c, mambafeat, dim_k):
        """
        Args:
            num_c: 分类数量
            mambafeat: Mamba3D_features实例
            dim_k: 特征维度
        """
        super().__init__()
        self.features = mambafeat
        
        # 分类头：特征向量 -> 分类结果
        list_layers = mlp_layers(dim_k, [512, 256], b_shared=False, bn_momentum=0.1, dropout=0.0)
        list_layers.append(torch.nn.Linear(256, num_c))
        self.classifier = torch.nn.Sequential(*list_layers)

    def forward(self, points):
        """
        前向传播
        
        Args:
            points: [B, N, 3] 输入点云
            
        Returns:
            out: [B, num_c] 分类输出
        """
        feat = self.features(points)  # [B, dim_k]
        out = self.classifier(feat)   # [B, num_c]
        return out

    def loss(self, out, target, w=0.001):
        """
        计算损失函数
        
        Args:
            out: [B, num_c] 分类输出
            target: [B] 真实标签
            w: 正则化权重
            
        Returns:
            loss: 总损失
        """
        # 分类损失
        loss_c = torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(out, dim=1), target, size_average=False)

        # 注意：对于mamba模型，我们目前不添加变换矩阵正则化
        # 因为mamba机制的结构与PointNet不同，没有对应的t_out_t2
        # 如果需要，可以添加其他形式的正则化
        t2 = self.features.t_out_t2
        if (t2 is None) or (w == 0):
            return loss_c

        # 如果存在变换矩阵，添加正则化项
        batch = t2.size(0)
        K = t2.size(1)  # [B, K, K]
        I = torch.eye(K).repeat(batch, 1, 1).to(t2)
        A = t2.bmm(t2.transpose(1, 2))
        loss_m = torch.nn.functional.mse_loss(A, I, size_average=False)
        loss = loss_c + w * loss_m
        return loss

#EOF 