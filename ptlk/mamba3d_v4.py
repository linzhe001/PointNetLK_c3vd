""" 3DMamba-based Point Cloud Feature Extractor v4.0 (CBAM)
    专为点云配准和分类任务设计的基于Mamba模型的特征提取器
    
    使用 mamba-ssm 库的高性能CUDA算子，并集成了 CBAM (Convolutional Block Attention Module)
    来增强特征通道和空间位置的表达能力。

    设计考虑:
    1. 点云的无序性 - 通过位置编码和空间感知SSM处理
    2. 点云配准任务 - 需要捕获局部和全局几何特征
    3. 与PointNet_features兼容的接口
    4. Mamba模型的线性复杂度和长序列处理能力
    
    v4.0 更新:
    - 基于 v2.0，在MLP层后添加 CBAM_Layer，实现通道和空间双重注意力机制
    - 调整 Mamba3D_features 的输出以同时返回全局和局部特征
    - 调整 Mamba3D_classifier 以适应新输出格式
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


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return self.sigmoid(y)

class CBAM_Layer(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super(CBAM_Layer, self).__init__()
        self.channel_attention = ChannelAttention(channel, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        y = self.spatial_attention(x)
        return x * y


def mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    """ 创建多层感知机层，并集成CBAM
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
        if b_shared: # 只对共享权重的卷积层添加CBAM模块
            layers.append(CBAM_Layer(outp))
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
        
        self.mamba_layer = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=expand
        )
        
        if d_ff is None:
            d_ff = int(d_model * 2)
            
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.mamba_layer(x) 
        
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
    weights = torch.softmax(torch.sum(x, dim=-1), dim=-1)
    weights = weights.unsqueeze(-1)
    
    aggregated = torch.sum(x * weights, dim=1)
    return aggregated


class Mamba3D_features(torch.nn.Module):
    """基于3DMamba的点云特征提取器 - v4.0 (CBAM)
    
    输入: [B, N, 3] 点云
    输出: ([B, K], [B, N, K]) 全局特征向量和逐点特征
    """
    def __init__(self, dim_k=1024, sym_fn=symfn_max, scale=1, num_mamba_blocks=3, 
                 d_state=16, expand=2):
        super().__init__()
        
        self.d_model = max(64, int(128 / scale))
        self.dim_k = int(dim_k / scale)
        self.num_mamba_blocks = num_mamba_blocks
        
        self.input_projection = nn.Linear(3, self.d_model)
        
        self.pos_encoding = nn.Parameter(torch.randn(1, 2048, self.d_model) * 0.02)
        
        self.mamba_blocks = nn.ModuleList([
            Mamba3DBlock(
                d_model=self.d_model, 
                d_state=d_state,
                expand=expand
            )
            for _ in range(self.num_mamba_blocks)
        ])
        
        self.feature_transform = MLPNet(
            self.d_model, 
            [int(256/scale), self.dim_k], 
            b_shared=True
        )
        
        self.sy = sym_fn
        
        self.t_out_t2 = None
        self.t_out_h1 = None

    def forward(self, points):
        """
        前向传播
        
        Args:
            points: [B, N, 3] 输入点云
            
        Returns:
            (global_features, point_features): ([B, K], [B, N, K])
        """
        batch_size, num_points, _ = points.size()
        
        x = self.input_projection(points)
        
        if num_points <= self.pos_encoding.size(1):
            pos_encoding = self.pos_encoding[:, :num_points, :]
        else:
            pos_encoding = F.interpolate(
                self.pos_encoding.transpose(1, 2), 
                size=num_points, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        x = x + pos_encoding
        
        self.t_out_h1 = x.transpose(1, 2)
        
        for mamba_block in self.mamba_blocks:
            x = mamba_block(x)
        
        x = x.transpose(1, 2)
        x = self.feature_transform(x)
        
        x = x.transpose(1, 2)
        point_features = x
        
        if self.sy == symfn_max:
            global_features = symfn_max(point_features)
        elif self.sy == symfn_avg:
            global_features = symfn_avg(point_features)
        elif self.sy == symfn_selective:
            global_features = symfn_selective(point_features)
        else:
            global_features = symfn_max(point_features)
        
        return global_features, point_features


class Mamba3D_classifier(torch.nn.Module):
    """基于3DMamba的点云分类器
    """
    def __init__(self, num_c, mambafeat, dim_k):
        super().__init__()
        self.features = mambafeat
        
        list_layers = mlp_layers(dim_k, [512, 256], b_shared=False, bn_momentum=0.1, dropout=0.0)
        list_layers.append(torch.nn.Linear(256, num_c))
        self.classifier = torch.nn.Sequential(*list_layers)

    def forward(self, points):
        global_feat, _ = self.features(points)
        out = self.classifier(global_feat)
        return out

    def loss(self, out, target, w=0.001):
        loss_c = torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(out, dim=1), target, size_average=False)

        t2 = self.features.t_out_t2
        if (t2 is None) or (w == 0):
            return loss_c

        batch = t2.size(0)
        K = t2.size(1)
        I = torch.eye(K).repeat(batch, 1, 1).to(t2)
        A = t2.bmm(t2.transpose(1, 2))
        loss_m = torch.nn.functional.mse_loss(A, I, size_average=False)
        loss = loss_c + w * loss_m
        return loss

#EOF 