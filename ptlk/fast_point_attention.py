""" Fast Point Attention v1.0 - 轻量级点云注意力特征提取器
    专为高效点云配准和分类任务设计的轻量级注意力机制
    
    设计理念:
    1. 计算高效 - 使用简化的注意力机制减少计算复杂度
    2. 内存友好 - 优化内存使用，支持大规模点云处理
    3. 保持精度 - 在效率和精度之间找到最佳平衡点
    4. API兼容 - 与PointNet_features完全兼容的接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
    """ 多层感知机网络，用于特征变换 """
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        self.b_shared = b_shared
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        return self.layers(inp)


class SimplifiedPositionalEncoding(nn.Module):
    """简化的3D位置编码，减少计算开销"""
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        # 使用更简单的位置编码：仅使用一个线性层
        self.pos_projection = nn.Linear(3, d_model // 4)  # 减少位置编码维度
        
    def forward(self, points):
        """
        Args:
            points: [B, N, 3] 点云坐标
        Returns:
            pos_encoding: [B, N, d_model//4] 简化位置编码
        """
        pos_encoding = self.pos_projection(points)  # [B, N, d_model//4]
        return pos_encoding


class FastAttention(nn.Module):
    """快速注意力机制 - 使用简化的单头注意力"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.scale = 1.0 / math.sqrt(d_model)
        
        # 使用单头注意力减少计算量
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: [B, N, d_model] 输入特征
        Returns:
            output: [B, N, d_model] 注意力输出
        """
        batch_size, seq_len, d_model = x.size()
        
        # 残差连接的输入
        residual = x
        
        # 计算Q, K, V - 单头注意力
        Q = self.query(x)  # [B, N, d_model]
        K = self.key(x)    # [B, N, d_model]
        V = self.value(x)  # [B, N, d_model]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, N, N]
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)  # [B, N, d_model]
        
        # 残差连接和层归一化
        output = self.layer_norm(context + residual)
        
        return output


class SimpleFeedForward(nn.Module):
    """简化的前馈网络"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # 使用更小的隐藏层维度
        d_ff = d_model * 2  # 原来是4倍，现在改为2倍
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.layer_norm(x + residual)


class FastAttentionBlock(nn.Module):
    """快速注意力块 - 结合了快速注意力和简化前馈网络"""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.fast_attention = FastAttention(d_model, dropout)
        self.feed_forward = SimpleFeedForward(d_model, dropout)
        
    def forward(self, x):
        x = self.fast_attention(x)
        x = self.feed_forward(x)
        return x


def symfn_max(x):
    """最大池化聚合函数 [B, N, K] -> [B, K]"""
    return torch.max(x, dim=1)[0]


def symfn_avg(x):
    """平均池化聚合函数 [B, N, K] -> [B, K]"""
    return torch.mean(x, dim=1)


def symfn_fast_attention_pool(x):
    """快速注意力池化聚合函数 [B, N, K] -> [B, K]"""
    # 使用简化的注意力权重计算
    batch_size, seq_len, d_model = x.size()
    
    # 计算全局平均特征作为查询
    global_feat = torch.mean(x, dim=1, keepdim=True)  # [B, 1, K]
    
    # 计算注意力权重 - 使用点积注意力
    attention_scores = torch.sum(x * global_feat, dim=-1)  # [B, N]
    attention_weights = F.softmax(attention_scores, dim=-1).unsqueeze(-1)  # [B, N, 1]
    
    # 加权求和
    aggregated = torch.sum(x * attention_weights, dim=1)  # [B, K]
    return aggregated


class FastPointAttention_features(torch.nn.Module):
    """快速点云注意力特征提取器
    
    轻量级设计，保持与PointNet_features的API兼容性
    输入: [B, N, 3] 点云
    输出: [B, K] 全局特征向量
    """
    def __init__(self, dim_k=1024, sym_fn=symfn_max, scale=1, num_attention_blocks=2):
        super().__init__()
        
        # 特征维度设置 - 使用更小的隐藏维度
        self.d_model = int(64 / scale)    # 大幅减少隐藏维度
        self.dim_k = int(dim_k / scale)   # 最终输出特征维度
        self.num_attention_blocks = num_attention_blocks
        
        # 输入嵌入层：3D坐标 -> 特征空间
        self.input_projection = nn.Linear(3, self.d_model - self.d_model // 4)
        
        # 简化的3D位置编码
        self.pos_encoding = SimplifiedPositionalEncoding(self.d_model)
        
        # 快速注意力块（数量较少）
        self.attention_blocks = nn.ModuleList([
            FastAttentionBlock(self.d_model)
            for _ in range(num_attention_blocks)
        ])
        
        # 特征变换层：使用更简单的MLP
        self.feature_transform = MLPNet(
            self.d_model, 
            [int(128/scale), self.dim_k], 
            b_shared=True
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
        
        # 输入投影：3D坐标 -> 特征空间
        x = self.input_projection(points)  # [B, N, d_model-d_model//4]
        
        # 添加简化位置编码
        pos_encoding = self.pos_encoding(points)  # [B, N, d_model//4]
        x = torch.cat([x, pos_encoding], dim=-1)  # [B, N, d_model]
        
        # 保存中间特征（兼容性）
        self.t_out_h1 = x.transpose(1, 2)  # [B, d_model, N] 格式用于兼容
        
        # 通过快速注意力块
        for attention_block in self.attention_blocks:
            x = attention_block(x)  # [B, N, d_model]
        
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
        elif self.sy == symfn_fast_attention_pool:
            global_features = symfn_fast_attention_pool(x)  # [B, dim_k]
        else:
            # 默认使用最大池化
            global_features = symfn_max(x)  # [B, dim_k]
        
        return global_features


class FastPointAttention_classifier(torch.nn.Module):
    """快速点云注意力分类器
    
    基于FastPointAttention_features的轻量级分类器
    """
    def __init__(self, num_c, fast_feat, dim_k):
        """
        Args:
            num_c: 分类数量
            fast_feat: FastPointAttention_features实例
            dim_k: 特征维度
        """
        super().__init__()
        self.features = fast_feat
        
        # 分类头：特征向量 -> 分类结果
        list_layers = mlp_layers(dim_k, [256, 128], b_shared=False, bn_momentum=0.1, dropout=0.0)
        list_layers.append(torch.nn.Linear(128, num_c))
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

        # 对于快速注意力模型，暂不添加额外的正则化项
        # 主要关注计算效率
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