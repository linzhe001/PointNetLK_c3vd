""" Attention-based Point Cloud Feature Extractor v1.0
    专为点云配准任务设计的基于注意力机制的特征提取器
    
    设计考虑:
    1. 点云的无序性 - 使用注意力机制天然处理permutation invariance
    2. 点云配准任务 - 需要捕获局部和全局几何特征
    3. 与PointNet_features兼容的接口
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
    """ 多层感知机网络，用于attention模块中的特征变换
        支持共享权重(Conv1d)和非共享权重(Linear)两种模式
    """
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        self.b_shared = b_shared
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        return self.layers(inp)


class PositionalEncoding3D(nn.Module):
    """3D位置编码，为点云中的每个点提供位置信息"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # 创建位置编码投影层
        self.pos_projection = nn.Linear(3, d_model)
        
    def forward(self, points):
        """
        Args:
            points: [B, N, 3] 点云坐标
        Returns:
            pos_encoding: [B, N, d_model] 位置编码
        """
        # 直接使用线性层将3D坐标映射到d_model维度
        pos_encoding = self.pos_projection(points)  # [B, N, d_model]
        return pos_encoding


class MultiHeadSelfAttention(nn.Module):
    """多头自注意力机制，专为点云设计"""
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Query, Key, Value投影层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
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
        
        # 计算Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, N, N]
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        context = torch.matmul(attention_weights, V)  # [B, H, N, d_k]
        
        # 重新组织多头输出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)  # [B, N, d_model]
        
        # 输出投影
        output = self.w_o(context)
        
        # 残差连接和层归一化
        output = self.layer_norm(output + residual)
        
        return output


class FeedForwardNetwork(nn.Module):
    """前馈网络，用于attention block中的特征变换"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return self.layer_norm(x + residual)


class AttentionBlock(nn.Module):
    """完整的注意力块，包含自注意力和前馈网络"""
    def __init__(self, d_model, num_heads=8, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
            
        self.self_attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff, dropout)
        
    def forward(self, x):
        x = self.self_attention(x)
        x = self.feed_forward(x)
        return x


def symfn_max(x):
    """最大池化聚合函数 [B, N, K] -> [B, K]"""
    return torch.max(x, dim=1)[0]


def symfn_avg(x):
    """平均池化聚合函数 [B, N, K] -> [B, K]"""
    return torch.mean(x, dim=1)


def symfn_attention_pool(x):
    """基于注意力的聚合函数 [B, N, K] -> [B, K]"""
    # 使用简单的注意力权重进行聚合
    batch_size, seq_len, d_model = x.size()
    
    # 计算注意力权重
    attention_weights = torch.softmax(torch.sum(x, dim=-1), dim=-1)  # [B, N]
    attention_weights = attention_weights.unsqueeze(-1)  # [B, N, 1]
    
    # 加权求和
    aggregated = torch.sum(x * attention_weights, dim=1)  # [B, K]
    return aggregated


class AttentionNet_features(torch.nn.Module):
    """基于注意力机制的点云特征提取器
    
    替换PointNet_features，专为点云配准任务设计
    输入: [B, N, 3] 点云
    输出: [B, K] 全局特征向量
    """
    def __init__(self, dim_k=1024, sym_fn=symfn_max, scale=1, num_attention_blocks=3, num_heads=8):
        super().__init__()
        
        # 特征维度设置
        self.d_model = int(128 / scale)  # 注意力模块的隐藏维度
        self.dim_k = int(dim_k / scale)   # 最终输出特征维度
        self.num_attention_blocks = num_attention_blocks
        
        # 输入嵌入层：将3D坐标映射到高维特征空间
        self.input_projection = nn.Linear(3, self.d_model)
        
        # 3D位置编码
        self.pos_encoding = PositionalEncoding3D(self.d_model)
        
        # 多层注意力块
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(self.d_model, num_heads=num_heads)
            for _ in range(num_attention_blocks)
        ])
        
        # 特征变换层：从attention维度映射到最终特征维度
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
        pos_encoding = self.pos_encoding(points)  # [B, N, d_model]
        x = x + pos_encoding
        
        # 保存中间特征（兼容性）
        self.t_out_h1 = x.transpose(1, 2)  # [B, d_model, N] 格式用于兼容
        
        # 通过多层注意力块
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
        elif self.sy == symfn_attention_pool:
            global_features = symfn_attention_pool(x)  # [B, dim_k]
        else:
            # 默认使用最大池化
            global_features = symfn_max(x)  # [B, dim_k]
        
        return global_features


class AttentionNet_classifier(torch.nn.Module):
    """基于注意力机制的点云分类器
    
    类似于PointNet_classifier，使用AttentionNet_features作为特征提取器
    """
    def __init__(self, num_c, attnfeat, dim_k):
        """
        Args:
            num_c: 分类数量
            attnfeat: AttentionNet_features实例
            dim_k: 特征维度
        """
        super().__init__()
        self.features = attnfeat
        
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

        # 注意：对于attention模型，我们目前不添加变换矩阵正则化
        # 因为attention机制的结构与PointNet不同，没有对应的t_out_t2
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