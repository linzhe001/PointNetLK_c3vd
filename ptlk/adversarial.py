import torch
import torch.nn as nn

class GradReverse(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL)
    此层在前向传播中是恒等变换，但在反向传播中会反转梯度的符号。
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        # 保存lambda值以供反向传播使用
        ctx.lambda_ = lambda_
        # 前向传播返回原始输入
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时，梯度乘以-lambda
        return (grad_output.neg() * ctx.lambda_), None

class DomainDiscriminator(nn.Module):
    """
    领域判别器网络。
    一个简单的多层感知机，用于区分特征来自哪个领域（例如MRI或视频）。
    """
    def __init__(self, input_dim=1024, hidden_dim=512):
        super(DomainDiscriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 2)  # 2个领域: MRI vs. Video
        )

    def forward(self, x):
        """
        Args:
            x: [B, input_dim] 输入的全局特征
        Returns:
            [B, 2] 每个样本属于两个领域的logits
        """
        return self.network(x) 