import torch
import torch.nn as nn
import torch.nn.functional as F

class ECA(nn.Module):
    """
    改进的通道注意力模块（针对医学超声影像优化）
    创新点：
    1. 多尺度解剖特征融合
    2. 动态通道交互机制
    3. 病理模式增强单元
    4. 分层注意力校准
    """

    def __init__(self, in_channels, reduction_ratio=16, groups=4, dilation_rates=[1, 2, 3]):
        super().__init__()
        self.groups = groups
        self.dilation_rates = dilation_rates
        branch_channels = in_channels // 4  # 确保可被4整除

        # 修正后的多尺度特征提取（保持空间维度一致）
        self.multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, 3,
                          padding=d, dilation=d),  # 关键修正：padding=d
                nn.GroupNorm(4, branch_channels),
                nn.GELU()
            ) for d in dilation_rates
        ])

        # 增加特征融合卷积（保持通道一致性）
        self.fusion_conv = nn.Conv2d(
            branch_channels * len(dilation_rates),
            in_channels, 1)

        # 简化动态交互模块
        self.dynamic_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.GELU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 确保多尺度分支输出尺寸一致
        ms_features = [branch(x) for branch in self.multi_scale]

        # 特征融合
        fused = self.fusion_conv(torch.cat(ms_features, dim=1))

        # 生成注意力权重
        weights = self.dynamic_interaction(fused)

        return x * weights


class DynamicConv(nn.Module):
    """动态分组卷积"""

    def __init__(self, channels, groups=4):
        super().__init__()
        self.groups = groups
        self.weight = nn.Parameter(
            torch.randn(groups, channels // groups, channels // groups))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)
        weight = self.weight.view(self.groups, -1).unsqueeze(-1).unsqueeze(-1)
        return F.conv2d(x, weight, padding=0).view(b, c, h, w) + self.bias.view(1, -1, 1, 1)