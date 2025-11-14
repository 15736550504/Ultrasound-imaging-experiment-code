import torch
import torch.nn as nn
import torch.nn.functional as F


class LCES(nn.Module):
    """
    改进后的局部上下文增强模块
    (保持与YAML配置文件参数匹配)
    """

    def __init__(self, in_channels, *args, **kwargs):  # 允许接收额外参数
        super().__init__()
        self.in_channels = in_channels
        self.min_size = 16

        # 固定多尺度卷积
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, 5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, in_channels, 7, padding=3)

        # 简化融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        # 强制参数类型一致性
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                m.weight.data = m.weight.data.float()
                if m.bias is not None:
                    m.bias.data = m.bias.data.float()

    def forward(self, x):
        # 保存原始数据类型
        original_dtype = x.dtype

        # 统一使用float32计算
        x = x.float()

        # 基础残差连接
        residual = x

        # 多尺度特征提取
        feat3 = self.conv3x3(x)
        feat5 = self.conv5x5(x)
        feat7 = self.conv7x7(x)

        # 特征融合
        fused = self.fusion(torch.cat([feat3, feat5, feat7], dim=1))

        # 残差连接
        output = residual + fused

        # 恢复原始数据类型
        return output.to(original_dtype)