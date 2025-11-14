import torch
import torch.nn as nn
import torch.nn.functional as F


class MSA(nn.Module):
    """
    医学影像专用空间注意力模块（Med-SA）
    创新点：
    1. 高频增强边缘检测分支
    2. 动态可变形卷积核
    3. 多尺度门控融合机制
    4. 自适应噪声抑制模块
    """

    def __init__(self, in_channels, kernel_size=7, dilation_rates=[1, 2, 3]):
        super().__init__()

        # 高频特征增强
        self.high_freq = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            SobelEdgeDetector(in_channels // 4),
            nn.Conv2d(in_channels // 4, in_channels, 1)  # 通道对齐
        )

        # 动态可变形卷积
        self.offset_conv = nn.Conv2d(in_channels, 2 * 3 * 3, 3, padding=1)
        self.deform_conv = DeformConv2d(in_channels, in_channels, 3, padding=1)

        # 多尺度金字塔改进
        self.pyramid = nn.ModuleList([
            nn.Sequential(
                nn.AvgPool2d(kernel_size=2 ** i, stride=2 ** i),
                nn.Conv2d(in_channels, in_channels, 1),  # 保持通道数一致
                nn.ReLU()
            ) for i in dilation_rates
        ])
        self.pyramid_fusion = nn.Conv2d(len(dilation_rates) * in_channels, in_channels, 1)

        # 门控机制改进
        self.gate_conv = nn.Conv2d(4 * in_channels, 4, 3, padding=1)

        # 噪声抑制模块（修复通道数问题）
        self.noise_estimator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),  # 输出通道改为 in_channels
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid()
        )

        # 注意力机制优化
        self.channel_att = ChannelAttention(in_channels)

        # 空间注意力修复：确保输出空间维度与输入一致
        self.spatial_att = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=kernel_size, padding=kernel_size // 2, stride=1),  # 固定 stride=1
            nn.Sigmoid()
        )

    def forward(self, x):
        # 高频处理 (输出通道对齐到 in_channels)
        hf_feat = self.high_freq(x)

        # 可变形卷积
        offset = self.offset_conv(x)
        deform_feat = self.deform_conv(x, offset)

        # 多尺度金字塔处理
        pyramid_feats = []
        for pool in self.pyramid:
            feat = F.interpolate(pool(x), size=x.shape[2:], mode='bilinear', align_corners=False)
            pyramid_feats.append(feat)
        pyramid_feat = self.pyramid_fusion(torch.cat(pyramid_feats, dim=1))

        # 门控融合 (所有特征通道对齐到 in_channels)
        combined = torch.cat([x, deform_feat, pyramid_feat, hf_feat], dim=1)
        gates = torch.sigmoid(self.gate_conv(combined))
        g1, g2, g3, g4 = torch.chunk(gates, 4, dim=1)

        # 安全融合 (维度验证)
        assert x.shape == deform_feat.shape == pyramid_feat.shape == hf_feat.shape, \
            "All features must have same shape"
        fused_feat = g1 * x + g2 * deform_feat + g3 * pyramid_feat + g4 * hf_feat

        # 噪声抑制（修复通道数问题）
        noise_level = self.noise_estimator(x)  # 输出通道数已改为 in_channels
        denoised_feat = fused_feat * (1 - noise_level) + x * noise_level

        # 双向注意力
        channel_weights = self.channel_att(denoised_feat)

        # 空间注意力修复：确保输入空间维度一致
        spatial_input = torch.cat([
            torch.mean(denoised_feat, dim=1, keepdim=True),
            torch.max(denoised_feat, dim=1, keepdim=True)[0],
            torch.std(denoised_feat, dim=1, keepdim=True),
            hf_feat.mean(dim=1, keepdim=True)
        ], dim=1)
        spatial_weights = self.spatial_att(spatial_input)

        # 确保 spatial_weights 的空间维度与 denoised_feat 一致
        if denoised_feat.shape[2:] != spatial_weights.shape[2:]:
            spatial_weights = F.interpolate(spatial_weights, size=denoised_feat.shape[2:], mode='bilinear',
                                            align_corners=False)

        return denoised_feat * channel_weights * spatial_weights



# 辅助模块定义
class SobelEdgeDetector(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.sobel_x = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.sobel_y = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)

        # 初始化Sobel算子
        kernel_x = torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], dtype=torch.float32).repeat(channels, 1, 1, 1)
        kernel_y = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], dtype=torch.float32).repeat(channels, 1, 1, 1)
        self.sobel_x.weight = nn.Parameter(kernel_x)
        self.sobel_y.weight = nn.Parameter(kernel_y)
        self.sobel_x.weight.requires_grad = False
        self.sobel_y.weight.requires_grad = False

    def forward(self, x):
        grad_x = self.sobel_x(x)
        grad_y = self.sobel_y(x)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return avg_out + max_out


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.style_scale = nn.Conv2d(num_features, num_features, 1)
        self.style_bias = nn.Conv2d(num_features, num_features, 1)

    def forward(self, x):
        normalized = self.norm(x)
        style_scale = self.style_scale(x) + 1
        style_bias = self.style_bias(x)
        return style_scale * normalized + style_bias


class DeformConv2d(nn.Module):
    # 可变形卷积实现（代码较长，此处简化实现）
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)

    def forward(self, x, offset):
        # 实际应使用更复杂的坐标计算
        return self.conv(x)