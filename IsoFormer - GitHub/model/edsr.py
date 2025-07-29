import torch
import torch.nn as nn
import torch.nn.init as init


# PixelShuffle3D 实现
def pixelshuffle3d(x, scale_factor):
    """
    输入: (B, C_in, D, H, W)
    输出: (B, C_out, D*scale, H*scale, W*scale)
    其中 C_out = C_in / (scale_factor^3)
    """
    batch_size, channels, depth, height, width = x.size()
    channels //= scale_factor ** 3
    out_depth = depth * scale_factor
    out_height = height * scale_factor
    out_width = width * scale_factor

    x = x.view(batch_size, channels, scale_factor, scale_factor, scale_factor,
               depth, height, width)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    x = x.view(batch_size, channels, out_depth, out_height, out_width)
    return x


class PixelShuffle3D(nn.Module):
    def __init__(self, scale_factor):
        super(PixelShuffle3D, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return pixelshuffle3d(x, self.scale_factor)


# 残差块 ResBlock3D
class ResBlock3D(nn.Module):
    def __init__(self, n_channels=64, kernel_size=3, res_scale=1.0):
        super(ResBlock3D, self).__init__()
        padding = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv3d(n_channels, n_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv3d(n_channels, n_channels, kernel_size, padding=padding)
        )
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


# EDSR-3D 主网络
class EDSR3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, num_blocks=16, scale_factor=2):
        super(EDSR3D, self).__init__()

        # 浅层特征提取
        self.head = nn.Conv3d(in_channels, num_features, kernel_size=3, padding=1)

        # 多个残差块
        self.body = nn.Sequential(
            *[ResBlock3D(num_features) for _ in range(num_blocks)]
        )

        # 最后一个卷积层用于残差连接
        self.tail_conv = nn.Conv3d(num_features, num_features, kernel_size=3, padding=1)

        # 上采样部分
        self.upsample = nn.Sequential(
            nn.Conv3d(num_features, num_features * (scale_factor ** 3), kernel_size=3, padding=1),
            PixelShuffle3D(scale_factor),
            nn.Conv3d(num_features, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # 初始特征提取
        head = self.head(x)

        # 残差块处理
        body = self.body(head)

        # 残差连接
        tail = self.tail_conv(body)
        residual = tail + head

        # 上采样输出
        out = self.upsample(residual)

        return out


