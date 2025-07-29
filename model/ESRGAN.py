import torch.nn as nn
import torch


class ResidualDenseBlock3D(nn.Module):
    def __init__(self, nf, gc=32, res_scale=0.2):
        super(ResidualDenseBlock3D, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv3d(nf + 0 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Conv3d(nf + 1 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Conv3d(nf + 2 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer4 = nn.Sequential(nn.Conv3d(nf + 3 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer5 = nn.Sequential(nn.Conv3d(nf + 4 * gc, nf, 3, padding=1, bias=True), nn.LeakyReLU())

        self.res_scale = res_scale

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(torch.cat((x, layer1), 1))
        layer3 = self.layer3(torch.cat((x, layer1, layer2), 1))
        layer4 = self.layer4(torch.cat((x, layer1, layer2, layer3), 1))
        layer5 = self.layer5(torch.cat((x, layer1, layer2, layer3, layer4), 1))
        return layer5.mul(self.res_scale) + x


class ResidualInResidualDenseBlock3D(nn.Module):
    def __init__(self, nf, gc=32, res_scale=0.2):
        super(ResidualInResidualDenseBlock3D, self).__init__()
        self.layer1 = ResidualDenseBlock3D(nf, gc)
        self.layer2 = ResidualDenseBlock3D(nf, gc)
        self.layer3 = ResidualDenseBlock3D(nf, gc)
        self.res_scale = res_scale

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out.mul(self.res_scale) + x


class PixelShuffle3D(nn.Module):
    def __init__(self, scale_factor):
        super(PixelShuffle3D, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()
        channels //= self.scale_factor ** 3

        # 重新排列通道到空间维度
        x = x.view(batch_size, channels, self.scale_factor, self.scale_factor, self.scale_factor, depth, height, width)
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(batch_size, channels,
                   depth * self.scale_factor,
                   height * self.scale_factor,
                   width * self.scale_factor)
        return x


def upsample_block3d(nf, scale_factor=2):
    block = []
    for _ in range(scale_factor // 2):
        block += [
            nn.Conv3d(nf, nf * (2 ** 3), 1),  # 通道数增加8倍(2^3)
            PixelShuffle3D(2),  # 空间分辨率提高2倍
            nn.LeakyReLU()
        ]
    return nn.Sequential(*block)


class ESRGAN3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, nf=64, gc=32, scale_factor=1, n_basic_block=6):
        super(ESRGAN3D, self).__init__()

        # 使用反射填充的3D替代方案
        self.conv1 = nn.Sequential(
            nn.ReplicationPad3d(1),  # 在深度、高度和宽度上各填充1
            nn.Conv3d(in_channels, nf, 3),
            nn.LeakyReLU()
        )

        basic_block_layer = []
        for _ in range(n_basic_block):
            basic_block_layer += [ResidualInResidualDenseBlock3D(nf, gc)]

        self.basic_block = nn.Sequential(*basic_block_layer)

        self.conv2 = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.Conv3d(nf, nf, 3),
            nn.LeakyReLU()
        )

        self.upsample = upsample_block3d(nf, scale_factor=scale_factor)

        self.conv3 = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.Conv3d(nf, nf, 3),
            nn.LeakyReLU()
        )

        self.conv4 = nn.Sequential(
            nn.ReplicationPad3d(1),
            nn.Conv3d(nf, out_channels, 3),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x = self.basic_block(x1)
        x = self.conv2(x)
        x = self.upsample(x + x1)  # 残差连接
        x = self.conv3(x)
        x = self.conv4(x)
        return x