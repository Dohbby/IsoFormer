import math
import torch
import torch.nn.functional as F
from natten.functional import na3d_av, na3d_qk
from torch import Tensor, nn
from torch.nn.init import trunc_normal_
from typing import List, Optional

from basicsr.archs.arch_util import make_layer
from basicsr.utils.registry import ARCH_REGISTRY


class ChannelAttention3D(nn.Module):
    def __init__(self, dim, squeeze_factor=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(dim, dim // squeeze_factor, 1),
            nn.SiLU(),
            nn.Conv3d(dim // squeeze_factor, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.attention(x)


class LAB3D(nn.Module):
    def __init__(self, dim, local_dwconv=3, expanded_ratio=1., squeeze_factor=4):
        super().__init__()
        hidden_dim = int(dim * expanded_ratio)
        self.net = nn.Sequential(
            nn.Conv3d(dim, hidden_dim, 1), nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, local_dwconv, padding=local_dwconv//2, groups=hidden_dim), nn.GELU(),
            ChannelAttention3D(dim=hidden_dim, squeeze_factor=squeeze_factor),
            nn.Conv3d(hidden_dim, dim, 1))

    def forward(self, x):
        u = x.clone()
        x = self.net(x)
        return u + x


class NeighborhoodAttention3D(nn.Module):
    """
    Neighborhood Attention 3D Module
    """
    def __init__(
        self,
        dim: int,
        num_head: int,
        kernel_sizes: List[int] = [7, 9, 11],
        dilations: List[int] = [1, 1, 1],
        is_causal: List[bool] = [False, False],
        rel_pos_bias: bool = False,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(dilations)
        if any(is_causal) and rel_pos_bias:
            raise NotImplementedError("Causal neighborhood attention is undefined with positional biases."
                                    "Please consider disabling positional biases, or open an issue.")

        self.k = len(kernel_sizes)
        self.channels = []
        for i in range(self.k):
            if i == 0:
                channels = dim * 3 - dim * 3 // len(kernel_sizes) * (len(kernel_sizes) - 1)
            else:
                channels = dim * 3 // len(kernel_sizes)
            assert (channels % (3 * num_head // self.k) == 0)
            self.channels.append(channels)

        self.num_head = num_head
        self.head_dim = dim // self.num_head
        self.scale = qk_scale or self.head_dim**-0.5
        self.kernel_sizes = tuple((i, i, i) for i in kernel_sizes)
        self.dilations = tuple((i, i, i) for i in dilations)
        self.is_causal = is_causal

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if rel_pos_bias:
            self.rpb = nn.ParameterList()
            for i in range(len(kernel_sizes)):
                temp = nn.Parameter(torch.zeros(
                    num_head // self.k,
                    (2 * kernel_sizes[i] - 1),
                    (2 * kernel_sizes[i] - 1),
                    (2 * kernel_sizes[i] - 1),
                ))
                trunc_normal_(temp, mean=0.0, std=0.02, a=-2.0, b=2.0)
                self.rpb.append(temp)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop_rate = attn_drop
        self.attn_drop = nn.Dropout(self.attn_drop_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        self.extra_repr()
        if x.dim() != 5:
            raise ValueError(f"NeighborhoodAttention3D expected a rank-5 input tensor; got {x.dim()=}.")

        x = self.qkv(x)
        x = torch.split(x, split_size_or_sections=self.channels, dim=4)
        attns = []
        for i, x_i in enumerate(x):
            B, D, H, W, C = x_i.shape
            qkv = (x_i.reshape(B, D, H, W, 3, self.num_head // self.k, self.head_dim).permute(4, 0, 5, 1, 2, 3, 6))
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * self.scale
            attn = na3d_qk(
                q,
                k,
                kernel_size=self.kernel_sizes[i],
                dilation=self.dilations[i],
                is_causal=self.is_causal,
                rpb=self.rpb[i] if self.rpb is not None else None,
            )
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            y = na3d_av(
                attn,
                v,
                kernel_size=self.kernel_sizes[i],
                dilation=self.dilations[i],
                is_causal=self.is_causal,
            )
            y = y.permute(0, 2, 3, 4, 1, 5).reshape(B, D, H, W, C // 3)
            attns.append(y)
        x = torch.cat(attns, dim=4)
        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (f"head_dim={self.head_dim}, num_head={self.num_head}, " + f"kernel_sizes={self.kernel_sizes}, " +
                f"dilations={self.dilations}, " + f"is_causal={self.is_causal}, " + f"has_bias={self.rpb is not None}")


class MSDWConv3D(nn.Module):
    def __init__(self, dim, dw_sizes=(1, 3, 5, 7)):
        super().__init__()
        self.dw_sizes = dw_sizes
        self.channels = []
        self.proj = nn.ModuleList()
        for i in range(len(dw_sizes)):
            if i == 0:
                channels = dim - dim // len(dw_sizes) * (len(dw_sizes) - 1)
            else:
                channels = dim // len(dw_sizes)
            conv = nn.Conv3d(channels, channels, kernel_size=dw_sizes[i],
                            padding=dw_sizes[i] // 2, groups=channels)
            self.channels.append(channels)
            self.proj.append(conv)

    def forward(self, x):
        x = torch.split(x, split_size_or_sections=self.channels, dim=1)
        out = []
        for i, feat in enumerate(x):
            out.append(self.proj[i](feat))
        x = torch.cat(out, dim=1)
        return x


class MSConvStar3D(nn.Module):
    def __init__(self, dim, mlp_ratio=2., dw_sizes=[1, 3, 5, 7]):
        super().__init__()
        self.dim = dim
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Conv3d(dim, hidden_dim, 1)
        self.dwconv = MSDWConv3D(dim=hidden_dim, dw_sizes=dw_sizes)
        self.fc2 = nn.Conv3d(hidden_dim // 2, dim, 1)
        self.num_head = len(dw_sizes)
        self.act = nn.GELU()

        assert hidden_dim // self.num_head % 2 == 0

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.fc1(x)
        x = x + self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.fc2(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        return x


class MAB3D(nn.Module):
    def __init__(self,
                dim,
                num_head,
                kernel_sizes=[7, 9, 11],
                dilations=[1, 1, 1],
                rel_pos_bias=True,
                qkv_bias=True,
                qk_scale=None,
                attn_drop=0.0,
                proj_drop=0.0,
                mlp_ratio=2.0,
                dw_sizes=[1, 3, 5, 7]) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NeighborhoodAttention3D(
            dim=dim,
            num_head=num_head,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MSConvStar3D(dim, mlp_ratio=mlp_ratio, dw_sizes=dw_sizes)
        self.dilations = dilations

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RMAG3D(nn.Module):
    def __init__(self,
                dim,
                local_dwconv=3,
                expanded_ratio=1.,
                squeeze_factor=4,
                depth=4,
                num_head=6,
                kernel_sizes=[7, 9, 11],
                dilations=[[1, 1, 1], [4, 4, 4]],
                rel_pos_bias=True,
                qkv_bias=True,
                qk_scale=None,
                mlp_ratio=2.0,
                dw_sizes=[1, 3, 5, 7]):
        super().__init__()
        self.lab = LAB3D(dim=dim, local_dwconv=local_dwconv, expanded_ratio=expanded_ratio, squeeze_factor=squeeze_factor)
        self.mabs = nn.ModuleList()
        for i_mab in range(depth):
            mab = MAB3D(
                dim=dim,
                num_head=num_head,
                kernel_sizes=kernel_sizes,
                dilations=dilations[i_mab % 2],
                rel_pos_bias=rel_pos_bias,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                mlp_ratio=mlp_ratio,
                dw_sizes=dw_sizes)
            self.mabs.append(mab)
        self.conv = nn.Conv3d(dim, dim, 3, 1, 1)

    def forward(self, x):
        shortcut = x
        x = self.lab(x)
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        for mab in self.mabs:
            x = mab(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.conv(x)
        x = x + shortcut
        return x


def UpsampleOneStep3D(in_channels, out_channels, upscale_factor=4):
    """
    Upsample features according to `upscale_factor` for 3D data.
    """
    conv = nn.Conv3d(in_channels, out_channels * (upscale_factor**3), 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])


class Upsample3D(nn.Sequential):
    """Upsample module for 3D data."""
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv3d(num_feat, 8 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv3d(num_feat, 27 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample3D, self).__init__(*m)


class PixelShuffleBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor=4):
        super().__init__()
        num_feat = 64
        self.conv_before_upsample = nn.Sequential(
            nn.Conv3d(in_channels, num_feat, 3, 1, 1),
            nn.LeakyReLU(inplace=True))
        self.upsample = Upsample3D(upscale_factor, num_feat)
        self.conv_last = nn.Conv3d(num_feat, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))
        return x


@ARCH_REGISTRY.register()
class MAT3D(nn.Module):
    def __init__(self,
                num_in_ch=3,
                num_out_ch=3,
                num_feat=60,
                num_block=4,
                local_dwconv=3,
                expanded_ratio=1.0,
                squeeze_factor=4,
                depth=4,
                num_head=6,
                kernel_sizes=[7, 9, 11],
                dilations=[[1, 1, 1], [4, 4, 4]],
                rel_pos_bias=True,
                qkv_bias=True,
                qk_scale=False,
                mlp_ratio=2.0,
                dw_sizes=[1, 3, 5, 7],
                upscale=4,
                upsampler='',
                img_range=1):
        super().__init__()
        self.img_range = img_range
        if num_in_ch == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1, 1)

        self.fea_conv = nn.Conv3d(num_in_ch, num_feat, 3, 1, 1)

        self.body = make_layer(
            RMAG3D,
            num_block,
            dim=num_feat,
            local_dwconv=local_dwconv,
            expanded_ratio=expanded_ratio,
            squeeze_factor=squeeze_factor,
            depth=depth,
            num_head=num_head,
            kernel_sizes=kernel_sizes,
            dilations=dilations,
            rel_pos_bias=rel_pos_bias,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mlp_ratio=mlp_ratio,
            dw_sizes=dw_sizes)
        self.conv_afterbody = nn.Conv3d(num_feat, num_feat, 3, 1, 1)

        self.upscale = upscale
        if upsampler == 'pixelshuffledirect':
            self.upsampler = UpsampleOneStep3D(num_feat, num_out_ch, upscale_factor=upscale)
        elif upsampler == 'pixelshuffle':
            self.upsampler = PixelShuffleBlock3D(num_feat, num_out_ch, upscale_factor=upscale)
        else:
            raise NotImplementedError("Check the Upsampler. None or not support yet.")

    def check_image_size(self, x, min_size=64):
        _, _, D, H, W = x.size()
        if D >= min_size and H >= min_size and W >= min_size:
            return x
        mod_pad_d = max(min_size - D, 0)
        mod_pad_h = max(min_size - H, 0)
        mod_pad_w = max(min_size - W, 0)
        padding = (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d)
        x = F.pad(x, padding, 'reflect')
        return x

    def forward(self, x):
        D, H, W = x.shape[2:]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.fea_conv(x)
        x = self.conv_afterbody(self.body(x)) + x
        x = self.upsampler(x)

        x = x / self.img_range + self.mean
        return x[:, :, :D * self.upscale, :H * self.upscale, :W * self.upscale]