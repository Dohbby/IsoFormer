from typing import Tuple, List
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from basicsr.utils.registry import ARCH_REGISTRY


######################
# Meta Architecture
######################
# @ARCH_REGISTRY.register()
class SeemoRe3D(nn.Module):
    def __init__(self,
                 scale: int = 1,
                 in_chans: int = 1,
                 num_experts: int = 3,
                 num_layers: int = 6,
                 embedding_dim: int = 64,
                 img_range: float = 1.0,
                 use_shuffle: bool = True,
                 global_kernel_size: int = 11,
                 recursive: int = 2,
                 lr_space: str='exp',
                 topk: int = 1, ):
        super().__init__()
        self.scale = scale
        self.num_in_channels = in_chans
        self.num_out_channels = in_chans
        self.img_range = img_range

        # rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor([0.0])

        # -- SHALLOW FEATURES --
        self.conv_1 = nn.Conv3d(self.num_in_channels, embedding_dim, kernel_size=3, padding=1)

        # -- DEEP FEATURES --
        self.body = nn.ModuleList(
            [ResGroup3D(in_ch=embedding_dim,
                        num_experts=num_experts,
                        use_shuffle=use_shuffle,
                        topk=topk,
                        lr_space=lr_space,
                        recursive=recursive,
                        global_kernel_size=global_kernel_size) for i in range(num_layers)]
        )

        # -- UPSCALE --
        self.norm = LayerNorm3D(embedding_dim, data_format='channels_first')
        self.conv_2 = nn.Conv3d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        self.upsampler = nn.Sequential(
            nn.Conv3d(embedding_dim, (scale ** 3) * self.num_out_channels, kernel_size=3, padding=1),
            # nn.PixelShuffle3D(scale)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # -- SHALLOW FEATURES --
        x = self.conv_1(x)
        res = x

        # -- DEEP FEATURES --
        for idx, layer in enumerate(self.body):
            x = layer(x)

        x = self.norm(x)

        # -- HR IMAGE RECONSTRUCTION --
        x = self.conv_2(x) + res
        x = self.upsampler(x)

        x = x / self.img_range + self.mean
        return x


#############################
# Components
#############################
class ResGroup3D(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int,
                 global_kernel_size: int = 11,
                 lr_space: int = 1,
                 topk: int = 2,
                 recursive: int = 2,
                 use_shuffle: bool = False):
        super().__init__()

        self.local_block = RME3D(in_ch=in_ch,
                                 num_experts=num_experts,
                                 use_shuffle=use_shuffle,
                                 lr_space=lr_space,
                                 topk=topk,
                                 recursive=recursive)
        self.global_block = SME3D(in_ch=in_ch,
                                  kernel_size=global_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_block(x)
        x = self.global_block(x)
        return x


#############################
# Global Block
#############################
class SME3D(nn.Module):
    def __init__(self,
                 in_ch: int,
                 kernel_size: int = 11):
        super().__init__()

        self.norm_1 = LayerNorm3D(in_ch, data_format='channels_first')
        self.block = StripedConvFormer3D(in_ch=in_ch, kernel_size=kernel_size)

        self.norm_2 = LayerNorm3D(in_ch, data_format='channels_first')
        self.ffn = GatedFFN3D(in_ch, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(self.norm_1(x)) + x
        x = self.ffn(self.norm_2(x)) + x
        return x


class StripedConvFormer3D(nn.Module):
    def __init__(self,
                 in_ch: int,
                 kernel_size: int):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.proj = nn.Conv3d(in_ch, in_ch, kernel_size=1, padding=0)
        self.to_qv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch * 2, kernel_size=1, padding=0),
            nn.GELU(),
        )

        self.attn = StripedConv3D(in_ch, kernel_size=kernel_size, depthwise=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, v = self.to_qv(x).chunk(2, dim=1)
        q = self.attn(q)
        x = self.proj(q * v)
        return x


#############################
# Local Blocks
#############################
class RME3D(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int,
                 topk: int,
                 lr_space: int = 1,
                 recursive: int = 2,
                 use_shuffle: bool = False, ):
        super().__init__()

        self.use_shuffle = use_shuffle
        self.recursive = recursive

        self.conv_1 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(in_ch, 2 * in_ch, kernel_size=1, padding=0)
        )

        self.agg_conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=4, stride=4, groups=in_ch),
            nn.GELU())

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch),
            nn.Conv3d(in_ch, in_ch, kernel_size=1, padding=0)
        )

        self.conv_2 = nn.Sequential(
            StripedConv3D(in_ch, kernel_size=3, depthwise=True),
            nn.GELU())

        if lr_space == "linear":
            grow_func = lambda i: i + 2
        elif lr_space == "exp":
            grow_func = lambda i: 2 ** (i + 1)
        elif lr_space == "double":
            grow_func = lambda i: 2 * i + 2
        else:
            raise NotImplementedError(f"lr_space {lr_space} not implemented")

        self.moe_layer = MoELayer3D(
            experts=[Expert3D(in_ch=in_ch, low_dim=grow_func(i)) for i in range(num_experts)],
            gate=Router3D(in_ch=in_ch, num_experts=num_experts),
            num_expert=topk,
        )

        self.proj = nn.Conv3d(in_ch, in_ch, kernel_size=1, padding=0)

    def calibrate(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        res = x

        for _ in range(self.recursive):
            x = self.agg_conv(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(d, h, w), mode='trilinear', align_corners=False)
        return res + x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)

        if self.use_shuffle:
            x = channel_shuffle3d(x, groups=2)
        x, k = torch.chunk(x, chunks=2, dim=1)

        x = self.conv_2(x)
        k = self.calibrate(k)

        x = self.moe_layer(x, k)
        x = self.proj(x)
        return x


class MoELayer3D(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_expert: int = 1):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_expert = num_expert

    def forward(self, inputs: torch.Tensor, k: torch.Tensor):
        out = self.gate(inputs)
        weights = F.softmax(out, dim=1, dtype=torch.float).to(inputs.dtype)
        topk_weights, topk_experts = torch.topk(weights, self.num_expert)
        out = inputs.clone()

        if self.training:
            exp_weights = torch.zeros_like(weights)
            exp_weights.scatter_(1, topk_experts, weights.gather(1, topk_experts))
            for i, expert in enumerate(self.experts):
                out += expert(inputs, k) * exp_weights[:, i:i + 1, None, None, None]
        else:
            selected_experts = [self.experts[i] for i in topk_experts.squeeze(dim=0)]
            for i, expert in enumerate(selected_experts):
                out += expert(inputs, k) * topk_weights[:, i:i + 1, None, None, None]

        return out


class Expert3D(nn.Module):
    def __init__(self,
                 in_ch: int,
                 low_dim: int, ):
        super().__init__()
        self.conv_1 = nn.Conv3d(in_ch, low_dim, kernel_size=1, padding=0)
        self.conv_2 = nn.Conv3d(in_ch, low_dim, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv3d(low_dim, in_ch, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.conv_2(k) * x
        x = self.conv_3(x)
        return x


class Router3D(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int):
        super().__init__()

        self.body = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            Rearrange('b c 1 1 1 -> b c'),
            nn.Linear(in_ch, num_experts, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


#################
# Utilities
#################
class StripedConv3D(nn.Module):
    def __init__(self,
                 in_ch: int,
                 kernel_size: int,
                 depthwise: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=(1, 1, self.kernel_size),
                      padding=(0, 0, self.padding), groups=in_ch if depthwise else 1),
            nn.Conv3d(in_ch, in_ch, kernel_size=(1, self.kernel_size, 1),
                      padding=(0, self.padding, 0), groups=in_ch if depthwise else 1),
            nn.Conv3d(in_ch, in_ch, kernel_size=(self.kernel_size, 1, 1),
                      padding=(self.padding, 0, 0), groups=in_ch if depthwise else 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def channel_shuffle3d(x, groups=2):
    bat_size, channels, d, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, d, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, d, w, h)
    return x


class GatedFFN3D(nn.Module):
    def __init__(self,
                 in_ch,
                 mlp_ratio,
                 kernel_size,
                 act_layer, ):
        super().__init__()
        mlp_ch = in_ch * mlp_ratio

        self.fn_1 = nn.Sequential(
            nn.Conv3d(in_ch, mlp_ch, kernel_size=1, padding=0),
            act_layer,
        )
        self.fn_2 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=1, padding=0),
            act_layer,
        )

        self.gate = nn.Conv3d(mlp_ch // 2, mlp_ch // 2,
                              kernel_size=kernel_size, padding=kernel_size // 2, groups=mlp_ch // 2)

    def forward(self, x: torch.Tensor):
        x = self.fn_1(x)
        x, gate = torch.chunk(x, 2, dim=1)

        gate = self.gate(gate)
        x = x * gate

        x = self.fn_2(x)
        return x


class LayerNorm3D(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class PixelShuffle3D(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        batch_size, channels, in_depth, in_height, in_width = x.size()
        channels //= self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        x_view = x.contiguous().view(
            batch_size, channels, self.scale, self.scale, self.scale,
            in_depth, in_height, in_width
        )

        return x_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous().view(
            batch_size, channels, out_depth, out_height, out_width
        )