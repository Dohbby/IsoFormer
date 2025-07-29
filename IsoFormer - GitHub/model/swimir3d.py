# SwinIR3D: 3D Image Restoration Using Swin Transformer
# Adapted from SwinIR (https://arxiv.org/abs/2108.10257) by Ze Liu, Modified by Jingyun Liang
# Extended to 3D by Grok for handling (33, 33, 33) volumetric data
# -----------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_3tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition_3d(x, window_size):
    """
    将输入体数据划分为非重叠的 3D 窗口
    Args:
        x: (B, D, H, W, C) 输入体数据
        window_size: (int, int, int) 窗口大小 (Wd, Wh, Ww)
    Returns:
        windows: (num_windows*B, window_size[0], window_size[1], window_size[2], C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse_3d(windows, window_size, D, H, W):
    """
    将窗口合并回体数据
    Args:
        windows: (num_windows*B, window_size[0], window_size[1], window_size[2], C)
        window_size: (int, int, int) 窗口大小 (Wd, Wh, Ww)
        D, H, W: 原始体数据的深度、高度、宽度
    Returns:
        x: (B, D, H, W, C)
    """
    B = int(windows.shape[0] / (D * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wd, Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 3D 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))

        # 获取 3D 窗口内每对 token 的相对位置索引
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w], indexing='ij'))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 移位从 0 开始
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        flops += N * self.dim * 3 * self.dim  # qkv
        flops += self.num_heads * N * (self.dim // self.num_heads) * N  # attn
        flops += self.num_heads * N * N * (self.dim // self.num_heads)  # x = attn @ v
        flops += N * self.dim * self.dim  # proj
        return flops


class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (D, H, W)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= min(self.window_size):
            self.shift_size = (0, 0, 0)
            self.window_size = (min(self.input_resolution), min(self.input_resolution), min(self.input_resolution))
        assert all(0 <= s < w for s, w in zip(self.shift_size, self.window_size)), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if any(s > 0 for s in self.shift_size):
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        D, H, W = x_size
        img_mask = torch.zeros((1, D, H, W, 1))  # 1 D H W 1
        d_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        h_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        w_slices = (slice(0, -self.window_size[2]),
                    slice(-self.window_size[2], -self.shift_size[2]),
                    slice(-self.shift_size[2], None))
        cnt = 0
        for d in d_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition_3d(img_mask, self.window_size)  # nW, window_size[0], window_size[1], window_size[2], 1
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        D, H, W = x_size
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # 循环移位
        if any(s > 0 for s in self.shift_size):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
        else:
            shifted_x = x

        # 划分窗口
        x_windows = window_partition_3d(shifted_x, self.window_size)  # nW*B, window_size[0], window_size[1], window_size[2], C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)

        # W-MSA/SW-MSA
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse_3d(attn_windows, self.window_size, D, H, W)

        # 反向循环移位
        if any(s > 0 for s in self.shift_size):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x
        x = x.view(B, D * H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        D, H, W = self.input_resolution
        flops += self.dim * D * H * W  # norm1
        nW = D * H * W / (self.window_size[0] * self.window_size[1] * self.window_size[2])
        flops += nW * self.attn.flops(self.window_size[0] * self.window_size[1] * self.window_size[2])
        flops += 2 * D * H * W * self.dim * self.dim * self.mlp_ratio  # mlp
        flops += self.dim * D * H * W  # norm2
        return flops


class PatchMerging3D(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)  # 3D 合并 8 个 patch
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        D, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"
        assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0, f"x size ({D}*{H}*{W}) are not even."

        x = x.view(B, D, H, W, C)

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B D/2 H/2 W/2 8*C
        x = x.view(B, -1, 8 * C)  # B (D/2*H/2*W/2) 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        D, H, W = self.input_resolution
        flops = D * H * W * self.dim
        flops += (D // 2) * (H // 2) * (W // 2) * 8 * self.dim * 2 * self.dim
        return flops


class BasicLayer3D(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock3D(dim=dim, input_resolution=input_resolution,
                                   num_heads=num_heads, window_size=window_size,
                                   shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2),
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                   drop=drop, attn_drop=attn_drop,
                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                   norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB3D(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=33, patch_size=1, resi_connection='1conv'):
        super(RSTB3D, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer3D(dim=dim,
                                           input_resolution=input_resolution,
                                           depth=depth,
                                           num_heads=num_heads,
                                           window_size=window_size,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path=drop_path,
                                           norm_layer=norm_layer,
                                           downsample=downsample,
                                           use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv3d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv3d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        D, H, W = self.input_resolution
        flops += D * H * W * self.dim * self.dim * 27  # 3x3x3 conv
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()
        return flops


class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=33, patch_size=1, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B Pd*Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        D, H, W = self.img_size
        if self.norm is not None:
            flops += D * H * W * self.embed_dim
        return flops


class PatchUnEmbed3D(nn.Module):
    def __init__(self, img_size=33, patch_size=1, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, L, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1], x_size[2])  # B C D H W
        return x

    def flops(self):
        return 0
#
# class Upsample3D(nn.Sequential):
#     def __init__(self, scale, num_feat):
#         m = []
#         if (scale & (scale - 1)) == 0:  # scale = 2^n
#             for _ in range(int(math.log(scale, 2))):
#                 m.append(nn.Conv3d(num_feat, num_feat, 3, 1, 1))  # 输出 num_feat * 8
#                 m.append(nn.PixelShuffle(2))  # 通道数除以 8，恢复为 num_feat
#         elif scale == 3:
#             m.append(nn.Conv3d(num_feat, num_feat, 3, 1, 1))
#             m.append(nn.PixelShuffle(3))
#         else:
#             raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
#         super(Upsample3D, self).__init__(*m)


class Upsample3D(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = [
            nn.Conv3d(num_feat, num_feat, 3, 1, 1),  # 保持通道数不变
            nn.LeakyReLU(inplace=True)  # 添加激活函数以增强非线性
        ]
        super(Upsample3D, self).__init__(*m)
        self.scale = scale

    def forward(self, x):
        x = super().forward(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=False)
        return x

class UpsampleOneStep3D(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv3d(num_feat, (scale ** 3) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep3D, self).__init__(*m)

    def flops(self):
        D, H, W = self.input_resolution
        flops = D * H * W * self.num_feat * 3 * 3 * 3
        return flops


class SwinIR3D(nn.Module):
    def __init__(self, img_size=32, patch_size=1, in_chans=1,
                 embed_dim=64, depths=[2,2,2,2,2,2], num_heads=[2,2,2,2,2,2],
                 window_size=(4,4,4), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=1, img_range=1., upsampler='pixelshuffle', resi_connection='3conv',
                 **kwargs):
        super(SwinIR3D, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        self.mean = torch.zeros(1, 1, 1, 1, 1) if in_chans == 1 else torch.Tensor((0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        # 浅层特征提取
        self.conv_first = nn.Conv3d(num_in_ch, embed_dim, 3, 1, 1)

        # 深层特征提取
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # 分割体数据为非重叠 patch
        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 合并 patch 回体数据
        self.patch_unembed = PatchUnEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度衰减
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 构建 RSTB 块
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB3D(
                dim=embed_dim,
                input_resolution=patches_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # 深层特征提取的最后一层卷积
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv3d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(embed_dim // 4, embed_dim, 3, 1, 1))

        # 高质量体重建
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv3d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True))
            self.upsample = Upsample3D(upscale, num_feat)
            self.conv_last = nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep3D(upscale, embed_dim, num_out_ch, patches_resolution)
        elif self.upsampler == 'nearest+conv':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv3d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv3d(num_feat, num_feat, 3, 1, 1)
            if self.upscale == 4:
                self.conv_up2 = nn.Conv3d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv3d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.conv_last = nn.Conv3d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # def check_image_size(self, x):
    #     _, _, d, h, w = x.size()
    #     mod_pad_d = (self.window_size[0] - d % self.window_size[0]) % self.window_size[0]
    #     mod_pad_h = (self.window_size[1] - h % self.window_size[1]) % self.window_size[1]
    #     mod_pad_w = (self.window_size[2] - w % self.window_size[2]) % self.window_size[2]
    #     # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d), 'reflect')
    #     return x

    def check_image_size(self, x):
        _, _, d, h, w = x.size()
        # 动态计算每个维度需要的填充量
        mod_pad_d = (self.window_size[0] - d % self.window_size[0]) % self.window_size[0]
        mod_pad_h = (self.window_size[1] - h % self.window_size[1]) % self.window_size[1]
        mod_pad_w = (self.window_size[2] - w % self.window_size[2]) % self.window_size[2]

        # 在深度、高度、宽度的末尾进行填充
        # 参数顺序: (左, 右, 上, 下, 前, 后) - 对于5D张量 (batch, channel, depth, height, width)
        padded_x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d), 'reflect')
        return padded_x, (mod_pad_d, mod_pad_h, mod_pad_w)

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3], x.shape[4])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x):
        D, H, W = x.shape[2:]
        # x = self.check_image_size(x)
        x, pads = self.check_image_size(x)
        pad_d, pad_h, pad_w = pads

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            # x=self.upsample(x)
            x = self.conv_last(x)
        elif self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean
        result = x[:, :,
                 :(D + pad_d) * self.upscale,
                 :(H + pad_h) * self.upscale,
                 :(W + pad_w) * self.upscale]

        # 裁剪回原始尺寸的倍数
        return result[:, :,
               :D * self.upscale,
               :H * self.upscale,
               :W * self.upscale]
        # return x[:, :, :D*self.upscale, :H*self.upscale, :W*self.upscale]

    def flops(self):
        flops = 0
        D, H, W = self.patches_resolution
        flops += D * H * W * 3 * self.embed_dim * 27  # 3x3x3 conv
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += D * H * W * 3 * self.embed_dim * self.embed_dim
        if self.upsampler:
            flops += self.upsample.flops()
        return flops

