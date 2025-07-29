import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_, _assert
from torchvision.transforms import functional as TF
from timm.models.fx_features import register_notrace_function

import numpy as np
from einops import rearrange
from basicsr.utils.registry import ARCH_REGISTRY
from huggingface_hub import PyTorchModelHubMixin


class DFE(nn.Module):
    """ Dual Feature Extraction (3D version) """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features

        self.conv = nn.Sequential(
            nn.Conv3d(in_features, in_features // 5, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_features // 5, in_features // 5, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(in_features // 5, out_features, 1, 1, 0)
        )

        self.linear = nn.Conv3d(in_features, out_features, 1, 1, 0)

    def forward(self, x, x_size):
        B, L, C = x.shape
        D, H, W = x_size
        x = x.permute(0, 2, 1).contiguous().view(B, C, D, H, W)
        x = self.conv(x) * self.linear(x)
        x = x.view(B, -1, D * H * W).permute(0, 2, 1).contiguous()
        return x


class Mlp(nn.Module):
    """ MLP-based Feed-Forward Network (3D version) """

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
    B, D, H, W, C = x.shape
    x = x.view(B,
               D // window_size[0], window_size[0],
               H // window_size[1], window_size[1],
               W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_unpartition_3d(windows, window_size, D, H, W):
    B = int(windows.shape[0] / (D * H * W / (window_size[0] * window_size[1] * window_size[2])))
    x = windows.view(B,
                     D // window_size[0],
                     H // window_size[1],
                     W // window_size[2],
                     window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class DynamicPosBias3D(nn.Module):
    """ Dynamic Relative Position Bias (3D version) """

    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(3, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class SCC3D(nn.Module):
    """ Spatial-Channel Correlation (3D version) """

    def __init__(self, dim, base_win_size, window_size, num_heads, value_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // (2 * num_heads)
        if dim % (2 * num_heads) > 0:
            head_dim = head_dim + 1
        self.attn_dim = head_dim * 2 * num_heads
        self.qv = DFE(dim, self.attn_dim)
        self.proj = nn.Linear(self.attn_dim, dim)

        self.value_drop = nn.Dropout(value_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        min_d = min(self.window_size[0], base_win_size[0])
        min_h = min(self.window_size[1], base_win_size[1])
        min_w = min(self.window_size[2], base_win_size[2])
        self.base_win_size = (min_d, min_h, min_w)

        self.scale = head_dim
        self.spatial_linear = nn.Linear(
            self.window_size[0] * self.window_size[1] * self.window_size[2] //
            (self.base_win_size[0] * self.base_win_size[1] * self.base_win_size[2]), 1)

        self.ngram_window_partition = NGramWindowPartition3D(dim, window_size, 2, num_heads, shift_size=0)

        self.D_sp, self.H_sp, self.W_sp = self.window_size
        self.pos = DynamicPosBias3D(self.dim // 4, self.num_heads, residual=False)

    def spatial_linear_projection(self, x):
        B, num_h, L, C = x.shape
        D, H, W = self.window_size
        map_D, map_H, map_W = self.base_win_size

        x = x.view(B, num_h, map_D, D // map_D, map_H, H // map_H, map_W, W // map_W, C)
        x = x.permute(0, 1, 2, 4, 6, 8, 3, 5, 7).contiguous().view(B, num_h, map_D * map_H * map_W, C, -1)
        x = self.spatial_linear(x).view(B, num_h, map_D * map_H * map_W, C)
        return x

    def spatial_self_correlation(self, q, v):
        B, num_head, L, C = q.shape

        v = self.spatial_linear_projection(v)
        corr_map = (q @ v.transpose(-2, -1)) / self.scale

        position_bias_d = torch.arange(1 - self.D_sp, self.D_sp, device=v.device)
        position_bias_h = torch.arange(1 - self.H_sp, self.H_sp, device=v.device)
        position_bias_w = torch.arange(1 - self.W_sp, self.W_sp, device=v.device)
        biases = torch.stack(torch.meshgrid([position_bias_d, position_bias_h, position_bias_w]))
        rpe_biases = biases.flatten(1).transpose(0, 1).contiguous().float()
        pos = self.pos(rpe_biases)

        coords_d = torch.arange(self.D_sp, device=v.device)
        coords_h = torch.arange(self.H_sp, device=v.device)
        coords_w = torch.arange(self.W_sp, device=v.device)
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.D_sp - 1
        relative_coords[:, :, 1] += self.H_sp - 1
        relative_coords[:, :, 2] += self.W_sp - 1
        relative_coords[:, :, 0] *= (2 * self.H_sp - 1) * (2 * self.W_sp - 1)
        relative_coords[:, :, 1] *= 2 * self.W_sp - 1
        relative_position_index = relative_coords.sum(-1)
        relative_position_bias = pos[relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.base_win_size[0], self.window_size[0] // self.base_win_size[0],
            self.base_win_size[1], self.window_size[1] // self.base_win_size[1],
            self.base_win_size[2], self.window_size[2] // self.base_win_size[2], -1)
        relative_position_bias = relative_position_bias.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous().view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.base_win_size[0] * self.base_win_size[1] * self.base_win_size[2],
            self.num_heads, -1).mean(-1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        corr_map = corr_map + relative_position_bias.unsqueeze(0)

        v_drop = self.value_drop(v)
        x = (corr_map @ v_drop).permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        return x

    def channel_self_correlation(self, q, v):
        B, num_head, L, C = q.shape
        q = q.permute(0, 2, 1, 3).contiguous().view(B, L, num_head * C)
        v = v.permute(0, 2, 1, 3).contiguous().view(B, L, num_head * C)
        corr_map = (q.transpose(-2, -1) @ v) / L
        v_drop = self.value_drop(v)
        x = (corr_map @ v_drop.transpose(-2, -1)).permute(0, 2, 1).contiguous().view(B, L, -1)
        return x

    def forward(self, x):
        xB, xD, xH, xW, xC = x.shape
        qv = self.qv(x.view(xB, -1, xC), (xD, xH, xW)).view(xB, xD, xH, xW, xC)

        qv = self.ngram_window_partition(qv)
        qv = qv.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], xC)

        B, L, C = qv.shape
        qv = qv.view(B, L, 2, self.num_heads, C // (2 * self.num_heads)).permute(2, 0, 3, 1, 4).contiguous()
        q, v = qv[0], qv[1]

        x_spatial = self.spatial_self_correlation(q, v)
        x_spatial = x_spatial.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C // 2)
        x_spatial = window_unpartition_3d(x_spatial, self.window_size, xD, xH, xW)

        x_channel = self.channel_self_correlation(q, v)
        x_channel = x_channel.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C // 2)
        x_channel = window_unpartition_3d(x_channel, self.window_size, xD, xH, xW)

        x = torch.cat([x_spatial, x_channel], -1)
        x = self.proj_drop(self.proj(x))
        return x


class NGramWindowAttention3D(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias for NGram attention (3D version) """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        table_size = [2 * ws - 1 for ws in window_size]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(table_size[0] * table_size[1] * table_size[2], num_heads))

        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)
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


class NGramContext3D(nn.Module):
    """ NGram Context (3D version) """

    def __init__(self, dim, window_size, ngram, ngram_num_heads, padding_mode='seq_refl_win_pad'):
        super().__init__()
        _assert(padding_mode in ['seq_refl_win_pad', 'zero_pad'],
                "padding mode should be 'seq_refl_win_pad' or 'zero_pad'")

        self.dim = dim
        self.window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size, window_size)
        self.ngram = ngram
        self.padding_mode = padding_mode

        self.unigram_embed = nn.Conv3d(2, 1,
                                       kernel_size=self.window_size,
                                       stride=self.window_size, padding=0, groups=1)

        self.ngram_attn = NGramWindowAttention3D(dim=dim // 2, num_heads=ngram_num_heads,
                                                 window_size=(ngram, ngram, ngram))
        self.avg_pool = nn.AvgPool3d(ngram)
        self.merge = nn.Conv3d(dim, dim, 1, 1, 0)

    def seq_refl_win_pad(self, x, back=False):
        if self.ngram == 1: return x
        padding = (0, 0, 0, 0, self.ngram - 1, self.ngram - 1) if not back else (
            self.ngram - 1, self.ngram - 1, 0, 0, 0, 0)
        x = F.pad(x, padding)
        if self.padding_mode == 'zero_pad':
            return x
        if not back:
            x[:, :, -(self.ngram - 1):, :, :] = x[:, :, -2 * self.ngram + 1:-self.ngram, :, :]
        else:
            x[:, :, :self.ngram - 1, :, :] = x[:, :, self.ngram:2 * self.ngram - 1, :, :]
        return x

    def sliding_window_attention(self, unigram):
        # unfold in 3D
        slide = unigram.unfold(2, self.ngram, 1).unfold(3, self.ngram, 1).unfold(4, self.ngram, 1)
        B, C, D, H, W, dd, hh, ww = slide.shape
        slide = rearrange(slide, 'b c d h w dd hh ww -> b (d h w) (dd hh ww) c')

        context = self.ngram_attn(slide)
        context = context.view(B, D, H, W, -1)
        context = rearrange(context, 'b d h w c -> b c d h w')
        context = self.avg_pool(context)
        return context

    def forward(self, x):
        B, pd, ph, pw, D = x.size()
        x = rearrange(x, 'b d h w c -> b c d h w')
        x = x.contiguous().view(B * (D // 2), 2, pd, ph, pw)
        unigram = self.unigram_embed(x).view(B, D // 2, pd // self.window_size[0], ph // self.window_size[1],
                                             pw // self.window_size[2])

        unigram_forward_pad = self.seq_refl_win_pad(unigram, False)
        unigram_backward_pad = self.seq_refl_win_pad(unigram, True)

        context_forward = self.sliding_window_attention(unigram_forward_pad)
        context_backward = self.sliding_window_attention(unigram_backward_pad)

        context_bidirect = torch.cat([context_forward, context_backward], dim=1)
        context_bidirect = self.merge(context_bidirect)
        context_bidirect = rearrange(context_bidirect, 'b c d h w -> b d h w c')

        return context_bidirect.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3).contiguous()


class NGramWindowPartition3D(nn.Module):
    """ NGram Window Partition (3D version) """

    def __init__(self, dim, window_size, ngram, ngram_num_heads, shift_size=0):
        super().__init__()
        self.window_size = window_size if isinstance(window_size, tuple) else (window_size, window_size, window_size)
        self.ngram = ngram
        self.shift_size = shift_size

        self.ngram_context = NGramContext3D(dim, window_size, ngram, ngram_num_heads, padding_mode='seq_refl_win_pad')

    def forward(self, x):
        B, pd, ph, pw, D = x.size()
        wd, wh, ww = pd // self.window_size[0], ph // self.window_size[1], pw // self.window_size[2]
        _assert(0 not in [wd, wh, ww], "feature map size should be larger than window size!")

        context = self.ngram_context(x)

        windows = rearrange(x, 'b (d wd) (h wh) (w ww) c -> b d h w wd wh ww c',
                            wd=self.window_size[0], wh=self.window_size[1], ww=self.window_size[2])
        windows += context

        if self.shift_size > 0:
            x = rearrange(windows, 'b d h w wd wh ww c -> b (d wd) (h wh) (w ww) c')
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
            windows = rearrange(shifted_x, 'b (d wd) (h wh) (w ww) c -> b d h w wd wh ww c',
                                wd=self.window_size[0], wh=self.window_size[1], ww=self.window_size[2])

        windows = rearrange(windows, 'b d h w wd wh ww c -> (b d h w) wd wh ww c')
        return windows


class HierarchicalTransformerBlock3D(nn.Module):
    """ Hierarchical Transformer Block (3D version) """

    def __init__(self, dim, input_resolution, num_heads, base_win_size, window_size,
                 mlp_ratio=4., drop=0., value_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        if (window_size[0] > base_win_size[0]) and (window_size[1] > base_win_size[1]) and (
                window_size[2] > base_win_size[2]):
            assert window_size[0] % base_win_size[0] == 0, "window size should be divisible by base window size"
            assert window_size[1] % base_win_size[1] == 0, "window size should be divisible by base window size"
            assert window_size[2] % base_win_size[2] == 0, "window size should be divisible by base window size"

        self.norm1 = norm_layer(dim)
        self.correlation = SCC3D(
            dim, base_win_size=base_win_size, window_size=self.window_size, num_heads=num_heads,
            value_drop=value_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def check_image_size(self, x, win_size):
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        _, _, d, h, w = x.size()
        mod_pad_d = (win_size[0] - d % win_size[0]) % win_size[0]
        mod_pad_h = (win_size[1] - h % win_size[1]) % win_size[1]
        mod_pad_w = (win_size[2] - w % win_size[2]) % win_size[2]

        if mod_pad_d >= d or mod_pad_h >= h or mod_pad_w >= w:
            pad_d, pad_h, pad_w = d - 1, h - 1, w - 1
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), 'reflect')
        else:
            pad_d, pad_h, pad_w = 0, 0, 0

        mod_pad_d = mod_pad_d - pad_d
        mod_pad_h = mod_pad_h - pad_h
        mod_pad_w = mod_pad_w - pad_w

        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d), 'reflect')
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        return x

    def forward(self, x, x_size, win_size):
        D, H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = x.view(B, D, H, W, C)

        x = self.check_image_size(x, (win_size[0] * 2, win_size[1] * 2, win_size[2] * 2))
        _, D_pad, H_pad, W_pad, _ = x.shape

        x = self.correlation(x)
        x = x[:, :D, :H, :W, :].contiguous()
        x = x.view(B, D * H * W, C)
        x = self.norm1(x)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x


class PatchMerging3D(nn.Module):
    """ Patch Merging Layer (3D version) """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        D, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"
        assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0, f"x size ({D}*{H}*{W}) are not even."

        x = x.view(B, D, H, W, C)

        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = x.view(B, -1, 8 * C)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class BasicLayer3D(nn.Module):
    """ A basic Hierarchical Transformer layer for one stage (3D version) """

    def __init__(self, dim, input_resolution, depth, num_heads, base_win_size,
                 mlp_ratio=4., drop=0., value_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False, hier_win_ratios=[0.5, 1, 2]):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.win_ds = [int(base_win_size[0] * ratio) for ratio in hier_win_ratios]
        self.win_hs = [int(base_win_size[1] * ratio) for ratio in hier_win_ratios]
        self.win_ws = [int(base_win_size[2] * ratio) for ratio in hier_win_ratios]

        self.blocks = nn.ModuleList([
            HierarchicalTransformerBlock3D(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                base_win_size=base_win_size,
                window_size=(self.win_ds[i], self.win_hs[i], self.win_ws[i]),
                mlp_ratio=mlp_ratio,
                drop=drop,
                value_drop=value_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        i = 0
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size, (self.win_ds[i], self.win_hs[i], self.win_ws[i]))
            else:
                x = blk(x, x_size, (self.win_ds[i], self.win_hs[i], self.win_ws[i]))
            i = i + 1

        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RHTB3D(nn.Module):
    """Residual Hierarchical Transformer Block (3D version) """

    def __init__(self, dim, input_resolution, depth, num_heads, base_win_size,
                 mlp_ratio=4., drop=0., value_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False, img_size=32, patch_size=1,
                 resi_connection='1conv', hier_win_ratios=[0.5, 1, 2]):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer3D(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            base_win_size=base_win_size,
            mlp_ratio=mlp_ratio,
            drop=drop,
            value_drop=value_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            hier_win_ratios=hier_win_ratios)

        if resi_connection == '1conv':
            self.conv = nn.Conv3d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv3d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x


class PatchEmbed3D(nn.Module):
    """ Image to Patch Embedding (3D version) """

    def __init__(self, img_size=32, patch_size=1, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        patches_resolution = [img_size[0] // patch_size[0],
                              img_size[1] // patch_size[1],
                              img_size[2] // patch_size[2]]
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
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed3D(nn.Module):
    """ Image to Patch Unembedding (3D version) """

    def __init__(self, img_size=32, patch_size=1, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size, img_size) if isinstance(img_size, int) else img_size
        patch_size = (patch_size, patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        patches_resolution = [img_size[0] // patch_size[0],
                              img_size[1] // patch_size[1],
                              img_size[2] // patch_size[2]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, DHW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1], x_size[2])
        return x


class Upsample3D(nn.Sequential):
    """Upsample module (3D version) """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv3d(num_feat, 8 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv3d(num_feat, 27 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super().__init__(*m)


class UpsampleOneStep3D(nn.Sequential):
    """UpsampleOneStep module (3D version) """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv3d(num_feat, (scale ** 3) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super().__init__(*m)


class HiT_SNG3D(nn.Module, PyTorchModelHubMixin):
    """ HiT-SNG network (3D version) """

    def __init__(self, img_size=32, patch_size=1, in_chans=1,
                 embed_dim=64, depths=[2, 2, 2], num_heads=[4, 4, 4],
                 base_win_size=[4, 4, 4], mlp_ratio=2.,
                 drop_rate=0., value_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='pixelshuffledirect',
                 resi_connection='1conv', hier_win_ratios=[0.5, 1, 2], **kwargs):
        super().__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        self.mean = torch.zeros(1, 1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.base_win_size = base_win_size

        # 1. Shallow feature extraction
        self.conv_first = nn.Conv3d(num_in_ch, embed_dim, 3, 1, 1)

        # 2. Deep feature extraction
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Patch unembedding
        self.patch_unembed = PatchUnEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build Residual Hierarchical Transformer blocks (RHTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHTB3D(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1], patches_resolution[2]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                base_win_size=base_win_size,
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                value_drop=value_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                hier_win_ratios=hier_win_ratios
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # Last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv3d(embed_dim, embed_dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(embed_dim // 4, embed_dim, 3, 1, 1))

        # 3. High quality image reconstruction
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv3d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True))
            self.upsample = Upsample3D(upscale, num_feat)
            self.conv_last = nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep3D(
                upscale, embed_dim, num_out_ch,
                (patches_resolution[0], patches_resolution[1], patches_resolution[2]))
        elif self.upsampler == 'nearest+conv':
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv3d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv3d(num_feat, num_feat, 3, 1, 1)
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
        D, H, W = x.shape[2:5]
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean
        return x[:, :, :D * self.upscale, :H * self.upscale, :W * self.upscale]