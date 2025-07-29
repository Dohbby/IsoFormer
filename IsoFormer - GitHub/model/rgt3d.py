import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from torch.nn import functional as F

from timm.models.layers import DropPath, trunc_normal_
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

import math
import numpy as np

import random

from basicsr.utils.registry import ARCH_REGISTRY


def img2windows3d(img, D_sp, H_sp, W_sp):
    """
    Input: 3D Image (B, C, D, H, W)
    Output: Window Partition (B', N, C)
    """
    B, C, D, H, W = img.shape
    img_reshape = img.view(B, C, D // D_sp, D_sp, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().reshape(-1, D_sp * H_sp * W_sp, C)
    return img_perm


def windows2img3d(img_splits_dhw, D_sp, H_sp, W_sp, D, H, W):
    """
    Input: Window Partition (B', N, C)
    Output: 3D Image (B, D, H, W, C)
    """
    B = int(img_splits_dhw.shape[0] / (D * H * W / D_sp / H_sp / W_sp))

    img = img_splits_dhw.view(B, D // D_sp, H // H_sp, W // W_sp, D_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return img


class Gate3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)  # 3D DW Conv

    def forward(self, x, D, H, W):
        # Split
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = self.conv(self.norm(x2).transpose(1, 2).contiguous().view(B, C // 2, D, H, W))
        x2 = x2.flatten(2).transpose(-1, -2).contiguous()
        return x1 * x2


class MLP3D(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = Gate3D(hidden_features // 2)
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, D, H, W):
        """
        Input: x: (B, D*H*W, C), D, H, W
        Output: x: (B, D*H*W, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.sg(x, D, H, W)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias3D(nn.Module):
    """ Dynamic Relative Position Bias for 3D.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        residual (bool): If True, use residual strage to connect conv.
    """

    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(3, self.pos_dim)  # 3D position (D, H, W)

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


class WindowAttention3D(nn.Module):
    def __init__(self, dim, idx, split_size=[4, 4, 4], dim_out=None, num_heads=6,
                 attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 3D window partitioning
        if idx == 0:
            D_sp, H_sp, W_sp = self.split_size[0], self.split_size[1], self.split_size[2]
        elif idx == 1:
            H_sp, D_sp, W_sp = self.split_size[0], self.split_size[1], self.split_size[2]
        elif idx == 2:
            W_sp, D_sp, H_sp = self.split_size[0], self.split_size[1], self.split_size[2]
        else:
            print("ERROR MODE", idx)
            exit(0)

        self.D_sp = D_sp
        self.H_sp = H_sp
        self.W_sp = W_sp

        if self.position_bias:
            self.pos = DynamicPosBias3D(self.dim // 4, self.num_heads, residual=False)

            # generate mother-set for 3D
            position_bias_d = torch.arange(1 - self.D_sp, self.D_sp)
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_d, position_bias_h, position_bias_w]))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer('rpe_biases', biases)

            # get pair-wise relative position index for each token inside the 3D window
            coords_d = torch.arange(self.D_sp)
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
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
            self.register_buffer('relative_position_index', relative_position_index)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2win3d(self, x, D, H, W):
        B, N, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, D, H, W)
        x = img2windows3d(x, self.D_sp, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.D_sp * self.H_sp * self.W_sp, self.num_heads, C // self.num_heads)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv, D, H, W, mask=None):
        """
        Input: qkv: (B, 3*L, C), D, H, W, mask: (B, N, N), N is the window size
        Output: x (B, D, H, W, C)
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        B, L, C = q.shape
        assert L == D * H * W, "flatten 3D tokens has wrong size"

        # partition the q,k,v, image to window
        q = self.im2win3d(q, D, H, W)
        k = self.im2win3d(k, D, H, W)
        v = self.im2win3d(v, D, H, W)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N

        # calculate drpe
        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            # select position bias
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.D_sp * self.H_sp * self.W_sp, self.D_sp * self.H_sp * self.W_sp, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        N = attn.shape[3]

        # use mask for shift window
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.D_sp * self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        # merge the window, window to image
        x = windows2img3d(x, self.D_sp, self.H_sp, self.W_sp, D, H, W)  # B D H W C

        return x


class L_SA3D(nn.Module):
    def __init__(self, dim, num_heads,
                 split_size=[2, 2, 4], shift_size=[1, 1, 2], qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., idx=0, reso=32, rs_id=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.idx = idx
        self.rs_id = rs_id
        self.patches_resolution = reso
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        assert 0 <= self.shift_size[0] < self.split_size[0], "shift_size must in 0-split_size0"
        assert 0 <= self.shift_size[1] < self.split_size[1], "shift_size must in 0-split_size1"
        assert 0 <= self.shift_size[2] < self.split_size[2], "shift_size must in 0-split_size2"

        self.branch_num = 3  # For 3D we have 3 branches

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        self.attns = nn.ModuleList([
            WindowAttention3D(
                dim // 3, idx=i,
                split_size=split_size, num_heads=num_heads // 3, dim_out=dim // 3,
                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
            for i in range(self.branch_num)])

        if (self.rs_id % 2 == 0 and self.idx > 0 and (self.idx - 2) % 4 == 0) or (
                self.rs_id % 2 != 0 and self.idx % 4 == 0):
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution, self.patches_resolution)
            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
            self.register_buffer("attn_mask_2", attn_mask[2])
        else:
            attn_mask = None
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)
            self.register_buffer("attn_mask_2", None)

        self.get_v = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)  # 3D DW Conv

    def calculate_mask(self, D, H, W):
        # calculate attention mask for 3D windows
        img_mask_0 = torch.zeros((1, D, H, W, 1))  # 1 D H W 1 idx=0
        img_mask_1 = torch.zeros((1, D, H, W, 1))  # 1 D H W 1 idx=1
        img_mask_2 = torch.zeros((1, D, H, W, 1))  # 1 D H W 1 idx=2

        # For each dimension (D, H, W)
        d_slices_0 = (slice(0, -self.split_size[0]),
                      slice(-self.split_size[0], -self.shift_size[0]),
                      slice(-self.shift_size[0], None))
        h_slices_0 = (slice(0, -self.split_size[1]),
                      slice(-self.split_size[1], -self.shift_size[1]),
                      slice(-self.shift_size[1], None))
        w_slices_0 = (slice(0, -self.split_size[2]),
                      slice(-self.split_size[2], -self.shift_size[2]),
                      slice(-self.shift_size[2], None))

        d_slices_1 = (slice(0, -self.split_size[1]),
                      slice(-self.split_size[1], -self.shift_size[1]),
                      slice(-self.shift_size[1], None))
        h_slices_1 = (slice(0, -self.split_size[2]),
                      slice(-self.split_size[2], -self.shift_size[2]),
                      slice(-self.shift_size[2], None))
        w_slices_1 = (slice(0, -self.split_size[0]),
                      slice(-self.split_size[0], -self.shift_size[0]),
                      slice(-self.shift_size[0], None))

        d_slices_2 = (slice(0, -self.split_size[2]),
                      slice(-self.split_size[2], -self.shift_size[2]),
                      slice(-self.shift_size[2], None))
        h_slices_2 = (slice(0, -self.split_size[0]),
                      slice(-self.split_size[0], -self.shift_size[0]),
                      slice(-self.shift_size[0], None))
        w_slices_2 = (slice(0, -self.split_size[1]),
                      slice(-self.split_size[1], -self.shift_size[1]),
                      slice(-self.shift_size[1], None))

        cnt = 0
        for d in d_slices_0:
            for h in h_slices_0:
                for w in w_slices_0:
                    img_mask_0[:, d, h, w, :] = cnt
                    cnt += 1
        cnt = 0
        for d in d_slices_1:
            for h in h_slices_1:
                for w in w_slices_1:
                    img_mask_1[:, d, h, w, :] = cnt
                    cnt += 1
        cnt = 0
        for d in d_slices_2:
            for h in h_slices_2:
                for w in w_slices_2:
                    img_mask_2[:, d, h, w, :] = cnt
                    cnt += 1

        # calculate mask for D-Shift
        img_mask_0 = img_mask_0.view(1, D // self.split_size[0], self.split_size[0],
                                     H // self.split_size[1], self.split_size[1],
                                     W // self.split_size[2], self.split_size[2], 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1,
                                                                                  self.split_size[0] * self.split_size[
                                                                                      1] * self.split_size[2], 1)
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1] * self.split_size[2])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))

        # calculate mask for H-Shift
        img_mask_1 = img_mask_1.view(1, D // self.split_size[1], self.split_size[1],
                                     H // self.split_size[2], self.split_size[2],
                                     W // self.split_size[0], self.split_size[0], 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1,
                                                                                  self.split_size[1] * self.split_size[
                                                                                      2] * self.split_size[0], 1)
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[2] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))

        # calculate mask for W-Shift
        img_mask_2 = img_mask_2.view(1, D // self.split_size[2], self.split_size[2],
                                     H // self.split_size[0], self.split_size[0],
                                     W // self.split_size[1], self.split_size[1], 1)
        img_mask_2 = img_mask_2.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1,
                                                                                  self.split_size[2] * self.split_size[
                                                                                      0] * self.split_size[1], 1)
        mask_windows_2 = img_mask_2.view(-1, self.split_size[2] * self.split_size[0] * self.split_size[1])
        attn_mask_2 = mask_windows_2.unsqueeze(1) - mask_windows_2.unsqueeze(2)
        attn_mask_2 = attn_mask_2.masked_fill(attn_mask_2 != 0, float(-100.0)).masked_fill(attn_mask_2 == 0, float(0.0))

        return attn_mask_0, attn_mask_1, attn_mask_2

    def forward(self, x, D, H, W):
        """
        Input: x: (B, D*H*W, C), x_size: (D, H, W)
        Output: x: (B, D*H*W, C)
        """
        B, L, C = x.shape
        assert L == D * H * W, "flatten 3D tokens has wrong size"

        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)  # 3, B, DHW, C
        # v without partition
        v = qkv[2].transpose(-2, -1).contiguous().view(B, C, D, H, W)

        max_split_size = max(self.split_size[0], self.split_size[1], self.split_size[2])
        pad_d = pad_h = pad_w = 0
        pad_d_back = (max_split_size - D % max_split_size) % max_split_size
        pad_h_back = (max_split_size - H % max_split_size) % max_split_size
        pad_w_back = (max_split_size - W % max_split_size) % max_split_size

        qkv = qkv.reshape(3 * B, D, H, W, C).permute(0, 4, 1, 2, 3)  # 3B C D H W
        qkv = F.pad(qkv, (pad_w, pad_w_back, pad_h, pad_h_back, pad_d, pad_d_back)).reshape(3, B, C, -1).transpose(-2,
                                                                                                                   -1)
        _D = pad_d_back + D
        _H = pad_h_back + H
        _W = pad_w_back + W
        _L = _D * _H * _W

        if (self.rs_id % 2 == 0 and self.idx > 0 and (self.idx - 2) % 4 == 0) or (
                self.rs_id % 2 != 0 and self.idx % 4 == 0):
            qkv = qkv.view(3, B, _D, _H, _W, C)
            # D-Shift
            qkv_0 = torch.roll(qkv[:, :, :, :, :, :C // 3],
                               shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(2, 3, 4))
            qkv_0 = qkv_0.view(3, B, _L, C // 3)
            # H-Shift
            qkv_1 = torch.roll(qkv[:, :, :, :, :, C // 3:2 * C // 3],
                               shifts=(-self.shift_size[1], -self.shift_size[2], -self.shift_size[0]), dims=(2, 3, 4))
            qkv_1 = qkv_1.view(3, B, _L, C // 3)
            # W-Shift
            qkv_2 = torch.roll(qkv[:, :, :, :, :, 2 * C // 3:],
                               shifts=(-self.shift_size[2], -self.shift_size[0], -self.shift_size[1]), dims=(2, 3, 4))
            qkv_2 = qkv_2.view(3, B, _L, C // 3)

            if self.patches_resolution != _D or self.patches_resolution != _H or self.patches_resolution != _W:
                mask_tmp = self.calculate_mask(_D, _H, _W)
                # D-Rwin
                x0_shift = self.attns[0](qkv_0, _D, _H, _W, mask=mask_tmp[0].to(x.device))
                # H-Rwin
                x1_shift = self.attns[1](qkv_1, _D, _H, _W, mask=mask_tmp[1].to(x.device))
                # W-Rwin
                x2_shift = self.attns[2](qkv_2, _D, _H, _W, mask=mask_tmp[2].to(x.device))
            else:
                # D-Rwin
                x0_shift = self.attns[0](qkv_0, _D, _H, _W, mask=self.attn_mask_0)
                # H-Rwin
                x1_shift = self.attns[1](qkv_1, _D, _H, _W, mask=self.attn_mask_1)
                # W-Rwin
                x2_shift = self.attns[2](qkv_2, _D, _H, _W, mask=self.attn_mask_2)

            x0 = torch.roll(x0_shift, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                            dims=(1, 2, 3))
            x1 = torch.roll(x1_shift, shifts=(self.shift_size[1], self.shift_size[2], self.shift_size[0]),
                            dims=(1, 2, 3))
            x2 = torch.roll(x2_shift, shifts=(self.shift_size[2], self.shift_size[0], self.shift_size[1]),
                            dims=(1, 2, 3))

            x0 = x0[:, :D, :H, :W, :].reshape(B, L, C // 3)
            x1 = x1[:, :D, :H, :W, :].reshape(B, L, C // 3)
            x2 = x2[:, :D, :H, :W, :].reshape(B, L, C // 3)

            # Concat
            attened_x = torch.cat([x0, x1, x2], dim=2)
        else:
            # D-Rwin
            x0 = self.attns[0](qkv[:, :, :, :C // 3], _D, _H, _W)[:, :D, :H, :W, :].reshape(B, L, C // 3)
            # H-Rwin
            x1 = self.attns[1](qkv[:, :, :, C // 3:2 * C // 3], _D, _H, _W)[:, :D, :H, :W, :].reshape(B, L, C // 3)
            # W-Rwin
            x2 = self.attns[2](qkv[:, :, :, 2 * C // 3:], _D, _H, _W)[:, :D, :H, :W, :].reshape(B, L, C // 3)
            # Concat
            attened_x = torch.cat([x0, x1, x2], dim=2)

        # mix
        lcm = self.get_v(v)
        lcm = lcm.permute(0, 2, 3, 4, 1).contiguous().view(B, L, C)

        x = attened_x + lcm

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class RG_SA3D(nn.Module):
    """
    3D Recursive-Generalization Self-Attention (RG-SA).
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        c_ratio (float): channel adjustment factor.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., c_ratio=0.5):
        super(RG_SA3D, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.cr = int(dim * c_ratio)  # scaled channel dimension
        self.scale = qk_scale or (head_dim * c_ratio) ** -0.5

        # RGM for 3D
        self.reduction1 = nn.Conv3d(dim, dim, kernel_size=4, stride=4, groups=dim)
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv = nn.Conv3d(dim, self.cr, kernel_size=1, stride=1)
        self.norm_act = nn.Sequential(
            nn.LayerNorm(self.cr),
            nn.GELU())

        # CA
        self.q = nn.Linear(dim, self.cr, bias=qkv_bias)
        self.k = nn.Linear(self.cr, self.cr, bias=qkv_bias)
        self.v = nn.Linear(self.cr, dim, bias=qkv_bias)

        # CPE for 3D
        self.cpe = nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, D, H, W):
        B, N, C = x.shape
        _scale = 1

        # reduction
        _x = x.permute(0, 2, 1).reshape(B, C, D, H, W).contiguous()

        if self.training:
            _time = max(int(math.log(D // 4, 4)), int(math.log(H // 4, 4)), int(math.log(W // 4, 4)))
        else:
            _time = max(int(math.log(D // 16, 4)), int(math.log(H // 16, 4)), int(math.log(W // 16, 4)))
            if _time < 2: _time = 2  # testing _time must equal or larger than training _time (2)

        _scale = 4 ** _time

        # Recursion xT for 3D
        for _ in range(_time):
            _x = self.reduction1(_x)

        _x = self.conv(self.dwconv(_x)).reshape(B, self.cr, -1).permute(0, 2, 1).contiguous()  # shape=(B, N', C')
        _x = self.norm_act(_x)

        # q, k, v, where q_shape=(B, N, C'), k_shape=(B, N', C'), v_shape=(B, N', C)
        q = self.q(x).reshape(B, N, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
        k = self.k(_x).reshape(B, -1, self.num_heads, int(self.cr / self.num_heads)).permute(0, 2, 1, 3)
        v = self.v(_x).reshape(B, -1, self.num_heads, int(C / self.num_heads)).permute(0, 2, 1, 3)

        # cross-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # CPE for 3D
        v = v + self.cpe(v.transpose(1, 2).reshape(B, -1, C).transpose(1, 2).contiguous()
                         .view(B, C, D // _scale, H // _scale, W // _scale))
        v = v.view(B, C, -1).view(B, self.num_heads, int(C / self.num_heads), -1).transpose(-1, -2)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block3D(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, idx=0,
                 rs_id=0, split_size=[2, 2, 4], shift_size=[1, 1, 2], reso=32, c_ratio=0.5, layerscale_value=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if idx % 2 == 0:
            self.attn = L_SA3D(
                dim, split_size=split_size, shift_size=shift_size, num_heads=num_heads, qkv_bias=qkv_bias,
                qk_scale=qk_scale, attn_drop=attn_drop, drop=drop, idx=idx, reso=reso, rs_id=rs_id
            )
        else:
            self.attn = RG_SA3D(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                proj_drop=drop, c_ratio=c_ratio
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP3D(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer)
        self.norm2 = norm_layer(dim)

        # HAI
        self.gamma = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x, x_size):
        D, H, W = x_size
        res = x

        x = x + self.drop_path(self.attn(self.norm1(x), D, H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), D, H, W))

        # HAI
        x = x + (res * self.gamma)

        return x


class ResidualGroup3D(nn.Module):
    def __init__(self, dim, reso, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_paths=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 depth=2, use_chk=False, resi_connection='1conv', rs_id=0, split_size=[4, 4, 8], c_ratio=0.5):
        super().__init__()
        self.use_chk = use_chk
        self.reso = reso

        self.blocks = nn.ModuleList([
            Block3D(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_paths[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                idx=i,
                rs_id=rs_id,
                split_size=split_size,
                shift_size=[split_size[0] // 2, split_size[1] // 2, split_size[2] // 2],
                c_ratio=c_ratio,
            ) for i in range(depth)])

        if resi_connection == '1conv':
            self.conv = nn.Conv3d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv3d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, x_size):
        """
        Input: x: (B, D*H*W, C), x_size: (D, H, W)
        Output: x: (B, D*H*W, C)
        """
        D, H, W = x_size
        res = x
        for blk in self.blocks:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        x = rearrange(x, "b (d h w) c -> b c d h w", d=D, h=H, w=W)
        x = self.conv(x)
        x = rearrange(x, "b c d h w -> b (d h w) c")
        x = res + x

        return x


class Upsample3D(nn.Sequential):
    """3D Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

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
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample3D, self).__init__(*m)


@ARCH_REGISTRY.register()
class rgt(nn.Module):
    def __init__(self,
                 img_size=32,
                 in_chans=1,
                 embed_dim=96,
                 depth=[2,2,2,2,2,2,2,2],
                 num_heads=[2,2,2,2,2,2,2,2],
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_chk=False,
                 upscale=1,
                 img_range=1.,
                 resi_connection='1conv',
                 split_size=[4, 4, 4],
                 c_ratio=0.5,
                 **kwargs):
        super().__init__()

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1, 1)
        self.upscale = upscale

        # ------------------------- 1, Shallow Feature Extraction ------------------------- #
        self.conv_first = nn.Conv3d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, Deep Feature Extraction ------------------------- #
        self.num_layers = len(depth)
        self.use_chk = use_chk
        self.num_features = self.embed_dim = embed_dim
        heads = num_heads

        self.before_RG = nn.Sequential(
            Rearrange('b c d h w -> b (d h w) c'),
            nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = ResidualGroup3D(
                dim=embed_dim,
                num_heads=heads[i],
                reso=img_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_paths=dpr[sum(depth[:i]):sum(depth[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                depth=depth[i],
                use_chk=use_chk,
                resi_connection=resi_connection,
                rs_id=i,
                split_size=split_size,
                c_ratio=c_ratio
            )
            self.layers.append(layer)

        self.norm = norm_layer(curr_dim)
        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv3d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3, Reconstruction ------------------------- #
        self.conv_before_upsample = nn.Sequential(
            nn.Conv3d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsample = Upsample3D(upscale, num_feat)
        self.conv_last = nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d, nn.GroupNorm, nn.InstanceNorm3d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        _, _, D, H, W = x.shape
        x_size = [D, H, W]
        x = self.before_RG(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        x = rearrange(x, "b (d h w) c -> b c d h w", d=D, h=H, w=W)
        return x

    def forward(self, x):
        """
        Input: x: (B, C, D, H, W)
        """
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x
        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean
        return x