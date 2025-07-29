import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_3tuple, trunc_normal_
from einops import rearrange

NEG_INF = -1000000


###############------DropOut------###############
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


###############------DropOut------###############

###############------Channel_Attention_Block(CAB)------###############
class ChannelAttention3D(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention3D, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB3D(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super(CAB3D, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv3d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv3d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention3D(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


###############------Channel_Attention_Block(CAB)------###############

###############------Multi_Layer_Perceptron------###############
class Mlp3D(nn.Module):
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


###############------Multi_Layer_Perceptron------###############

###################---------Window_Attention_D---------###################
class WindowAttention3D_D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww, Wd
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias table for 3D
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, rpi, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
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


###################---------Window_Attention_D---------###################

###################---------Window_Attention_S---------###################
class WindowAttention3D_S(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww, Wd
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias table for 3D
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, rpi, mask=None, sp_mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.window_size[0] * self.window_size[1] * self.window_size[2], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if sp_mask is not None:
            nP = sp_mask.shape[0]
            attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + sp_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
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


###################---------Window_Attention_S---------###################

###############------3D Window Partition------###############
def window_partition3d(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B,
               D // window_size[0], window_size[0],
               H // window_size[1], window_size[1],
               W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse3d(windows, window_size, D, H, W):
    B = int(windows.shape[0] / (D * H * W / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B,
                     D // window_size[0],
                     H // window_size[1],
                     W // window_size[2],
                     window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


###############------3D Window Partition------###############

###################---------Hybrid_Attention_Block_Dense(HAB_D)---------###################
class HAB3D_D(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=16,
                 shift_size=0,
                 interval=0,
                 triangular_flag=0,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.interval = interval
        self.mlp_ratio = mlp_ratio
        self.triangular_flag = triangular_flag

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D_D(
            dim,
            window_size=to_3tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.conv_scale = conv_scale
        self.conv_block = CAB3D(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp3D(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi_sa, attn_mask, triangular_masks):
        D, H, W = x_size
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # CAB
        conv_x = self.conv_block(x.permute(0, 4, 1, 2, 3))
        conv_x = conv_x.permute(0, 2, 3, 4, 1).contiguous().view(B, D * H * W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
            if self.shift_size == 8:
                attn_mask = attn_mask[0]
            elif self.shift_size == 16:
                attn_mask = attn_mask[1]
            elif self.shift_size == 24:
                attn_mask = attn_mask[2]
        else:
            shifted_x = x
            attn_mask = None

        # Window partition
        x_windows = window_partition3d(shifted_x, to_3tuple(self.window_size))
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size, C)

        # Window attention
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x = window_reverse3d(attn_windows, to_3tuple(self.window_size), D, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        else:
            attn_x = shifted_x

        attn_x = attn_x.view(B, D * H * W, C)

        # Combine with shortcut and CAB
        x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


###################---------Hybrid_Attention_Block_Dense(HAB_D)---------###################

###################---------Hybrid_Attention_Block_Sparse(HAB_S)---------###################
class HAB3D_S(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=16,
                 shift_size=0,
                 interval=2,
                 triangular_flag=0,
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.interval = interval
        self.mlp_ratio = mlp_ratio
        self.triangular_flag = triangular_flag

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D_S(
            dim,
            window_size=to_3tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.conv_scale = conv_scale
        self.conv_block = CAB3D(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp3D(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi_sa, attn_mask, triangular_masks):
        D, H, W = x_size
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # CAB
        conv_x = self.conv_block(x.permute(0, 4, 1, 2, 3))
        conv_x = conv_x.permute(0, 2, 3, 4, 1).contiguous().view(B, D * H * W, C)

        # Padding for sparse attention
        size_par = self.interval
        pad_d = (size_par - D % size_par) % size_par
        pad_h = (size_par - H % size_par) % size_par
        pad_w = (size_par - W % size_par) % size_par

        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        _, Dd, Hd, Wd, _ = x.shape

        # Mask for padding
        mask = torch.zeros((1, Dd, Hd, Wd, 1), device=x.device)
        if pad_d > 0:
            mask[:, -pad_d:, :, :, :] = -1
        if pad_h > 0:
            mask[:, :, -pad_h:, :, :] = -1
        if pad_w > 0:
            mask[:, :, :, -pad_w:, :] = -1

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
            if self.shift_size == 8:
                attn_mask = attn_mask[0]
            elif self.shift_size == 16:
                attn_mask = attn_mask[1]
            elif self.shift_size == 24:
                attn_mask = attn_mask[2]
        else:
            shifted_x = x
            attn_mask = None

        # Sparse attention
        I, Gd, Gh, Gw = self.interval, Dd // self.interval, Hd // self.interval, Wd // self.interval
        shifted_sparse_x = shifted_x.reshape(B, Gd, I, Gh, I, Gw, I, C).permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
        shifted_sparse_x = shifted_sparse_x.reshape(B * I * I * I, Gd, Gh, Gw, C)
        nP = I ** 3  # number of partitioning groups

        # Sparse attention mask
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            mask = mask.reshape(1, Gd, I, Gh, I, Gw, I, 1).permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
            mask = mask.reshape(nP, 1, Gd * Gh * Gw)
            attn_mask_sp = torch.zeros((nP, Gd * Gh * Gw, Gd * Gh * Gw), device=x.device)
            attn_mask_sp = attn_mask_sp.masked_fill(mask < 0, NEG_INF)
        else:
            attn_mask_sp = None

        # Window partition for sparse attention
        x_windows = window_partition3d(shifted_sparse_x, to_3tuple(self.window_size))
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size, C)

        # Window attention
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask, sp_mask=attn_mask_sp)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_sparse_x = window_reverse3d(attn_windows, to_3tuple(self.window_size), Gd, Gh, Gw)

        # Reverse sparse attention
        shifted_sparse_x = shifted_sparse_x.reshape(B, I, I, I, Gd, Gh, Gw, C).permute(0, 4, 1, 5, 2, 6, 3,
                                                                                       7).contiguous()
        shifted_x = shifted_sparse_x.reshape(B, Dd, Hd, Wd, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        else:
            attn_x = shifted_x

        # Remove padding
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            attn_x = attn_x[:, :D, :H, :W, :].contiguous()
        attn_x = attn_x.view(B, D * H * W, C)

        # Combine with shortcut and CAB
        x = shortcut + self.drop_path(attn_x) + conv_x * self.conv_scale

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


###################---------Hybrid_Attention_Block_Sparse(HAB_S)---------###################

###################---------Overlapping_Cross-attention_Block(OCAB)---------###################
class OCAB3D(nn.Module):
    def __init__(self, dim,
                 input_resolution,
                 window_size,
                 overlap_ratio,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 mlp_ratio=2,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.mlp_ratio = mlp_ratio

        # Calculate padding and stride for manual unfolding
        self.stride = window_size
        self.padding = (self.overlap_win_size - window_size) // 2

        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Relative position bias table for 3D
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) *
                        (window_size + self.overlap_win_size - 1) *
                        (window_size + self.overlap_win_size - 1), num_heads))

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp3D(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU)

    def forward(self, x, x_size, rpi):
        D, H, W = x_size
        B, L, C = x.shape
        assert L == D * H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # Linear projection
        qkv = self.qkv(x).reshape(B, D, H, W, 3, C).permute(4, 0, 5, 1, 2, 3)
        q = qkv[0].permute(0, 2, 3, 4, 1)  # B, D, H, W, C
        kv = torch.cat((qkv[1], qkv[2]), dim=1)  # B, 2*C, D, H, W

        # Window partition for query
        q_windows = window_partition3d(q, to_3tuple(self.window_size))
        q_windows = q_windows.view(-1, self.window_size * self.window_size * self.window_size, C)

        # Manual 3D unfolding for key and value
        kv = kv.permute(0, 2, 3, 4, 1).reshape(B, D * H * W, 2 * C)
        kv = kv.permute(0, 2, 1).reshape(B, 2 * C, D, H, W)

        # Pad the input
        kv = F.pad(kv, [self.padding] * 6)

        # Unfold in D dimension
        kv = kv.unfold(2, self.overlap_win_size, self.stride)  # B, 2*C, nD, H, W, wD
        # Unfold in H dimension
        kv = kv.unfold(3, self.overlap_win_size, self.stride)  # B, 2*C, nD, nH, W, wD, wH
        # Unfold in W dimension
        kv = kv.unfold(4, self.overlap_win_size, self.stride)  # B, 2*C, nD, nH, nW, wD, wH, wW

        # Reshape to get windows
        kv = kv.reshape(B, 2 * C, -1, self.overlap_win_size * self.overlap_win_size * self.overlap_win_size)
        kv = kv.permute(0, 2, 3, 1)  # B, nW, wD*wH*wW, 2*C

        # Rearrange to separate key and value
        kv_windows = rearrange(kv, 'b nw whww (nc ch) -> nc (b nw) (whww) ch',
                               nc=2, ch=C, whww=self.overlap_win_size * self.overlap_win_size * self.overlap_win_size)
        k_windows, v_windows = kv_windows[0], kv_windows[1]

        # Reshape for attention
        B_, Nq, _ = q_windows.shape
        _, N, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(B_, Nq, self.num_heads, d).permute(0, 2, 1, 3)
        k = k_windows.reshape(B_, N, self.num_heads, d).permute(0, 2, 1, 3)
        v = v_windows.reshape(B_, N, self.num_heads, d).permute(0, 2, 1, 3)

        # Attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size * self.window_size,
            self.overlap_win_size * self.overlap_win_size * self.overlap_win_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn_windows = (attn @ v).transpose(1, 2).reshape(B_, Nq, self.dim)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, C)
        x = window_reverse3d(attn_windows, to_3tuple(self.window_size), D, H, W)
        x = x.view(B, D * H * W, C)
        x = self.proj(x) + shortcut

        # FFN
        x = x + self.mlp(self.norm2(x))

        return x


###################---------Overlapping_Cross-attention_Block(OCAB)---------###################

###################---------Patch_Merging---------###################
class PatchMerging3D(nn.Module):
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

        x0 = x[:, 0::2, 0::2, 0::2, :]  # B D/2 H/2 W/2 C
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B D/2 H/2 W/2 8*C
        x = x.view(B, -1, 8 * C)  # B D/2*H/2*W/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


###################---------Patch_Merging---------###################

###################---------Atten_Blocks(HAB_D+HAB_S+OCAB)---------###################
class AttenBlocks3D(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 sparse_flag,
                 depth,
                 num_heads,
                 window_size,
                 shift_size,
                 interval,
                 compress_ratio,
                 squeeze_factor,
                 conv_scale,
                 overlap_ratio,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.sparse_flag = sparse_flag
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # HAB blocks
        if not sparse_flag:
            self.blocks = nn.ModuleList([HAB3D_D(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size[i],
                interval=interval,
                triangular_flag=0 if (i % 2 == 0) else 1,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([HAB3D_S(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=shift_size[i],
                interval=interval,
                triangular_flag=0 if (i % 2 == 0) else 1,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)])

        # OCAB
        self.overlap_attn = OCAB3D(
            dim=dim,
            input_resolution=input_resolution,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer)

        # Patch merging
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, params):
        for blk in self.blocks:
            x = blk(x, x_size, params['rpi_sa'], params['attn_mask'], params['triangular_masks'])

        x = self.overlap_attn(x, x_size, params['rpi_oca'])

        if self.downsample is not None:
            x = self.downsample(x)
        return x


###################---------Atten_Blocks(HAB+OCAB)---------###################

###################---------Residual_Hybrid_Attention_Group(RHAG)---------###################
class RHAG3D(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 sparse_flag,
                 depth,
                 num_heads,
                 window_size,
                 shift_size,
                 interval,
                 compress_ratio,
                 squeeze_factor,
                 conv_scale,
                 overlap_ratio,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=32,
                 patch_size=1,
                 resi_connection='1conv'):
        super(RHAG3D, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.sparse_flag = sparse_flag

        self.residual_group = AttenBlocks3D(
            dim=dim,
            input_resolution=input_resolution,
            sparse_flag=sparse_flag,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            interval=interval,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            conv_scale=conv_scale,
            overlap_ratio=overlap_ratio,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv3d(dim, dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv = nn.Identity()

        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None)
        self.patch_unembed = PatchUnEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size, params):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))) + x


###################---------Residual_Hybrid_Attention_Group(RHAG)---------###################

###################---------Patch_Embedding------###################
class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=32, patch_size=1, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
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
        x = x.flatten(2).transpose(1, 2)  # B D*H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x


###################---------Patch_Embedding------###################

###################---------Patch_Unembedding------###################
class PatchUnEmbed3D(nn.Module):
    def __init__(self, img_size=32, patch_size=1, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
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
        D, H, W = x_size
        x = x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, D, H, W)
        return x


###################---------Patch_Unembedding------###################

###################---------Upsample------###################
class Upsample3D(nn.Sequential):
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


###################---------Upsample------###################

#######################---------Composite_Fusion_Attention_Transformer(CFAT)---------#######################
@ARCH_REGISTRY.register()
class CFAT3D(nn.Module):
    def __init__(self,
                 img_size=32,
                 patch_size=1,
                 in_chans=1,
                 embed_dim=64,
                 depths=(2,2,2,2,2),
                 num_heads=(2,2,2,2,2),
                 window_size=4,
                 shift_size=(0, 0, 8, 8, 16, 16, 24, 24),
                 interval=(0, 2, 0, 2, 0),
                 compress_ratio=3,
                 squeeze_factor=30,
                 conv_scale=0.01,
                 overlap_ratio=0.5,
                 mlp_ratio=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=1,
                 img_range=1.,
                 upsampler='pixelshuffle',
                 resi_connection='1conv',
                 **kwargs):
        super(CFAT3D, self).__init__()

        self.window_size = window_size
        self.shift_size = shift_size
        self.overlap_ratio = overlap_ratio
        self.upscale = upscale
        self.upsampler = upsampler
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64

        # Mean operation
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1, 1)

        # Relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        relative_position_index_OCA = self.calculate_rpi_oca()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)
        self.register_buffer('relative_position_index_OCA', relative_position_index_OCA)

        # Shallow feature extraction
        self.conv_first = nn.Conv3d(num_in_ch, embed_dim, 3, 1, 1)

        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Patch unembedding
        self.patch_unembed = PatchUnEmbed3D(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # Absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Residual Hybrid Attention Groups
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RHAG3D(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1], patches_resolution[2]),
                sparse_flag=0 if (i_layer % 2 == 0) else 1,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                shift_size=shift_size,
                interval=interval[i_layer],
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                overlap_ratio=overlap_ratio,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging3D if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)

        # Reconstruction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == 'identity':
            self.conv_after_body = nn.Identity()

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv3d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True))
            self.upsample = Upsample3D(upscale, num_feat)
            self.conv_last = nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def calculate_rpi_sa(self):
        # Calculate relative position index for self-attention
        coords_d = torch.arange(self.window_size)
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w]))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 2] += self.window_size - 1
        relative_coords[:, :, 0] *= (2 * self.window_size - 1) * (2 * self.window_size - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        return relative_position_index

    def calculate_rpi_oca(self):
        # Calculate relative position index for overlapping cross-attention
        window_size_ori = self.window_size
        window_size_ext = self.window_size + int(self.overlap_ratio * self.window_size)

        coords_d = torch.arange(window_size_ori)
        coords_h = torch.arange(window_size_ori)
        coords_w = torch.arange(window_size_ori)
        coords_ori = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w]))  # 3, ws, ws, ws
        coords_ori_flatten = torch.flatten(coords_ori, 1)  # 3, ws*ws*ws

        coords_d = torch.arange(window_size_ext)
        coords_h = torch.arange(window_size_ext)
        coords_w = torch.arange(window_size_ext)
        coords_ext = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w]))  # 3, wse, wse, wse
        coords_ext_flatten = torch.flatten(coords_ext, 1)  # 3, wse*wse*wse

        relative_coords = coords_ext_flatten[:, None, :] - coords_ori_flatten[:, :, None]  # 3, ws*ws*ws, wse*wse*wse
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # ws*ws*ws, wse*wse*wse, 3
        relative_coords[:, :, 0] += window_size_ori - window_size_ext + 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size_ori - window_size_ext + 1
        relative_coords[:, :, 2] += window_size_ori - window_size_ext + 1
        relative_coords[:, :, 0] *= window_size_ori + window_size_ext - 1
        relative_coords[:, :, 1] *= window_size_ori + window_size_ext - 1
        relative_position_index = relative_coords.sum(-1)  # ws*ws*ws, wse*wse*wse
        return relative_position_index

    def calculate_mask(self, x_size, shift_size):
        D, H, W = x_size
        img_mask = torch.zeros((1, D, H, W, 1))  # 1 D H W 1

        d_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -shift_size),
                    slice(-shift_size, None))
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -shift_size),
                    slice(-shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -shift_size),
                    slice(-shift_size, None))

        cnt = 0
        for d in d_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

        mask_windows = window_partition3d(img_mask, to_3tuple(self.window_size))  # nw, ws, ws, ws, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def triangle_masks(self, x):
        ws = 2 * self.window_size
        # Create 3D triangular masks (simplified version for 3D)
        mask = torch.ones((ws, ws, ws), dtype=torch.bool, device=x.device)
        for d in range(ws):
            for h in range(ws):
                for w in range(ws):
                    if d + h + w < ws:
                        mask[d, h, w] = False
        return [mask]

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3], x.shape[4])

        # Calculate attention mask and relative position index
        attn_mask = tuple([self.calculate_mask(x_size, shift_size).to(x.device) for shift_size in (8, 16, 24)])
        triangular_masks = tuple(self.triangle_masks(x))

        params = {'attn_mask': attn_mask, 'triangular_masks': triangular_masks,
                  'rpi_sa': self.relative_position_index_SA, 'rpi_oca': self.relative_position_index_OCA}

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size, params)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean
        return x