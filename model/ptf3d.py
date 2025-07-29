import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from basicsr.archs.arch_util import to_3tuple, trunc_normal_  # Modified to_2tuple to to_3tuple
from fairscale.nn import checkpoint_wrapper
from basicsr.utils.registry import ARCH_REGISTRY
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import smm_cuda  # Assumed to be adapted for 3D

class SMM_QmK(Function):
    """
    Custom PyTorch autograd Function for sparse matrix multiplication (SMM) of
    query (Q) and key (K) matrices, based on given sparse indices.
    Assumed to be adapted for 3D in smm_cuda.
    """
    @staticmethod
    def forward(ctx, A, B, index):
        ctx.save_for_backward(A, B, index)
        return smm_cuda.SMM_QmK_forward_cuda(A.contiguous(), B.contiguous(), index.contiguous())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        A, B, index = ctx.saved_tensors
        grad_A, grad_B = smm_cuda.SMM_QmK_backward_cuda(
            grad_output.contiguous(), A.contiguous(), B.contiguous(), index.contiguous()
        )
        return grad_A, grad_B, None

class SMM_AmV(Function):
    """
    Custom PyTorch autograd Function for sparse matrix multiplication (SMM)
    between an activation matrix (A) and a value matrix (V).
    Assumed to be adapted for 3D in smm_cuda.
    """
    @staticmethod
    def forward(ctx, A, B, index):
        ctx.save_for_backward(A, B, index)
        return smm_cuda.SMM_AmV_forward_cuda(A.contiguous(), B.contiguous(), index.contiguous())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        A, B, index = ctx.saved_tensors
        grad_A, grad_B = smm_cuda.SMM_AmV_backward_cuda(
            grad_output.contiguous(), A.contiguous(), B.contiguous(), index.contiguous()
        )
        return grad_A, grad_B, None

class dwconv3d(nn.Module):
    def __init__(self, hidden_features, kernel_size=3):  # Reduced kernel size for 3D to manage computation
        super(dwconv3d, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv3d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        # x_size: (d, h, w)
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1], x_size[2]).contiguous()  # b Pd*Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=3, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv3d(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (b, d, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, window_size, c)
    """
    b, d, h, w, c = x.shape
    x = x.view(b, d // window_size, window_size, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, c)
    return windows

def window_reverse(windows, window_size, d, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, window_size, c)
        window_size (int): Window size
        d (int): Depth of volume
        h (int): Height of volume
        w (int): Width of volume

    Returns:
        x: (b, d, h, w, c)
    """
    b = int(windows.shape[0] / (d * h * w / window_size / window_size / window_size))
    x = windows.view(b, d // window_size, h // window_size, w // window_size, window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)
    return x

class WindowAttention(nn.Module):
    """
    Shifted Window-based Multi-head Self-Attention (MSA) for 3D.

    Args:
        dim (int): Number of input channels.
        layer_id (int): Index of the current layer.
        window_size (tuple[int]): The depth, height, and width of the window.
        num_heads (int): Number of attention heads.
        num_topk (tuple[int]): Number of top-k attention values retained for sparsity.
        qkv_bias (bool, optional): If True, add a learnable bias. Default: True.
    """
    def __init__(self, dim, layer_id, window_size, num_heads, num_topk, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        self.num_topk = num_topk
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.eps = 1e-20

        # Relative position bias table for 3D
        if dim > 100:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), self.num_heads)
            )
        else:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), 1)
            )
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.topk = self.num_topk[self.layer_id]

    def forward(self, qkvp, pfa_values, pfa_indices, rpi, mask=None, shift=0):
        b_, n, c4 = qkvp.shape
        c = c4 // 4
        qkvp = qkvp.reshape(b_, n, 4, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v, v_lepe = qkvp[0], qkvp[1], qkvp[2], qkvp[3]

        q = q * self.scale
        if pfa_indices[shift] is None:
            attn = (q @ k.transpose(-2, -1))  # b_, num_heads, n, n
            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            if not self.training:
                attn.add_(relative_position_bias)
            else:
                attn = attn + relative_position_bias

            if shift:
                nw = mask.shape[0]
                attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, n, n)
        else:
            topk = pfa_indices[shift].shape[-1]
            q = q.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            k = k.contiguous().view(b_ * self.num_heads, n, c // self.num_heads).transpose(-2, -1)
            smm_index = pfa_indices[shift].view(b_ * self.num_heads, n, topk).int()
            attn = SMM_QmK.apply(q, k, smm_index).view(b_, self.num_heads, n, topk)

            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0).expand(b_, self.num_heads, n, n)
            relative_position_bias = torch.gather(relative_position_bias, dim=-1, index=pfa_indices[shift])
            if not self.training:
                attn.add_(relative_position_bias)
            else:
                attn = attn + relative_position_bias

        if not self.training:
            attn = torch.softmax(attn, dim=-1, out=attn)
        else:
            attn = self.softmax(attn)

        if pfa_values[shift] is not None:
            if not self.training:
                attn.mul_(pfa_values[shift])
                attn.add_(self.eps)
                denom = attn.sum(dim=-1, keepdim=True).add_(self.eps)
                attn.div_(denom)
            else:
                attn = (attn * pfa_values[shift])
                attn = (attn + self.eps) / (attn.sum(dim=-1, keepdim=True) + self.eps)

        if self.topk < self.window_size[0] * self.window_size[1] * self.window_size[2]:
            topk_values, topk_indices = torch.topk(attn, self.topk, dim=-1, largest=True, sorted=False)
            attn = topk_values
            if pfa_indices[shift] is not None:
                pfa_indices[shift] = torch.gather(pfa_indices[shift], dim=-1, index=topk_indices)
            else:
                pfa_indices[shift] = topk_indices

        pfa_values[shift] = attn

        if pfa_indices[shift] is None:
            x = ((attn @ v) + v_lepe).transpose(1, 2).reshape(b_, n, c)
        else:
            topk = pfa_indices[shift].shape[-1]
            attn = attn.view(b_ * self.num_heads, n, topk)
            v = v.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            smm_index = pfa_indices[shift].view(b_ * self.num_heads, n, topk).int()
            x = (SMM_AmV.apply(attn, v, smm_index).view(b_, self.num_heads, n, c // self.num_heads) + v_lepe).transpose(1, 2).reshape(b_, n, c)

        if not self.training:
            del q, k, v, relative_position_bias
            torch.cuda.empty_cache()

        x = self.proj(x)
        return x, pfa_values, pfa_indices

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'

    def flops(self, n):
        flops = 0
        if self.layer_id < 2:
            flops += self.num_heads * n * (self.dim // self.num_heads) * n
            flops += self.num_heads * n * n * (self.dim // self.num_heads)
        else:
            flops += self.num_heads * n * (self.dim // self.num_heads) * self.num_topk[self.layer_id-2]
            flops += self.num_heads * n * self.num_topk[self.layer_id] * (self.dim // self.num_heads)
        flops += n * self.dim * self.dim
        return flops

class PFTransformerLayer(nn.Module):
    """
    PFT Transformer Layer for 3D.

    Args:
        dim (int): Number of input channels.
        block_id (int): Block index.
        layer_id (int): Layer index.
        input_resolution (tuple[int]): Input resolution (d, h, w).
        num_heads (int): Number of attention heads.
        num_topk (tuple[int]): Number of top-k attention values.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        convffn_kernel_size (int): Convolutional kernel size for ConvFFN.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add bias. Default: True
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """
    def __init__(self, dim, block_id, layer_id, input_resolution, num_heads, num_topk, window_size,
                 shift_size, convffn_kernel_size, mlp_ratio, qkv_bias=True, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.softmax = nn.Softmax(dim=-1)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.convlepe_kernel_size = convffn_kernel_size
        self.v_LePE = dwconv3d(hidden_features=dim, kernel_size=self.convlepe_kernel_size)

        self.attn_win = WindowAttention(
            self.dim,
            layer_id=layer_id,
            window_size=to_3tuple(self.window_size),
            num_heads=num_heads,
            num_topk=num_topk,
            qkv_bias=qkv_bias,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size,
                               act_layer=act_layer)

    def forward(self, x, pfa_list, x_size, params):
        pfa_values, pfa_indices = pfa_list[0], pfa_list[1]
        d, h, w = x_size
        b, n, c = x.shape
        c4 = 4 * c

        shortcut = x
        x = self.norm1(x)
        x_qkv = self.wqkv(x)

        v_lepe = self.v_LePE(torch.split(x_qkv, c, dim=-1)[-1], x_size)
        x_qkvp = torch.cat([x_qkv, v_lepe], dim=-1)

        if self.shift_size > 0:
            shift = 1
            shifted_x = torch.roll(x_qkvp.reshape(b, d, h, w, c4),
                                   shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
        else:
            shift = 0
            shifted_x = x_qkvp.reshape(b, d, h, w, c4)

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size, c4)

        attn_windows, pfa_values, pfa_indices = self.attn_win(
            x_windows, pfa_values=pfa_values, pfa_indices=pfa_indices, rpi=params['rpi_sa'], mask=params['attn_mask'], shift=shift
        )

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, d, h, w)

        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        else:
            attn_x = shifted_x

        x_win = attn_x
        x = shortcut + x_win.view(b, n, c)
        x = x + self.convffn(self.norm2(x), x_size)

        pfa_list = [pfa_values, pfa_indices]
        return x, pfa_list

    def flops(self, input_resolution=None):
        flops = 0
        d, h, w = self.input_resolution if input_resolution is None else input_resolution
        flops += self.dim * 3 * self.dim * d * h * w
        nw = d * h * w / self.window_size / self.window_size / self.window_size
        flops += nw * self.attn_win.flops(self.window_size * self.window_size * self.window_size)
        flops += 2 * d * h * w * self.dim * self.dim * self.mlp_ratio
        flops += d * h * w * self.dim * (self.convffn_kernel_size ** 3) * self.mlp_ratio
        flops += d * h * w * self.dim * (self.convlepe_kernel_size ** 3)
        return flops

class PatchMerging(nn.Module):
    """
    Patch Merging Layer for 3D.

    Args:
        input_resolution (tuple[int]): Resolution of input feature (d, h, w).
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)  # 2x2x2 patches = 8
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        d, h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == d * h * w, 'input feature has wrong size'
        assert d % 2 == 0 and h % 2 == 0 and w % 2 == 0, f'x size ({d}*{h}*{w}) are not even.'

        x = x.view(b, d, h, w, c)

        x0 = x[:, 0::2, 0::2, 0::2, :]  # b d/2 h/2 w/2 c
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # b d/2 h/2 w/2 8*c
        x = x.view(b, -1, 8 * c)

        x = self.norm(x)
        x = self.reduction(x)
        return x

    def extra_repr(self) -> str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self, input_resolution=None):
        d, h, w = self.input_resolution if input_resolution is None else input_resolution
        flops = d * h * w * self.dim
        flops += (d // 2) * (h // 2) * (w // 2) * 8 * self.dim * 2 * self.dim
        return flops

class BasicBlock(nn.Module):
    """
    Basic PFT Block for one stage in 3D.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution (d, h, w).
        idx (int): Block index.
        layer_id (int): Layer index.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        num_topk (tuple[int]): Number of top-k attention values.
        window_size (int): Local window size.
        convffn_kernel_size (int): Convolutional kernel size for ConvFFN.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add bias. Default: True
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing. Default: False.
    """
    def __init__(self, dim, input_resolution, idx, layer_id, depth, num_heads, num_topk, window_size,
                 convffn_kernel_size, mlp_ratio=4., qkv_bias=True, norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                PFTransformerLayer(
                    dim=dim,
                    block_id=idx,
                    layer_id=layer_id + i,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    num_topk=num_topk,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    convffn_kernel_size=convffn_kernel_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
            )

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, pfa_list, x_size, params):
        for layer in self.layers:
            x, pfa_list = layer(x, pfa_list, x_size, params)
        if self.downsample is not None:
            x = self.downsample(x)
        return x, pfa_list

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self, input_resolution=None):
        flops = 0
        for layer in self.layers:
            flops += layer.flops(input_resolution)
        if self.downsample is not None:
            flops += self.downsample.flops(input_resolution)
        return flops

class PFTB(nn.Module):
    """
    Adaptive Token Dictionary Block (PFTB) for 3D.

    Args:
        dim (int): Number of input channels.
        idx (int): Block index.
        layer_id (int): Layer index.
        input_resolution (tuple[int]): Input resolution (d, h, w).
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        num_topk (tuple[int]): Number of top-k attention values.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add bias. Default: True
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing. Default: False.
        img_size: Input volume size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """
    def __init__(self, dim, idx, layer_id, input_resolution, depth, num_heads, num_topk, window_size,
                 convffn_kernel_size, mlp_ratio, qkv_bias=True, norm_layer=nn.LayerNorm,
                 downsample=None, use_checkpoint=False, img_size=32, patch_size=1, resi_connection='1conv'):
        super(PFTB, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.residual_group = BasicBlock(
            dim=dim,
            input_resolution=input_resolution,
            idx=idx,
            layer_id=layer_id,
            depth=depth,
            num_heads=num_heads,
            num_topk=num_topk,
            window_size=window_size,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )

        if resi_connection == '1conv':
            self.conv = nn.Conv3d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv3d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, pfa_list, x_size, params):
        x_Basicblock, pfa_list = self.residual_group(x, pfa_list, x_size, params)
        return self.patch_embed(self.conv(self.patch_unembed(x_Basicblock, x_size))) + x, pfa_list

    def flops(self, input_resolution=None):
        flops = 0
        flops += self.residual_group.flops(input_resolution)
        d, h, w = self.input_resolution if input_resolution is None else input_resolution
        flops += d * h * w * self.dim * self.dim * 27  # 3x3x3 kernel
        flops += self.patch_embed.flops(input_resolution)
        flops += self.patch_unembed.flops(input_resolution)
        return flops

class PatchEmbed(nn.Module):
    """
    Volume to Patch Embedding for 3D.

    Args:
        img_size (int): Volume size. Default: 32.
        patch_size (int): Patch token size. Default: 1.
        in_chans (int): Number of input volume channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 64.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, img_size=32, patch_size=1, in_chans=1, embed_dim=64, norm_layer=None):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        patches_resolution = [img_size[i] // patch_size[i] for i in range(3)]
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
        x = x.flatten(2).transpose(1, 2)  # b Pd*Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, input_resolution=None):
        flops = 0
        d, h, w = self.img_size if input_resolution is None else input_resolution
        if self.norm is not None:
            flops += d * h * w * self.embed_dim
        return flops

class PatchUnEmbed(nn.Module):
    """
    Volume to Patch Unembedding for 3D.

    Args:
        img_size (int): Volume size. Default: 32.
        patch_size (int): Patch token size. Default: 1.
        in_chans (int): Number of input volume channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 64.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, img_size=32, patch_size=1, in_chans=1, embed_dim=64, norm_layer=None):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        patches_resolution = [img_size[i] // patch_size[i] for i in range(3)]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1], x_size[2])  # b Pd*Ph*Pw c
        return x

    def flops(self, input_resolution=None):
        return 0

class Upsample(nn.Sequential):
    """
    Upsample module for 3D.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """
    def __init__(self, scale, num_feat):
        m = []
        self.scale = scale
        self.num_feat = num_feat
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv3d(num_feat, 8 * num_feat, 3, 1, 1))  # 2x2x2 = 8
                m.append(nn.PixelShuffle(upscale_factor=2))  # Note: PyTorch PixelShuffle is 2D; assume 3D equivalent
        elif scale == 3:
            m.append(nn.Conv3d(num_feat, 27 * num_feat, 3, 1, 1))  # 3x3x3 = 27
            m.append(nn.PixelShuffle(upscale_factor=3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        d, h, w = input_resolution
        if (self.scale & (self.scale - 1)) == 0:
            flops += self.num_feat * 8 * self.num_feat * 27 * d * h * w * int(math.log(self.scale, 2))
        else:
            flops += self.num_feat * 27 * self.num_feat * 27 * d * h * w
        return flops

class UpsampleOneStep(nn.Sequential):
    """
    UpsampleOneStep module for 3D (1conv + 1pixelshuffle).

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
        num_out_ch (int): Number of output channels.
        input_resolution (tuple[int]): Input resolution.
    """
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv3d(num_feat, (scale ** 3) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(upscale_factor=scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        d, h, w = self.patches_resolution if input_resolution is None else input_resolution
        flops = d * h * w * self.num_feat * 3 * 27
        return flops

@ARCH_REGISTRY.register()
class PFT(nn.Module):
    """
    Progressive Focused Transformer for 3D Single Volume Super-Resolution.

    Args:
        img_size (int): Volume size. Default: 32.
        patch_size (int): Patch size. Default: 1.
        in_chans (int): Number of input channels. Default: 1.
        embed_dim (int): Embedding dimension. Default: 64.
        depths (tuple[int]): Depths of each stage. Default: (2,2,2,2).
        num_heads (tuple[int]): Number of attention heads. Default: (6,6,6,6).
        num_topk (list[int]): Top-k attention values. Default: [256,256,128,128,64,64,32,32,16,16].
        window_size (int): Window size. Default: 4.
        convffn_kernel_size (int): ConvFFN kernel size. Default: 3.
        mlp_ratio (float): MLP ratio. Default: 2.
        qkv_bias (bool): If True, add bias. Default: True.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): Absolute position embedding. Default: False.
        patch_norm (bool): Patch normalization. Default: True.
        use_checkpoint (bool): Use checkpointing. Default: False.
        upscale (int): Upscale factor. Default: 1.
        img_range (float): Image range. Default: 1.
        upsampler (str): Upsampler type. Default: 'pixelshuffle'.
        resi_connection (str): Residual connection type. Default: '1conv'.
    """
    def __init__(self, img_size=32, patch_size=1, in_chans=1, embed_dim=64, depths=(2,2,2,2),
                 num_heads=(6,6,6,6), num_topk=[256,256,128,128,64,64,32,32,16,16], window_size=4,
                 convffn_kernel_size=3, mlp_ratio=2., qkv_bias=True, norm_layer=nn.LayerNorm,
                 ape=False, patch_norm=True, use_checkpoint=False, upscale=1, img_range=1.,
                 upsampler='pixelshuffle', resi_connection='1conv', **kwargs):
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
        self.upsampler = upsampler

        self.conv_first = nn.Conv3d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.layer_id = 0
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = PFTB(
                dim=embed_dim,
                idx=i_layer,
                layer_id=self.layer_id,
                input_resolution=patches_resolution,
                depth=depths[i_layer],
                num_heads=num_heads,
                num_topk=num_topk,
                window_size=window_size,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
            )
            self.layers.append(layer)
            self.layer_id += depths[i_layer]

        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv3d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv3d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(embed_dim // 4, embed_dim, 3, 1, 1))

        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv3d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch, patches_resolution)
        elif self.upsampler == 'nearest+conv':
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv3d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
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
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3], x.shape[4])
        pfa_values = [None, None]
        pfa_indices = [None, None]
        pfa_list = [pfa_values, pfa_indices]

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed

        for layer in self.layers:
            x, pfa_list = layer(x, pfa_list, x_size, params)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def calculate_rpi_sa(self):
        coords_d = torch.arange(self.window_size)
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w]))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 2] += self.window_size - 1
        relative_coords[:, :, 0] *= (2 * self.window_size - 1) * (2 * self.window_size - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size - 1)
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size):
        d, h, w = x_size
        img_mask = torch.zeros((1, d, h, w, 1))
        d_slices = (slice(0, -self.window_size), slice(-self.window_size, -(self.window_size // 2)),
                    slice(-(self.window_size // 2), None))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -(self.window_size // 2)),
                    slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -(self.window_size // 2)),
                    slice(-(self.window_size // 2), None))
        cnt = 0
        for d_s in d_slices:
            for h_s in h_slices:
                for w_s in w_slices:
                    img_mask[:, d_s, h_s, w_s, :] = cnt
                    cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        d_ori, h_ori, w_ori = x.size()[-3], x.size()[-2], x.size()[-1]
        mod = self.window_size
        d_pad = ((d_ori + mod - 1) // mod) * mod - d_ori
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        d, h, w = d_ori + d_pad, h_ori + h_pad, w_ori + w_pad
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :d, :, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :h, :]
        x = torch.cat([x, torch.flip(x, [4])], 4)[:, :, :, :, :w]

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        attn_mask = self.calculate_mask([d, h, w]).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        if self.upsampler == 'pixelshuffle':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean
        x = x[..., :d_ori * self.upscale, :h_ori * self.upscale, :w_ori * self.upscale]
        return x

    def flops(self, input_resolution=None):
        flops = 0
        resolution = self.patches_resolution if input_resolution is None else input_resolution
        d, h, w = resolution
        flops += d * h * w * 3 * self.embed_dim * 27
        flops += self.patch_embed.flops(resolution)
        for layer in self.layers:
            flops += layer.flops(resolution)
        flops += d * h * w * 3 * self.embed_dim * self.embed_dim
        if self.upsampler == 'pixelshuffle':
            flops += self.upsample.flops(resolution)
        else:
            flops += self.upsample.flops(resolution)
        return flops