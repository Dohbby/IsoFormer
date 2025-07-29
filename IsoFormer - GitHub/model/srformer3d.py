import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import to_3tuple, trunc_normal_
import torch.nn.functional as F


class emptyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


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


class dwconv3d(nn.Module):
    def __init__(self, hidden_features):
        super(dwconv3d, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv3d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                      dilation=1, groups=hidden_features),
            nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        # x: [B, L, C] where L = D*H*W
        # x_size: [D, H, W]
        B, L, C = x.shape
        x = x.transpose(1, 2).view(B, C, x_size[0], x_size[1], x_size[2]).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN3D(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.before_add = emptyModule()
        self.after_add = emptyModule()
        self.dwconv = dwconv3d(hidden_features=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = self.before_add(x)
        x = x + self.dwconv(x, x_size)
        x = self.after_add(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition3d(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B,
               D // window_size[0], window_size[0],
               H // window_size[1], window_size[1],
               W // window_size[2], window_size[2],
               C)
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


class PSA3D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [D, H, W]
        self.permuted_window_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.permuted_window_size[0] - 1) *
                        (2 * self.permuted_window_size[1] - 1) *
                        (2 * self.permuted_window_size[2] - 1), num_heads))

        # Relative position index
        coords_d = torch.arange(self.permuted_window_size[0])
        coords_h = torch.arange(self.permuted_window_size[1])
        coords_w = torch.arange(self.permuted_window_size[2])
        coords = torch.stack(torch.meshgrid([coords_d, coords_h, coords_w]))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.permuted_window_size[0] - 1
        relative_coords[:, :, 1] += self.permuted_window_size[1] - 1
        relative_coords[:, :, 2] += self.permuted_window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.permuted_window_size[1] - 1) * (2 * self.permuted_window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.permuted_window_size[2] - 1
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww

        # Expand for 3D
        relative_position_index = relative_position_index.reshape(
            self.permuted_window_size[0], self.permuted_window_size[1], self.permuted_window_size[2],
            1, 1, 1, self.permuted_window_size[0] * self.permuted_window_size[1] * self.permuted_window_size[2]
        ).repeat(1, 1, 1, 2, 2, 2, 1).permute(0, 3, 1, 4, 2, 5, 6).reshape(
            8 * self.permuted_window_size[0] * self.permuted_window_size[1] * self.permuted_window_size[2],
            self.permuted_window_size[0] * self.permuted_window_size[1] * self.permuted_window_size[2]
        )

        self.register_buffer('aligned_relative_position_index', relative_position_index)

        self.kv = nn.Linear(dim, dim // 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        # 修改 PSA3D.forward() 方法中的 kv reshape 部分
        kv = self.kv(x).reshape(
            B_,
            self.permuted_window_size[0], 2,
            self.permuted_window_size[1], 2,
            self.permuted_window_size[2], 2,
            self.num_heads, C // (self.num_heads * 8)
        ).permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
            B_, N // 8, 8, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        # KV compression

        k, v = kv[0], kv[1]  # Each takes 1/8 of the channels

        # Q remains full channels
        q = self.q(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))  # (B_, num_heads, N, N//8)

        relative_position_bias = self.relative_position_bias_table[
            self.aligned_relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2],
            self.permuted_window_size[0] * self.permuted_window_size[1] * self.permuted_window_size[2],
            -1
        )  # (N, N//8, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, N, N//8)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(B_ // nw, nw, self.num_heads, N, N // 8) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N // 8)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PSA_Block3D(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
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
        self.window_size = window_size if isinstance(window_size, (list, tuple)) else [window_size] * 3
        self.permuted_window_size = [ws // 2 for ws in self.window_size]
        self.shift_size = shift_size if isinstance(shift_size, (list, tuple)) else [shift_size] * 3

        if min(self.input_resolution) <= min(self.window_size):
            self.shift_size = [0, 0, 0]
            self.window_size = [min(self.input_resolution)] * 3

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = PSA3D(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvFFN3D(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if any([s > 0 for s in self.shift_size]):
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)

        # emptyModule for Power Spectrum Based Evaluation
        self.after_norm1 = emptyModule()
        self.after_attention = emptyModule()
        self.residual_after_attention = emptyModule()
        self.after_norm2 = emptyModule()
        self.after_mlp = emptyModule()
        self.residual_after_mlp = emptyModule()

    def calculate_mask(self, x_size):
        D, H, W = x_size
        img_mask = torch.zeros((1, D, H, W, 1))  # 1 D H W 1

        # Original window masking
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

        mask_windows = window_partition3d(img_mask, self.window_size)  # nW, wD, wH, wW, 1
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])

        # Permuted window masking
        permuted_window_mask = torch.zeros((1, D // 2, H // 2, W // 2, 1))
        d_slices = (slice(0, -self.permuted_window_size[0]),
                    slice(-self.permuted_window_size[0], -self.shift_size[0] // 2),
                    slice(-self.shift_size[0] // 2, None))
        h_slices = (slice(0, -self.permuted_window_size[1]),
                    slice(-self.permuted_window_size[1], -self.shift_size[1] // 2),
                    slice(-self.shift_size[1] // 2, None))
        w_slices = (slice(0, -self.permuted_window_size[2]),
                    slice(-self.permuted_window_size[2], -self.shift_size[2] // 2),
                    slice(-self.shift_size[2] // 2, None))

        cnt = 0
        for d in d_slices:
            for h in h_slices:
                for w in w_slices:
                    permuted_window_mask[:, d, h, w, :] = cnt
                    cnt += 1

        permuted_windows = window_partition3d(permuted_window_mask, self.permuted_window_size)
        permuted_windows = permuted_windows.view(-1,
                                                 self.permuted_window_size[0] *
                                                 self.permuted_window_size[1] *
                                                 self.permuted_window_size[2])

        # Attention mask
        attn_mask = mask_windows.unsqueeze(2) - permuted_windows.unsqueeze(1)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        D, H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = self.after_norm1(x)
        x = x.view(B, D, H, W, C)

        # Cyclic shift
        if any([s > 0 for s in self.shift_size]):
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]),
                                   dims=(1, 2, 3))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition3d(shifted_x, self.window_size)  # nW*B, wD, wH, wW, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)

        # W-MSA/SW-MSA
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse3d(attn_windows, self.window_size, D, H, W)  # B D' H' W' C

        # Reverse cyclic shift
        if any([s > 0 for s in self.shift_size]):
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]),
                           dims=(1, 2, 3))
        else:
            x = shifted_x

        x = x.view(B, D * H * W, C)
        x = self.after_attention(x)

        # FFN
        x = shortcut + self.drop_path(x)
        x = self.residual_after_attention(x)
        x = self.residual_after_mlp(
            x + self.drop_path(
                self.after_mlp(
                    self.mlp(
                        self.after_norm2(self.norm2(x)), x_size
                    )
                )
            )
        )
        return x


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
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B D/2 H/2 W/2 8*C
        x = x.view(B, -1, 8 * C)  # B D/2*H/2*W/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer3D(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
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
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            PSA_Block3D(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else [w // 2 for w in window_size],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
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


class PSA_Group3D(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
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
        super(PSA_Group3D, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer3D(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
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
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv3d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed3D(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.before_PSA_Group_conv = emptyModule()
        self.after_PSA_Group_conv = emptyModule()
        self.after_PSA_Group_Residual = emptyModule()

    def forward(self, x, x_size):
        return self.after_PSA_Group_Residual(
            self.after_PSA_Group_conv(
                self.patch_embed(
                    self.conv(
                        self.patch_unembed(
                            self.before_PSA_Group_conv(
                                self.residual_group(x, x_size)
                            ), x_size
                        )
                    )
                )
            ) + x
        )


class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=32, window_size=16, patch_size=1, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        if any([img_size % ws != 0 for ws in to_3tuple(window_size)]):
            new_size = [img_size + (ws - img_size % ws) if img_size % ws != 0 else img_size for ws in
                        to_3tuple(window_size)]
            img_size = new_size

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
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw*Pd, C
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchUnEmbed3D(nn.Module):
    def __init__(self, img_size=32, window_size=16, patch_size=1, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        if any([img_size % ws != 0 for ws in to_3tuple(window_size)]):
            new_size = [img_size + (ws - img_size % ws) if img_size % ws != 0 else img_size for ws in
                        to_3tuple(window_size)]
            img_size = new_size

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
        B, D, H, W = x_size[0], x_size[1], x_size[2], x_size[3] if len(x_size) > 3 else x_size[0]
        x = x.transpose(1, 2).view(B, self.embed_dim, D, H, W)
        return x


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


class UpsampleOneStep3D(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv3d(num_feat, (scale ** 3) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep3D, self).__init__(*m)


@ARCH_REGISTRY.register()
class SRFormer3D(nn.Module):
    def __init__(self,
                 img_size=32,
                 patch_size=1,
                 in_chans=1,
                 embed_dim=64,
                 depths=(2,2,2,2,2,2),
                 num_heads=(2,2,2,2,2,2),
                 window_size=4,
                 mlp_ratio=2.,
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
        super(SRFormer3D, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        self.mean = torch.zeros(1, 1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size if isinstance(window_size, (list, tuple)) else [window_size] * 3

        # 1. Shallow feature extraction
        self.conv_first = nn.Conv3d(num_in_ch, embed_dim, 3, 1, 1)

        # 2. Deep feature extraction
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # Split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            img_size=img_size,
            window_size=self.window_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # Merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed3D(
            img_size=img_size,
            window_size=self.window_size,
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

        # Build Permuted Self Attention Groups (PSA_Group)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = PSA_Group3D(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1], patches_resolution[2]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
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

        # Build the last conv layer in deep feature extraction
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
            # For classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv3d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True))
            self.upsample = Upsample3D(upscale, num_feat)
            self.conv_last = nn.Conv3d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # For lightweight SR
            self.upsample = UpsampleOneStep3D(upscale, embed_dim, num_out_ch,
                                              (patches_resolution[0], patches_resolution[1], patches_resolution[2]))
        elif self.upsampler == 'nearest+conv':
            # For real-world SR
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
            # For image denoising and JPEG compression artifact reduction
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

    def check_image_size(self, x):
        _, _, D, H, W = x.size()
        mod_pad_d = (self.window_size[0] - D % self.window_size[0]) % self.window_size[0]
        mod_pad_h = (self.window_size[1] - H % self.window_size[1]) % self.window_size[1]
        mod_pad_w = (self.window_size[2] - W % self.window_size[2]) % self.window_size[2]
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h, 0, mod_pad_d), 'reflect')
        return x

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
        D, H, W = x.shape[2], x.shape[3], x.shape[4]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # For classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # For lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # For real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(F.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(F.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # For image denoising and compression artifact reduction
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean
        return x[:, :, :D * self.upscale, :H * self.upscale, :W * self.upscale]