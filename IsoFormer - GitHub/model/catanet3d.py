import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from inspect import isfunction
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import trunc_normal_
import math


def exists(val):
    return val is not None


def is_empty(t):
    return t.nelement() == 0


def expand_dim(t, dim, k):
    t = t.unsqueeze(dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)


def default(x, d):
    if not exists(x):
        return d if not isfunction(d) else d()
    return x


def ema(old, new, decay):
    if not exists(old):
        return new
    return old * decay + new * (1 - decay)


def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def similarity(x, means):
    return torch.einsum('bld,cd->blc', x, means)


def dists_and_buckets(x, means):
    dists = similarity(x, means)
    _, buckets = torch.max(dists, dim=-1)
    return dists, buckets


def batched_bincount(index, num_classes, dim=-1):
    shape = list(index.shape)
    shape[dim] = num_classes
    out = index.new_zeros(shape)
    out.scatter_add_(dim, index, torch.ones_like(index, dtype=index.dtype))
    return out


def center_iter(x, means, buckets=None):
    b, l, d, dtype, num_tokens = *x.shape, x.dtype, means.shape[0]

    if not exists(buckets):
        _, buckets = dists_and_buckets(x, means)

    bins = batched_bincount(buckets, num_tokens).sum(0, keepdim=True)
    zero_mask = bins.long() == 0

    means_ = buckets.new_zeros(b, num_tokens, d, dtype=dtype)
    means_.scatter_add_(-2, expand_dim(buckets, -1, d), x)
    means_ = F.normalize(means_.sum(0, keepdim=True), dim=-1).type(dtype)
    means = torch.where(zero_mask.unsqueeze(-1), means, means_)
    means = means.squeeze(0)
    return means


class IASA(nn.Module):
    def __init__(self, dim, qk_dim, heads, group_size):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.group_size = group_size

    def forward(self, normed_x, idx_last, k_global, v_global):
        x = normed_x
        B, N, _ = x.shape

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q = torch.gather(q, dim=-2, index=idx_last.expand(q.shape))
        k = torch.gather(k, dim=-2, index=idx_last.expand(k.shape))
        v = torch.gather(v, dim=-2, index=idx_last.expand(v.shape))

        gs = min(N, self.group_size)  # group size
        ng = (N + gs - 1) // gs
        pad_n = ng * gs - N

        paded_q = torch.cat((q, torch.flip(q[:, N - pad_n:N, :], dims=[-2])), dim=-2)
        paded_q = rearrange(paded_q, "b (ng gs) (h d) -> b ng h gs d", ng=ng, h=self.heads)
        paded_k = torch.cat((k, torch.flip(k[:, N - pad_n - gs:N, :], dims=[-2])), dim=-2)
        paded_k = paded_k.unfold(-2, 2 * gs, gs)
        paded_k = rearrange(paded_k, "b ng (h d) gs -> b ng h gs d", h=self.heads)
        paded_v = torch.cat((v, torch.flip(v[:, N - pad_n - gs:N, :], dims=[-2])), dim=-2)
        paded_v = paded_v.unfold(-2, 2 * gs, gs)
        paded_v = rearrange(paded_v, "b ng (h d) gs -> b ng h gs d", h=self.heads)
        out1 = F.scaled_dot_product_attention(paded_q, paded_k, paded_v)

        k_global = k_global.reshape(1, 1, *k_global.shape).expand(B, ng, -1, -1, -1)
        v_global = v_global.reshape(1, 1, *v_global.shape).expand(B, ng, -1, -1, -1)

        out2 = F.scaled_dot_product_attention(paded_q, k_global, v_global)
        out = out1 + out2
        out = rearrange(out, "b ng h gs d -> b (ng gs) (h d)")[:, :N, :]

        out = out.scatter(dim=-2, index=idx_last.expand(out.shape), src=out)
        out = self.proj(out)

        return out


class IRCA(nn.Module):
    def __init__(self, dim, qk_dim, heads):
        super().__init__()
        self.heads = heads
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

    def forward(self, normed_x, x_means):
        x = normed_x
        if self.training:
            x_global = center_iter(F.normalize(x, dim=-1), F.normalize(x_means, dim=-1))
        else:
            x_global = x_means

        k, v = self.to_k(x_global), self.to_v(x_global)
        k = rearrange(k, 'n (h dim_head)->h n dim_head', h=self.heads)
        v = rearrange(v, 'n (h dim_head)->h n dim_head', h=self.heads)

        return k, v, x_global.detach()


class TAB(nn.Module):
    def __init__(self, dim, qk_dim, mlp_dim, heads, n_iter=2,
                 num_tokens=8, group_size=128,
                 ema_decay=0.999):
        super().__init__()

        self.n_iter = n_iter
        self.ema_decay = ema_decay
        self.num_tokens = num_tokens

        self.norm = nn.LayerNorm(dim)
        self.mlp = PreNorm(dim, ConvFFN3D(dim, mlp_dim))
        self.irca_attn = IRCA(dim, qk_dim, heads)
        self.iasa_attn = IASA(dim, qk_dim, heads, group_size)
        self.register_buffer('means', torch.randn(num_tokens, dim))
        self.register_buffer('initted', torch.tensor(False))
        self.conv1x1 = nn.Conv3d(dim, dim, 1, bias=False)

    def forward(self, x):
        _, _, d, h, w = x.shape
        x = rearrange(x, 'b c d h w->b (d h w) c')
        residual = x
        x = self.norm(x)
        B, N, _ = x.shape

        idx_last = torch.arange(N, device=x.device).reshape(1, N).expand(B, -1)
        if not self.initted:
            pad_n = self.num_tokens - N % self.num_tokens
            paded_x = torch.cat((x, torch.flip(x[:, N - pad_n:N, :], dims=[-2])), dim=-2)
            x_means = torch.mean(rearrange(paded_x, 'b (cnt n) c->cnt (b n) c', cnt=self.num_tokens), dim=-2).detach()
        else:
            x_means = self.means.detach()

        if self.training:
            with torch.no_grad():
                for _ in range(self.n_iter - 1):
                    x_means = center_iter(F.normalize(x, dim=-1), F.normalize(x_means, dim=-1))

        k_global, v_global, x_means = self.irca_attn(x, x_means)

        with torch.no_grad():
            x_scores = torch.einsum('b i c,j c->b i j',
                                    F.normalize(x, dim=-1),
                                    F.normalize(x_means, dim=-1))
            x_belong_idx = torch.argmax(x_scores, dim=-1)

            idx = torch.argsort(x_belong_idx, dim=-1)
            idx_last = torch.gather(idx_last, dim=-1, index=idx).unsqueeze(-1)

        y = self.iasa_attn(x, idx_last, k_global, v_global)
        y = rearrange(y, 'b (d h w) c->b c d h w', d=d, h=h).contiguous()
        y = self.conv1x1(y)
        x = residual + rearrange(y, 'b c d h w->b (d h w) c')
        x = self.mlp(x, x_size=(d, h, w)) + x

        if self.training:
            with torch.no_grad():
                new_means = x_means
                if not self.initted:
                    self.means.data.copy_(new_means)
                    self.initted.data.copy_(torch.tensor(True))
                else:
                    ema_inplace(self.means, new_means, self.ema_decay)

        return rearrange(x, 'b (d h w) c->b c d h w', d=d, h=h)


def patch_divide_3d(x, step, ps):
    """Crop 3D volume into patches."""
    b, c, t, h, w = x.size()
    if h == ps and w == ps and t == ps:
        step = ps
    crop_x = []
    nt = 0
    for k in range(0, t + step - ps, step):
        top_t = k
        bottom_t = k + ps
        if bottom_t > t:
            top_t = t - ps
            bottom_t = t
        nt += 1
        nh = 0
        for i in range(0, h + step - ps, step):
            top_h = i
            bottom_h = i + ps
            if bottom_h > h:
                top_h = h - ps
                bottom_h = h
            nh += 1
            for j in range(0, w + step - ps, step):
                left_w = j
                right_w = j + ps
                if right_w > w:
                    left_w = w - ps
                    right_w = w
                crop_x.append(x[:, :, top_t:bottom_t, top_h:bottom_h, left_w:right_w])
    nw = len(crop_x) // (nt * nh)
    crop_x = torch.stack(crop_x, dim=0)  # (n, b, c, ps, ps, ps)
    crop_x = crop_x.permute(1, 0, 2, 3, 4, 5).contiguous()  # (b, n, c, ps, ps, ps)
    return crop_x, nt, nh, nw


def patch_reverse_3d(crop_x, x, step, ps):
    """Reverse 3D patches into volume."""
    b, c, t, h, w = x.size()
    output = torch.zeros_like(x)
    index = 0
    for k in range(0, t + step - ps, step):
        top_t = k
        bottom_t = k + ps
        if bottom_t > t:
            top_t = t - ps
            bottom_t = t
        for i in range(0, h + step - ps, step):
            top_h = i
            bottom_h = i + ps
            if bottom_h > h:
                top_h = h - ps
                bottom_h = h
            for j in range(0, w + step - ps, step):
                left_w = j
                right_w = j + ps
                if right_w > w:
                    left_w = w - ps
                    right_w = w
                output[:, :, top_t:bottom_t, top_h:bottom_h, left_w:right_w] += crop_x[:, index]
                index += 1

    # Handle overlapping regions
    for k in range(step, t + step - ps, step):
        top_t = k
        bottom_t = k + ps - step
        if top_t + ps > t:
            top_t = t - ps
        output[:, :, top_t:bottom_t, :, :] /= 2

    for i in range(step, h + step - ps, step):
        top_h = i
        bottom_h = i + ps - step
        if top_h + ps > h:
            top_h = h - ps
        output[:, :, :, top_h:bottom_h, :] /= 2

    for j in range(step, w + step - ps, step):
        left_w = j
        right_w = j + ps - step
        if left_w + ps > w:
            left_w = w - ps
        output[:, :, :, :, left_w:right_w] /= 2

    return output


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class dwconv3d(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv3d, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv3d(hidden_features, hidden_features, kernel_size=kernel_size,
                      stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features),
            nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features,
                                   x_size[0], x_size[1], x_size[2]).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN3D(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
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


class Attention3D(nn.Module):
    def __init__(self, dim, heads, qk_dim):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.qk_dim = qk_dim
        self.scale = qk_dim ** -0.5

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.proj(out)


class LRSA3D(nn.Module):
    def __init__(self, dim, qk_dim, mlp_dim, heads=1):
        super().__init__()
        self.layer = nn.ModuleList([
            PreNorm(dim, Attention3D(dim, heads, qk_dim)),
            PreNorm(dim, ConvFFN3D(dim, mlp_dim))])

    def forward(self, x, ps):
        step = ps - 2
        crop_x, nt, nh, nw = patch_divide_3d(x, step, ps)
        b, n, c, pt, ph, pw = crop_x.shape
        crop_x = rearrange(crop_x, 'b n c t h w -> (b n) (t h w) c')

        attn, ff = self.layer
        crop_x = attn(crop_x) + crop_x
        crop_x = rearrange(crop_x, '(b n) (t h w) c -> b n c t h w', n=n, t=pt, w=pw)

        x = patch_reverse_3d(crop_x, x, step, ps)
        _, _, t, h, w = x.shape
        x = rearrange(x, 'b c t h w-> b (t h w) c')
        x = ff(x, x_size=(t, h, w)) + x
        x = rearrange(x, 'b (t h w) c->b c t h w', t=t, h=h)

        return x


@ARCH_REGISTRY.register()
class CATANet3D(nn.Module):
    setting = dict(dim=40, block_num=4, qk_dim=36, mlp_dim=96, heads=4,
                   patch_size=[4,8,16,32])

    def __init__(self, in_chans=1, n_iters=[2, 2, 2, 2],
                 num_tokens=[16, 32, 64, 128],
                 group_size=[256, 128, 64, 32],
                 upscale: int = 1):
        super().__init__()

        self.dim = self.setting['dim']
        self.block_num = self.setting['block_num']
        self.patch_size = self.setting['patch_size']
        self.qk_dim = self.setting['qk_dim']
        self.mlp_dim = self.setting['mlp_dim']
        self.upscale = upscale
        self.heads = self.setting['heads']

        self.n_iters = n_iters
        self.num_tokens = num_tokens
        self.group_size = group_size

        # First convolution
        self.first_conv = nn.Conv3d(in_chans, self.dim, 3, 1, 1)

        # Main blocks
        self.blocks = nn.ModuleList()
        self.mid_convs = nn.ModuleList()

        for i in range(self.block_num):
            self.blocks.append(nn.ModuleList([
                TAB(self.dim, self.qk_dim, self.mlp_dim, self.heads,
                    self.n_iters[i], self.num_tokens[i], self.group_size[i]),
                LRSA3D(self.dim, self.qk_dim, self.mlp_dim, self.heads)
            ]))
            self.mid_convs.append(nn.Conv3d(self.dim, self.dim, 3, 1, 1))

        # Upsampling
        if upscale == 4:
            self.upconv1 = nn.Conv3d(self.dim, self.dim * 8, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv3d(self.dim, self.dim * 8, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif upscale == 2 or upscale == 3:
            self.upconv = nn.Conv3d(self.dim, self.dim * (upscale ** 3), 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(upscale)

        self.last_conv = nn.Conv3d(self.dim, in_chans, 3, 1, 1)
        if upscale != 1:
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        for i in range(self.block_num):
            residual = x
            global_attn, local_attn = self.blocks[i]
            x = global_attn(x)
            x = local_attn(x, self.patch_size[i])
            x = residual + self.mid_convs[i](x)
        return x

    def forward(self, x):
        if self.upscale != 1:
            base = F.interpolate(x, scale_factor=self.upscale, mode='trilinear', align_corners=False)
        else:
            base = x

        x = self.first_conv(x)
        x = self.forward_features(x) + x

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(x)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 1:
            out = x
        else:
            out = self.lrelu(self.pixel_shuffle(self.upconv(x)))

        out = self.last_conv(out) + base
        return out

    def __repr__(self):
        num_parameters = sum(map(lambda x: x.numel(), self.parameters()))
        return '#Params of {}: {:<.4f} [K]'.format(self._get_name(),
                                                   num_parameters / 10 ** 3)