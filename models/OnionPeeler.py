
from einops import rearrange
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.nn.init import normal_, zeros_, kaiming_uniform_

NF_times = 5

class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, *x): return torch.cat(*x, dim=self.dim)
    def __repr__(self): return f'{self.__class__.__name__}(dim={self.dim})'


def init_layer(layer, act_func=None, init='auto', bias_std=0.01):
    if hasattr(layer, 'bias') and layer.bias is not None:
        if bias_std != 0:
            normal_(layer.bias, mean=0., std=bias_std)
        else:
            zeros_(layer.bias)

    if init == 'auto':
        if act_func in [F.relu, F.leaky_relu]:
            init = kaiming_uniform_
        else:
            # 默认初始化方法，可以根据需要进行修改
            init = torch.nn.init.xavier_uniform_

    if callable(init):
        init(layer.weight)


def Conv1d(ni, nf, kernel_size=None, ks=None, stride=1, padding='same', dilation=1, init='auto', bias_std=0.01,
           **kwargs):
    "conv1d layer with padding='same', 'causal', 'valid', or any integer (defaults to 'same')"
    assert not (kernel_size and ks), 'use kernel_size or ks but not both simultaneously'
    assert kernel_size is not None or ks is not None, 'you need to pass a ks'
    kernel_size = kernel_size or ks
    if padding == 'same':
        if kernel_size % 2 == 1:
            conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=kernel_size // 2 * dilation, dilation=dilation,
                             **kwargs)
        else:
            conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, dilation=dilation,padding='same' , **kwargs)
    elif padding == 'valid':
        conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=0, dilation=dilation, **kwargs)
    else:
        conv = nn.Conv1d(ni, nf, kernel_size, stride=stride, padding=padding, dilation=dilation, **kwargs)
    init_layer(conv, None, init=init, bias_std=bias_std)
    return conv


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 32, num_heads: int = 4, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask = None):
        # split keys, queries and values in num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        att = att.mean(dim=1)
        return att  # .squeeze(1)


class SELayer1D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class InceptionModule(nn.Module):
    def __init__(self, ni, nf, ks = 20):
        super(InceptionModule, self).__init__()
        ks = [ks // (2 ** i) for i in range(4)]
        global NF_times
        NF_times = len(ks) + 1
        ks = [k if k % 2 != 0 else k - 1 for k in ks]  # ensure odd ks
        self.bottleneck = Conv1d(NF_times * nf, NF_times * nf, 1, bias=False)
        self.convs = nn.ModuleList([Conv1d(ni, nf, k, bias=False) for k in ks])
        self.maxconvpool = nn.Sequential(*[nn.MaxPool1d(3, stride=1, padding=1), Conv1d(ni, nf, 1, bias=False)])
        self.concat = Concat()
        self.MHSA = MultiHeadAttention(emb_size=nf)
        self.se = SELayer1D(channel=nf * NF_times, reduction=16)
        self.bn = nn.BatchNorm1d(nf * NF_times)
        self.act = nn.ReLU()

    def feature_map_weighting(self, feature_maps, att):
        updated_feature_maps = [torch.zeros_like(feature_map) for feature_map in feature_maps]
        for i in range(len(feature_maps)):
            for j in range(len(feature_maps)):
                att_expanded = att[:, i, j].unsqueeze(-1).unsqueeze(-1).expand_as(feature_maps[j])
                updated_feature_maps[i] = updated_feature_maps[i] + att_expanded * feature_maps[j]
        return updated_feature_maps

    def forward(self, x):
        input_tensor = x
        feature_maps = [l(x) for l in self.convs] + [self.maxconvpool(input_tensor)]
        gaps = [F.adaptive_avg_pool1d(t, 1).squeeze(-1) for t in feature_maps]
        concatenated = torch.cat([g.unsqueeze(1) for g in gaps], dim=1)

        att = self.MHSA(concatenated)
        feature_maps = self.feature_map_weighting(feature_maps, att)
        x = self.concat(feature_maps)
        x = self.se(x)
        x = self.bottleneck(x)
        return self.act(self.bn(x))


class InceptionBlock(nn.Module):
    def __init__(self, ni, nf=32, residual=True, depth=6,seq_len=2000, **kwargs):
        super(InceptionBlock, self).__init__()
        self.residual, self.depth = residual, depth
        self.inception, self.shortcut = nn.ModuleList(), nn.ModuleList()
        for d in range(depth):
            self.inception.append(InceptionModule(ni if d == 0 else nf * NF_times, nf,ks=int(seq_len/100), **kwargs))
            if self.residual and d % 3 == 2:
                n_in, n_out = ni if d == 2 else nf * NF_times, nf * NF_times
                self.shortcut.append(nn.BatchNorm1d(n_in) if n_in == n_out else Conv1d(n_in, n_out, 1))
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        for d, l in enumerate(range(self.depth)):
            x = self.inception[d](x)
            if self.residual and d % 3 == 2: res = x = self.act(x+self.shortcut[d // 3](res))
        return x


class GAP1d(nn.Module):
    "Global Adaptive Pooling + Flatten"
    def __init__(self, output_size=1):
        super(GAP1d, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(output_size)
    def forward(self, x):
        return torch.flatten(self.gap(x), start_dim=1)


class InceptionTime(nn.Module):
    def __init__(self, c_in, c_out, seq_len=None, nf=32, depth=6, **kwargs):
        super(InceptionTime, self).__init__()
        self.inceptionblock = InceptionBlock(c_in, nf, depth=depth,seq_len=seq_len, **kwargs)
        self.gap = GAP1d(1)
        self.fc = nn.Linear(nf * NF_times, c_out)

    def forward(self, x):
        x = self.inceptionblock(x)
        x = self.gap(x)
        x = self.fc(x)
        return x
