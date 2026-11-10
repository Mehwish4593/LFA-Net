"""LFA-Net in PyTorch, a faithful port of lfanet.py (which is the notebook).

Keras is NHWC and normalises LayerNorm over the channel axis; PyTorch is NCHW.
Every layer below mirrors its Keras counterpart exactly, and load_from_keras()
copies the released weights across so the port can be checked numerically
rather than trusted. See verify_port.py.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelLayerNorm(nn.Module):
    """Keras LayerNormalization(axis=-1) on NHWC == normalise over C in NCHW."""

    def __init__(self, c, eps=1e-3):
        super().__init__()
        self.g = nn.Parameter(torch.ones(c))
        self.b = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = x.var(1, unbiased=False, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.g[None, :, None, None] + self.b[None, :, None, None]


def gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / np.sqrt(2.0)))


class Up(nn.Module):
    """Keras Conv2DTranspose(padding='same', stride=2).

    TensorFlow's 'same' transposed convolution is NOT PyTorch's
    padding=1/output_padding=1: it is padding=0 with the result cropped to
    exactly twice the input size from the top left. Verified against the Keras
    layer at 1.9e-06; the padding=1 form is off by 14.7.
    """

    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.conv = nn.ConvTranspose2d(cin, cout, k, stride=2, padding=0, output_padding=0)

    def forward(self, x):
        return self.conv(x)[..., :x.shape[-2] * 2, :x.shape[-1] * 2]


class ConvBlock(nn.Module):
    """1x1 + 3x3 + 3x3-dilated, no bias, summed, LeakyReLU(0.1)."""

    def __init__(self, cin, cout):
        super().__init__()
        self.a = nn.Conv2d(cin, cout, 1, padding=0, bias=False)
        self.b = nn.Conv2d(cin, cout, 3, padding=1, bias=False)
        self.c = nn.Conv2d(cin, cout, 3, padding=2, dilation=2, bias=False)

    def forward(self, x):
        return F.leaky_relu(self.a(x) + self.b(x) + self.c(x), 0.1)


class ContextAggregation(nn.Module):
    """focal_modulation_context_aggregation + focal_modulation, as written."""

    def __init__(self, cin, filters, gamma=2.0, alpha=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, filters, 3, padding=1)
        self.conv2 = nn.Conv2d(cin, filters, 1)
        self.gate = nn.Conv2d(filters, filters, 1)
        self.focal = nn.Conv2d(filters, filters, 1)
        self.gamma, self.alpha = gamma, alpha

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        c2 = F.relu(self.conv2(x))
        g = c2.mean((2, 3), keepdim=True)
        g = torch.sigmoid(self.gate(g))
        s = c1 * g                                            # gated context
        mean = s.mean((2, 3), keepdim=True)
        mx = s.amax((2, 3), keepdim=True)
        mod = torch.sigmoid(self.focal((mx - mean) * self.alpha))
        fm = (s * mod) ** self.gamma
        return torch.cat([c1, fm], 1)


class VisionMamba(nn.Module):
    def __init__(self, dim, p=0.1):
        super().__init__()
        self.norm1 = ChannelLayerNorm(dim)
        self.dw = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = ChannelLayerNorm(dim)
        self.fc1 = nn.Conv2d(dim, dim * 4, 1)                 # Keras Dense on NHWC
        self.fc2 = nn.Conv2d(dim * 4, dim, 1)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        sc = x
        x = self.norm1(x)
        x = self.drop(gelu(self.dw(x))) + sc
        sc = x
        x = self.norm2(x)
        x = self.drop(self.fc2(self.drop(gelu(self.fc1(x))))) + sc
        return x


class LiteFusionAttention(nn.Module):
    def __init__(self, cin, filters):
        super().__init__()
        self.proj_in = nn.Conv2d(cin, filters, 1)
        self.norm = ChannelLayerNorm(filters)
        self.conv = nn.Conv2d(filters, filters, 3, padding=1)
        self.ctx = ContextAggregation(filters, filters)
        self.proj_out = nn.Conv2d(filters * 2, filters, 1)
        self.mamba = VisionMamba(filters)

    def forward(self, x):
        t = self.proj_in(x)
        y = self.conv(self.norm(t))
        y = self.ctx(y)
        y = self.proj_out(y)
        return self.mamba(y + t)


class RAAttention(nn.Module):
    """RA_attention_block(inputs, n_classes=1, k=16)."""

    def __init__(self, cin, n_classes=1, k=16):
        super().__init__()
        self.n, self.k = n_classes, k
        self.conv = nn.Conv2d(cin, k * n_classes, 3, padding=1)
        self.bn = nn.BatchNorm2d(k * n_classes, eps=1e-3, momentum=0.01)

    def forward(self, x):
        f1 = F.relu(self.bn(self.conv(x)))                    # B, n*k, H, W
        B, _, H, W = f1.shape
        x1 = f1.amax((2, 3))                                  # global max  -> B, n*k
        x2 = f1.mean((2, 3))                                  # global avg
        s = (x1 * x2).reshape(B, self.n, self.k).mean(-1)     # B, n
        m = f1.reshape(B, self.n, self.k, H, W).mean(2)       # B, n, H, W
        m = m * s[:, :, None, None]
        m = m.mean(1, keepdim=True)                           # B, 1, H, W
        return x * m


class LFANet(nn.Module):
    def __init__(self, feature_scale=2):
        super().__init__()
        f = [x // feature_scale for x in (16, 32, 64)]        # 8, 16, 32
        self.cb1, self.bn1 = ConvBlock(3, f[0]), nn.BatchNorm2d(f[0], eps=1e-3, momentum=0.01)
        self.cb2, self.bn2 = ConvBlock(f[0], f[1]), nn.BatchNorm2d(f[1], eps=1e-3, momentum=0.01)
        self.cb3, self.bn3 = ConvBlock(f[1], f[2]), nn.BatchNorm2d(f[2], eps=1e-3, momentum=0.01)
        self.att = LiteFusionAttention(f[2], 32)
        self.ra1 = RAAttention(32)
        self.up1 = Up(32 + 32, f[2])
        self.ra2 = RAAttention(f[1])
        self.cv1 = nn.Conv2d(f[1] + f[2], f[2], 3, padding=1)
        self.up2 = Up(f[2], f[2])
        self.ra3 = RAAttention(f[0])
        self.cv2 = nn.Conv2d(f[0] + f[2], f[2], 3, padding=1)
        self.up3 = Up(f[2], f[0])
        self.cv3 = nn.Conv2d(f[0], f[0], 3, padding=1)
        self.out = nn.Conv2d(f[0], 1, 1)

    def forward(self, x):
        c1 = self.bn1(F.max_pool2d(self.cb1(x), 2))
        c2 = self.bn2(F.max_pool2d(self.cb2(c1), 2))
        c3 = self.bn3(F.max_pool2d(self.cb3(c2), 2))
        a = self.att(c3)
        d = F.relu(self.up1(torch.cat([self.ra1(a), a], 1)))
        d = F.relu(self.cv1(torch.cat([self.ra2(c2), d], 1)))
        d = F.relu(self.up2(d))
        d = F.relu(self.cv2(torch.cat([self.ra3(c1), d], 1)))
        d = F.relu(self.up3(d))
        d = F.relu(self.cv3(d))
        return torch.sigmoid(self.out(d))
