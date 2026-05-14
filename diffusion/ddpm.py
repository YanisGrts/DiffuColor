import torch
import torch.nn as nn
import math


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True))
        self.time_proj = nn.Linear(t_emb_dim, out_ch)  # inject t between
        self.conv2 = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1),
                                   nn.BatchNorm2d(out_ch),
                                   nn.ReLU(inplace=True))

    def forward(self, x, emb):
        x = self.conv1(x)
        # broadcast over H,W
        x = x + self.time_proj(emb).unsqueeze(-1).unsqueeze(-1)
        return self.conv2(x)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-torch.arange(half, device=t.device) *
                          (math.log(10000) / half))
        args = t[:, None].float() * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


class UNetDDPM(nn.Module):
    def __init__(self, t_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(t_emb_dim),
            nn.Linear(t_emb_dim, t_emb_dim),
            nn.ReLU(),
        )

        # Input: 2 (noisy ab) + 1 (L condition) = 3 channels
        self.enc1 = double_conv(3,   64,  t_emb_dim)
        self.enc2 = double_conv(64,  128, t_emb_dim)
        self.enc3 = double_conv(128, 256, t_emb_dim)
        self.enc4 = double_conv(256, 512, t_emb_dim)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = double_conv(512, 1024, t_emb_dim)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = double_conv(1024, 512, t_emb_dim)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = double_conv(512,  256, t_emb_dim)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = double_conv(256,  128, t_emb_dim)
        self.up1 = nn.ConvTranspose2d(128, 64,  2, stride=2)
        self.dec1 = double_conv(128,  64,  t_emb_dim)

        self.head = nn.Conv2d(64, 2, 1)  # predict 2-channel noise

    def forward(self, ab_noisy, L, t, drop_condition=None):
        if drop_condition is not None:
            # Zero out L for samples where drop_condition is True
            mask = drop_condition.view(-1, 1, 1, 1).float()
            L = L * (1.0 - mask)

        x = torch.cat([ab_noisy, L], dim=1)  # (B, 3, 128, 128)
        emb = self.time_mlp(t)                  # (B, t_emb_dim)

        e1 = self.enc1(x,  emb)
        e2 = self.enc2(self.pool(e1), emb)
        e3 = self.enc3(self.pool(e2), emb)
        e4 = self.enc4(self.pool(e3), emb)
        b = self.bottleneck(self.pool(e4), emb)

        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1), emb)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1), emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), emb)

        return self.head(d1)                       # (B, 2, 128, 128)
