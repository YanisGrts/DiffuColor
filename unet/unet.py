import torch
import torch.nn as nn


def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = double_conv(1, 64)
        self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256)
        self.enc4 = double_conv(256, 512)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = double_conv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = double_conv(1024, 512)  # 512 + 512 skip

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = double_conv(512, 256)   # 256 + 256 skip

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = double_conv(256, 128)   # 128 + 128 skip

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = double_conv(128, 64)    # 64 + 64 skip

        # Output: predict ab channels
        self.head = nn.Sequential(
            nn.Conv2d(64, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)                  # (B,  64, 128, 128)
        e2 = self.enc2(self.pool(e1))      # (B, 128,  64,  64)
        e3 = self.enc3(self.pool(e2))      # (B, 256,  32,  32)
        e4 = self.enc4(self.pool(e3))      # (B, 512,  16,  16)

        # Bottleneck
        b = self.bottleneck(self.pool(e4)) # (B,1024,   8,   8)

        # Decode
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))  # (B, 512, 16, 16)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # (B, 256, 32, 32)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 128, 64, 64)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B,  64,128,128)

        return self.head(d1)                                   # (B,   2,128,128)
