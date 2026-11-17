import torch
import torch.nn as nn


def double_conv(in_ch, out_ch):
    """Dva konvoluciona sloja (3x3) + BatchNorm + ReLU — osnovni blok U-Net-a."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """U-Net: ulaz L (1 kanal) -> izlaz ab (2 kanala)."""

    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # enkoder (silazak)
        self.enc1 = double_conv(1, 64)            # 256x256
        self.enc2 = double_conv(64, 128)          # 128x128
        self.enc3 = double_conv(128, 256)         # 64x64
        self.bottleneck = double_conv(256, 512)   # 32x32

        # dekoder (uspon) — transponovana konvolucija duplira rezoluciju
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = double_conv(512, 256)         # 256 (up) + 256 (skip) = 512
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = double_conv(256, 128)         # 128 + 128 = 256
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = double_conv(128, 64)          # 64 + 64 = 128

        # izlazni sloj -> 2 kanala (a i b)
        self.out = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        # enkoder — cuvamo izlaze za skip veze
        e1 = self.enc1(x)                      # (64, 256, 256)
        e2 = self.enc2(self.pool(e1))          # (128, 128, 128)
        e3 = self.enc3(self.pool(e2))          # (256, 64, 64)
        b = self.bottleneck(self.pool(e3))     # (512, 32, 32)

        # dekoder — spajamo (concat) sa odgovarajucim skip vezama
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.tanh(self.out(d1))        # (2, 256, 256), opseg [-1, 1]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Uredjaj:", device)

    model = UNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Broj parametara: {n_params:,}")

    # test: propusti lazni batch kroz mrezu i proveri oblik izlaza
    dummy = torch.randn(2, 1, 256, 256, device=device)
    with torch.no_grad():
        out = model(dummy)
    print("Ulaz :", tuple(dummy.shape))
    print("Izlaz:", tuple(out.shape),
          "| opseg:", round(float(out.min()), 2), "do", round(float(out.max()), 2))