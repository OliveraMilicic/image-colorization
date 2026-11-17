import torch
import torch.nn as nn

from kvantizacija import BROJ_KORPI    # 484


def double_conv(in_ch, out_ch):
    """Dva konvoluciona sloja (3x3) + BatchNorm + ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class UNetKlasifikacija(nn.Module):
    """U-Net: ulaz L (1 kanal) -> izlaz BROJ_KORPI kanala (verovatnoce korpi)."""

    def __init__(self, broj_korpi=BROJ_KORPI):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        # enkoder (isto kao pre)
        self.enc1 = double_conv(1, 64)
        self.enc2 = double_conv(64, 128)
        self.enc3 = double_conv(128, 256)
        self.bottleneck = double_conv(256, 512)

        # dekoder (isto kao pre)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = double_conv(128, 64)

        # JEDINA promena: izlaz -> broj_korpi kanala, bez tanh
        self.out = nn.Conv2d(64, broj_korpi, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)     # (broj_korpi, 256, 256) — sirovi logiti, bez tanh


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Uredjaj:", device)

    model = UNetKlasifikacija().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Broj parametara: {n_params:,}")
    print(f"Broj izlaznih kanala (korpi): {BROJ_KORPI}")

    dummy = torch.randn(2, 1, 256, 256, device=device)
    with torch.no_grad():
        out = model(dummy)
    print("Ulaz :", tuple(dummy.shape))
    print("Izlaz:", tuple(out.shape))     # ocekujemo (2, 484, 256, 256)