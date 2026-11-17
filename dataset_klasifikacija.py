import os
import numpy as np
from PIL import Image
from skimage import color
import torch
from torch.utils.data import Dataset

from kvantizacija import ab_u_korpu, BROJ_KORPI


class KlasifikacijaDataset(Dataset):
    """Vraca (L, korpa): L je ulaz, korpa je mapa indeksa korpi (cilj klasifikacije)."""

    def __init__(self, folder, image_size=256):
        self.image_size = image_size
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        self.paths = [
            os.path.join(folder, f)
            for f in sorted(os.listdir(folder))
            if f.lower().endswith(exts)
        ]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB").resize((self.image_size, self.image_size))
        rgb = np.asarray(img, dtype=np.float32) / 255.0
        lab = color.rgb2lab(rgb).astype(np.float32)

        L = lab[:, :, 0:1] / 100.0            # (H,W,1) u [0,1]  -> ULAZ (kao pre)
        ab = lab[:, :, 1:3]                    # (H,W,2) SIROVE vrednosti (NE delimo sa 128!)

        korpa = ab_u_korpu(ab)                 # (H,W) indeks korpe za svaki piksel -> CILJ

        L = torch.from_numpy(L).permute(2, 0, 1)   # (1,H,W)
        korpa = torch.from_numpy(korpa).long()     # (H,W) celi brojevi
        return L, korpa


if __name__ == "__main__":
    DATA_DIR = "val2017"          # ili "data"
    ds = KlasifikacijaDataset(DATA_DIR, image_size=256)
    print("Broj slika:", len(ds))

    L, korpa = ds[0]
    print("L oblik    :", tuple(L.shape), "| opseg:", round(float(L.min()), 2), "-", round(float(L.max()), 2))
    print("korpa oblik:", tuple(korpa.shape), "| dtype:", korpa.dtype)
    print("korpa opseg:", int(korpa.min()), "-", int(korpa.max()), f"(mora izmedju 0 i {BROJ_KORPI - 1})")
    print("Razlicitih korpi u slici:", len(torch.unique(korpa)))