import os
import numpy as np
from PIL import Image
from skimage import color
import torch
from torch.utils.data import Dataset


class ColorizationDataset(Dataset):
    """Ucitava slike iz foldera i vraca par (L, ab) spreman za mrezu."""

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
        # 1. ucitaj sliku, osiguraj 3 kanala (RGB) i fiksnu velicinu
        img = Image.open(self.paths[idx]).convert("RGB")
        img = img.resize((self.image_size, self.image_size))
        rgb = np.asarray(img, dtype=np.float32) / 255.0     # (H, W, 3), 0-1

        # 2. RGB -> Lab, pa razdvoji na L (ulaz) i ab (cilj)
        lab = color.rgb2lab(rgb).astype(np.float32)
        L = lab[:, :, 0:1]          # (H, W, 1),  0..100
        ab = lab[:, :, 1:3]         # (H, W, 2), ~ -128..127

        # 3. normalizacija u male, centrirane opsege
        L = L / 100.0               # -> [0, 1]
        ab = ab / 128.0             # -> ~ [-1, 1]

        # 4. u tensore, format (kanali, visina, sirina)
        L = torch.from_numpy(L).permute(2, 0, 1)     # (1, H, W)
        ab = torch.from_numpy(ab).permute(2, 0, 1)   # (2, H, W)
        return L, ab


# ---- test pipeline-a ----
if __name__ == "__main__":
    dataset = ColorizationDataset("data", image_size=256)
    print("Broj slika u skupu:", len(dataset))

    if len(dataset) == 0:
        print("Folder 'data' je prazan. Ubaci par kolor slika pa pokreni ponovo.")
        raise SystemExit

    # a) oblici i opsezi jednog primera
    L, ab = dataset[0]
    print("L  oblik:", tuple(L.shape),  "| opseg:", round(float(L.min()), 2),  "-", round(float(L.max()), 2))
    print("ab oblik:", tuple(ab.shape), "| opseg:", round(float(ab.min()), 2), "-", round(float(ab.max()), 2))

    # b) rekonstrukcija: iz (L, ab) nazad u RGB -> dokaz da pipeline radi ispravno
    import matplotlib.pyplot as plt

    L_np = L.permute(1, 2, 0).numpy() * 100.0
    ab_np = ab.permute(1, 2, 0).numpy() * 128.0
    lab_rec = np.concatenate([L_np, ab_np], axis=2)
    rgb_rec = color.lab2rgb(lab_rec)

    orig = Image.open(dataset.paths[0]).convert("RGB").resize((256, 256))

    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    ax[0].imshow(orig);                              ax[0].set_title("Original")
    ax[1].imshow(L.squeeze().numpy(), cmap="gray");  ax[1].set_title("L (ulaz u mrezu)")
    ax[2].imshow(rgb_rec);                            ax[2].set_title("Rekonstrukcija iz L+ab")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig("dataset_test.png", dpi=100)
    print("Sacuvano: dataset_test.png")

    # c) provera batch-a preko DataLoader-a
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    L_batch, ab_batch = next(iter(loader))
    print("Batch L :", tuple(L_batch.shape))
    print("Batch ab:", tuple(ab_batch.shape))