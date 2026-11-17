import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage import color
import matplotlib.pyplot as plt

from model import UNet
from model_klasifikacija import UNetKlasifikacija
from kvantizacija import korpa_u_ab, BROJ_KORPI

ULAZ = "porodicne_crnobele_slike"
IZLAZ = "rezultati_porodicne"
IMAGE_SIZE = 256
MAX_DIM = 1024          # ogranicenje pune rezolucije radi brzine
T = 0.38
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(IZLAZ, exist_ok=True)

m_reg = UNet().to(device)
m_reg.load_state_dict(torch.load("checkpoints/najbolji_regresija.pth", map_location=device))
m_reg.eval()

m_kla = UNetKlasifikacija().to(device)
m_kla.load_state_dict(torch.load("checkpoints/najbolji_klasifikacija_bal.pth", map_location=device))
m_kla.eval()

centri = torch.tensor(korpa_u_ab(np.arange(BROJ_KORPI)), dtype=torch.float32)

exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
putanje = [os.path.join(ULAZ, f) for f in sorted(os.listdir(ULAZ)) if f.lower().endswith(exts)]
print("Broj slika:", len(putanje))


def L_iz_sive(gray):
    """gray: (H,W) u [0,1] -> L kanal (H,W,1) u 0..100"""
    rgb = np.stack([gray] * 3, axis=2).astype(np.float32)
    return color.rgb2lab(rgb)[:, :, 0:1].astype(np.float32)


for p in putanje:
    img = Image.open(p).convert("L")          # cista siva (skida sepiju/izbledelost)
    w, h = img.size
    if max(w, h) > MAX_DIM:
        s = MAX_DIM / max(w, h)
        img = img.resize((int(w * s), int(h * s)))

    gray_full = np.asarray(img, dtype=np.float32) / 255.0
    L_full = L_iz_sive(gray_full)
    H, W = gray_full.shape

    # ulaz u mrezu na 256x256
    small = np.asarray(img.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32) / 255.0
    L_t = torch.from_numpy(L_iz_sive(small) / 100.0).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        ab_reg = m_reg(L_t) * 128.0
        logiti = m_kla(L_t)[0].float().cpu()
    probs = torch.softmax(logiti / T, dim=0)
    C, hh, ww = probs.shape
    ab_kla = (centri.t() @ probs.reshape(C, -1)).reshape(1, 2, hh, ww)

    # boju podigni na punu rezoluciju i spoji sa punim L (cuva detalje)
    def na_punu(ab_t):
        up = F.interpolate(ab_t.cpu(), size=(H, W), mode="bilinear", align_corners=False)
        return up[0].permute(1, 2, 0).numpy()

    rgb_reg = np.clip(color.lab2rgb(np.concatenate([L_full, na_punu(ab_reg)], axis=2)), 0, 1)
    rgb_kla = np.clip(color.lab2rgb(np.concatenate([L_full, na_punu(ab_kla)], axis=2)), 0, 1)

    ime = os.path.splitext(os.path.basename(p))[0]
    Image.fromarray((rgb_reg * 255).astype(np.uint8)).save(f"{IZLAZ}/{ime}_regresija.png")
    Image.fromarray((rgb_kla * 255).astype(np.uint8)).save(f"{IZLAZ}/{ime}_klasifikacija.png")

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    ax[0].imshow(gray_full, cmap="gray"); ax[0].set_title("Original (crno-bela)")
    ax[1].imshow(rgb_reg);                ax[1].set_title("Regresija")
    ax[2].imshow(rgb_kla);                ax[2].set_title("Klasifikacija")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(f"{IZLAZ}/{ime}_poredjenje.png", dpi=110)
    plt.close()
    print("  gotovo:", ime)

print(f"Gotovo. Rezultati u folderu '{IZLAZ}/'")