import os
import csv
import numpy as np
import torch
from torch.utils.data import random_split
from PIL import Image
from skimage import color
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
import matplotlib.pyplot as plt

from dataset import ColorizationDataset
from model import UNet
from model_klasifikacija import UNetKlasifikacija
from kvantizacija import korpa_u_ab, BROJ_KORPI

# ---- podesavanja ----
DATA_DIR = "train2017"
IMAGE_SIZE = 256
N_SLIKA = 500          # koliko validacionih slika za evaluaciju
T = 0.38               # temperatura za annealed-mean dekodiranje
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("rezultati_eval", exist_ok=True)

# ---- ISTI validacioni skup kao pri treningu (isti seed i ista podela) ----
full = ColorizationDataset(DATA_DIR, image_size=IMAGE_SIZE)
full.paths = full.paths[:40000]
n_val = int(len(full) * 0.1); n_train = len(full) - n_val
_, val_set = random_split(full, [n_train, n_val],
                          generator=torch.Generator().manual_seed(42))
indices = list(val_set.indices)[:N_SLIKA]
print(f"Evaluacija na {len(indices)} validacionih slika (model ih nije video)")

# ---- ucitaj oba modela ----
m_reg = UNet().to(device)
m_reg.load_state_dict(torch.load("checkpoints/najbolji_regresija.pth", map_location=device))
m_reg.eval()

m_kla = UNetKlasifikacija().to(device)
m_kla.load_state_dict(torch.load("checkpoints/najbolji_klasifikacija_bal.pth", map_location=device))
m_kla.eval()


centri = torch.tensor(korpa_u_ab(np.arange(BROJ_KORPI)), dtype=torch.float32)


# ---- pomocne funkcije ----
def ucitaj(path):
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    lab = color.rgb2lab(rgb).astype(np.float32)
    return rgb, lab


def sklopi(L_np, ab):
    """L_np: (H,W,1) u 0..100, ab: (H,W,2) -> RGB u [0,1]"""
    return np.clip(color.lab2rgb(np.concatenate([L_np, ab], axis=2)), 0, 1)


def zasicenost(rgb):
    """Mera 'sarenila' slike: veca vrednost = zivlje boje."""
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    return (np.sqrt(rg.std()**2 + yb.std()**2)
            + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2))


def metrike(orig, pred):
    return (psnr_fn(orig, pred, data_range=1.0),
            ssim_fn(orig, pred, data_range=1.0, channel_axis=2),
            zasicenost(pred))


# ---- glavna petlja ----
rezultati = {"siva": [], "regresija": [], "klasifikacija": []}
po_slici = []

for k, idx in enumerate(indices):
    rgb_orig, lab = ucitaj(full.paths[idx])
    L_np = lab[:, :, 0:1]                                  # (H,W,1) 0..100
    L_t = torch.from_numpy(L_np / 100.0).permute(2, 0, 1).unsqueeze(0).to(device)

    # 1) siva osnova: bez boje (ab = 0)
    rgb_siva = sklopi(L_np, np.zeros_like(lab[:, :, 1:3]))

    # 2) regresija
    with torch.no_grad():
        ab_reg = m_reg(L_t)[0].cpu().numpy().transpose(1, 2, 0) * 128.0
    rgb_reg = sklopi(L_np, ab_reg)


    # 3) klasifikacija (annealed-mean dekodiranje)
    with torch.no_grad():
        logiti = m_kla(L_t)[0].float().cpu()
    probs = torch.softmax(logiti / T, dim=0)
    C, H, W = probs.shape
    ab_kla = (centri.t() @ probs.reshape(C, -1)).reshape(2, H, W).permute(1, 2, 0).numpy()
    rgb_kla = sklopi(L_np, ab_kla)

    red = {"slika": os.path.basename(full.paths[idx])}
    for naziv, slika in [("siva", rgb_siva), ("regresija", rgb_reg), ("klasifikacija", rgb_kla)]:
        p, s, z = metrike(rgb_orig, slika)
        rezultati[naziv].append((p, s, z))
        red[f"{naziv}_psnr"] = round(p, 3)
        red[f"{naziv}_ssim"] = round(s, 4)
        red[f"{naziv}_zasicenost"] = round(z, 3)
    po_slici.append(red)

    if (k + 1) % 100 == 0:
        print(f"  obradjeno {k+1}/{len(indices)}")

# ---- tabela ----
print("\n" + "=" * 62)
print(f"{'Metoda':<16}{'PSNR (dB)':>12}{'SSIM':>12}{'Zasicenost':>14}")
print("-" * 62)
for naziv in ["siva", "regresija", "klasifikacija"]:
    a = np.array(rezultati[naziv])
    print(f"{naziv:<16}{a[:,0].mean():>12.2f}{a[:,1].mean():>12.4f}{a[:,2].mean():>14.3f}")
print("=" * 62)
print("(PSNR i SSIM: vece = vernije originalu | Zasicenost: vece = zivlje boje)")

# zasicenost originala kao referenca
zas_orig = np.mean([zasicenost(ucitaj(full.paths[i])[0]) for i in indices[:100]])
print(f"\nZasicenost originalnih slika (referenca): {zas_orig:.3f}")

# ---- CSV za rad ----
with open("rezultati_eval/metrike.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(po_slici[0].keys()))
    w.writeheader(); w.writerows(po_slici)
print("Sacuvano: rezultati_eval/metrike.csv")

# ---- zajednicka figura: Original / L / Regresija / Klasifikacija ----
n = 5
fig, ax = plt.subplots(n, 4, figsize=(13, 3.2 * n))
for i in range(n):
    rgb_orig, lab = ucitaj(full.paths[indices[i]])
    L_np = lab[:, :, 0:1]
    L_t = torch.from_numpy(L_np / 100.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        ab_reg = m_reg(L_t)[0].cpu().numpy().transpose(1, 2, 0) * 128.0
        logiti = m_kla(L_t)[0].float().cpu()
    probs = torch.softmax(logiti / T, dim=0)
    C, H, W = probs.shape
    ab_kla = (centri.t() @ probs.reshape(C, -1)).reshape(2, H, W).permute(1, 2, 0).numpy()

    ax[i, 0].imshow(rgb_orig);                    ax[i, 0].set_title("Оригинал")
    ax[i, 1].imshow(L_np[:, :, 0], cmap="gray");  ax[i, 1].set_title("Улаз (L)")
    ax[i, 2].imshow(sklopi(L_np, ab_reg));        ax[i, 2].set_title("Регресија")
    ax[i, 3].imshow(sklopi(L_np, ab_kla));        ax[i, 3].set_title("Класификација")
for a in ax.ravel():
    a.axis("off")
plt.tight_layout()
plt.savefig("rezultati_eval/poredjenje.png", dpi=120)
print("Sacuvano: rezultati_eval/poredjenje.png")