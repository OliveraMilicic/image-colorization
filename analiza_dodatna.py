import os
import csv
import numpy as np
import torch
from torch.utils.data import random_split
from PIL import Image
from skimage import color
import matplotlib.pyplot as plt

from dataset import ColorizationDataset
from kvantizacija import ab_u_korpu, BROJ_KORPI

os.makedirs("rezultati_eval", exist_ok=True)

METODE = ["regresija", "klasifikacija"]
NAZIVI = {"siva": "Неутрална (сива) основа", "regresija": "Регресиони приступ", "klasifikacija": "Класификациони приступ"}
BOJE = {"siva": "#9e9e9e", "regresija": "#4c72b0", "klasifikacija": "#dd8452"}

with open("rezultati_eval/metrike.csv", newline="", encoding="utf-8") as f:
    redovi = list(csv.DictReader(f))


def kolona(naziv):
    return np.array([float(r[naziv]) for r in redovi])


# =====================================================================
# 1) PSNR naspram SSIM, po slici, obojeno po metodi
# =====================================================================
plt.figure(figsize=(6.5, 5.5))
for m in ["siva"] + METODE:
    plt.scatter(kolona(f"{m}_ssim"), kolona(f"{m}_psnr"), s=14, alpha=0.5, color=BOJE[m], label=NAZIVI[m])
plt.xlabel("SSIM")
plt.ylabel("PSNR (dB)")
plt.ylim(top=45)
plt.title("Однос SSIM и PSNR по слици")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rezultati_eval/psnr_vs_ssim.png", dpi=130)
plt.close()
print("Sacuvano: rezultati_eval/psnr_vs_ssim.png")

# =====================================================================
# 2) broj aktivnih korpi boja po slici (isti 500-slika uzorak kao u evaluacija.py)
# =====================================================================
DATA_DIR = "train2017"
IMAGE_SIZE = 256
N_SLIKA = 500

full = ColorizationDataset(DATA_DIR, image_size=IMAGE_SIZE)
full.paths = full.paths[:40000]
n_val = int(len(full) * 0.1); n_train = len(full) - n_val
_, val_set = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(42))
indices = list(val_set.indices)[:N_SLIKA]

brojevi_korpi = []
for idx in indices:
    img = Image.open(full.paths[idx]).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    lab = color.rgb2lab(rgb).astype(np.float32)
    korpe = ab_u_korpu(lab[:, :, 1:3])
    brojevi_korpi.append(len(np.unique(korpe)))

brojevi_korpi = np.array(brojevi_korpi)
plt.figure(figsize=(7, 4.5))
plt.hist(brojevi_korpi, bins=30, color="#4c72b0", edgecolor="white")
plt.axvline(brojevi_korpi.mean(), color="#dd8452", linestyle="--",
            label=f"средња вредност = {brojevi_korpi.mean():.1f}")
plt.xlabel("Број различитих активних корпи по слици")
plt.ylabel("Број слика")
plt.title(f"Расподела броја активних корпи боја по слици (од укупно {BROJ_KORPI})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rezultati_eval/broj_aktivnih_korpi.png", dpi=130)
plt.close()
print("Sacuvano: rezultati_eval/broj_aktivnih_korpi.png")
print(f"Prosecan broj aktivnih korpi: {brojevi_korpi.mean():.2f} (medijana {np.median(brojevi_korpi):.0f}, od {BROJ_KORPI} ukupno)")

# =====================================================================
# 3) razlika u odnosu na sivu osnovu, po slici (histogram)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for mera, ax, naslov in zip(["psnr", "ssim"], axes, ["ΔPSNR (dB)", "ΔSSIM"]):
    for m in METODE:
        delta = kolona(f"{m}_{mera}") - kolona(f"siva_{mera}")
        ax.hist(delta, bins=30, alpha=0.55, color=BOJE[m], label=NAZIVI[m])
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel(naslov)
    ax.set_ylabel("Број слика")
    ax.grid(True, alpha=0.3)
    ax.legend()
fig.suptitle("Разлика у односу на неутралну (сиву) основу, по слици")
plt.tight_layout()
plt.savefig("rezultati_eval/delta_vs_siva.png", dpi=130)
plt.close()
print("Sacuvano: rezultati_eval/delta_vs_siva.png")

for m in METODE:
    d_psnr = kolona(f"{m}_psnr") - kolona("siva_psnr")
    d_ssim = kolona(f"{m}_ssim") - kolona("siva_ssim")
    print(f"{m}: PSNR > siva u {100*(d_psnr>0).mean():.1f}% slika | SSIM > siva u {100*(d_ssim>0).mean():.1f}% slika")
