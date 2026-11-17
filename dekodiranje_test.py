import os
import numpy as np
import torch
from torch.utils.data import random_split
from skimage import color
import matplotlib.pyplot as plt

from dataset_klasifikacija import KlasifikacijaDataset
from model_klasifikacija import UNetKlasifikacija
from kvantizacija import korpa_u_ab, BROJ_KORPI

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("rezultati_klas", exist_ok=True)

# isti val skup kao pri treningu (isti seed!)
full = KlasifikacijaDataset("val2017", image_size=256)
n_val = int(len(full) * 0.1); n_train = len(full) - n_val
_, val_set = random_split(full, [n_train, n_val],
                          generator=torch.Generator().manual_seed(42))

# ucitaj istrenirani model
model = UNetKlasifikacija().to(device)
model.load_state_dict(torch.load("checkpoints/najbolji_klasifikacija_bal.pth", map_location=device))
model.eval()

# centri svih korpi kao ab vektori: (484, 2)
centri = torch.tensor(korpa_u_ab(np.arange(BROJ_KORPI)), dtype=torch.float32)


def sklopi(L, ab):
    L_np = L.cpu().numpy().transpose(1, 2, 0) * 100.0
    lab = np.concatenate([L_np, ab], axis=2)
    return np.clip(color.lab2rgb(lab), 0, 1)


def dekodiraj_argmax(L, logiti):                       # STARO
    korpe = logiti.argmax(dim=0).cpu().numpy()
    return sklopi(L, korpa_u_ab(korpe).astype(np.float32))


def dekodiraj_mean(L, logiti, T=0.3):                 # NOVO (annealed-mean)
    probs = torch.softmax(logiti.float().cpu() / T, dim=0)     # (C,H,W)
    C, H, W = probs.shape
    ab = (centri.t() @ probs.reshape(C, -1)).reshape(2, H, W)  # ponderisani prosek
    return sklopi(L, ab.permute(1, 2, 0).numpy())


n = 4
fig, ax = plt.subplots(n, 3, figsize=(9, 3 * n))
with torch.no_grad():
    for i in range(n):
        L, korpa = val_set[i]
        logiti = model(L.unsqueeze(0).to(device))[0]
        orig = sklopi(L, korpa_u_ab(korpa.numpy()).astype(np.float32))
        ax[i, 0].imshow(orig);                        ax[i, 0].set_title("Оригинал")
        ax[i, 1].imshow(dekodiraj_argmax(L, logiti)); ax[i, 1].set_title("Избор највероватније корпе")
        ax[i, 2].imshow(dekodiraj_mean(L, logiti));   ax[i, 2].set_title("Пондерисана средина")
for a in ax.ravel():
    a.axis("off")
plt.tight_layout()
plt.savefig("rezultati_klas/dekodiranje_poredjenje.png", dpi=100)
print("Sacuvano: rezultati_klas/dekodiranje_poredjenje.png")