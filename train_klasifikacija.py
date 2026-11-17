import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from skimage import color
import matplotlib.pyplot as plt

from dataset_klasifikacija import KlasifikacijaDataset
from model_klasifikacija import UNetKlasifikacija
from kvantizacija import korpa_u_ab

# ---- podesavanja ----
DATA_DIR = "train2017"
IMAGE_SIZE = 256
BATCH_SIZE = 12          # klasifikacioni izlaz trosi vise memorije -> malo manji batch
EPOCHS = 12
LR = 2e-4
VAL_UDEO = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = (device == "cuda")
print("Uredjaj:", device, "| mixed precision:", use_amp)

os.makedirs("rezultati_klas_bal", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# ---- podaci ----
full = KlasifikacijaDataset(DATA_DIR, image_size=IMAGE_SIZE)
full.paths = full.paths[:40000]   #koristimo prvih 40k slika
n_val = int(len(full) * VAL_UDEO)
n_train = len(full) - n_val
train_set, val_set = random_split(
    full, [n_train, n_val], generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
print(f"Trening: {n_train} | Validacija: {n_val}")

model = UNetKlasifikacija().to(device)
tezine = torch.from_numpy(np.load("tezine_klasa.npy")).to(device)
criterion = nn.CrossEntropyLoss(weight=tezine)      # balansiranje retkih boja
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


# ---- iz predvidjenih logita (C,H,W) napravi RGB sliku ----
def logiti_u_rgb(L, logiti):
    korpe = logiti.argmax(dim=0).cpu().numpy()          # (H,W) najverovatnija korpa
    ab = korpa_u_ab(korpe).astype(np.float32)           # (H,W,2) centar te korpe
    L_np = L.cpu().numpy().transpose(1, 2, 0) * 100.0    # (H,W,1)
    lab = np.concatenate([L_np, ab], axis=2)
    return np.clip(color.lab2rgb(lab), 0, 1)


# ---- iz prave korpe (H,W) napravi RGB (za "Original" u prikazu) ----
def korpa_u_rgb(L, korpa):
    ab = korpa_u_ab(korpa.numpy()).astype(np.float32)
    L_np = L.cpu().numpy().transpose(1, 2, 0) * 100.0
    lab = np.concatenate([L_np, ab], axis=2)
    return np.clip(color.lab2rgb(lab), 0, 1)


def sacuvaj_primere(epoch, n=4):
    model.eval()
    fig, ax = plt.subplots(n, 3, figsize=(9, 3 * n))
    with torch.no_grad():
        for i in range(n):
            L, korpa = val_set[i]
            logiti = model(L.unsqueeze(0).to(device))[0]
            ax[i, 0].imshow(korpa_u_rgb(L, korpa));            ax[i, 0].set_title("Original (kvantizovan)")
            ax[i, 1].imshow(L.squeeze().numpy(), cmap="gray"); ax[i, 1].set_title("Ulaz (L)")
            ax[i, 2].imshow(logiti_u_rgb(L, logiti));          ax[i, 2].set_title("Model")
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout()
    plt.savefig(f"rezultati_klas_bal/val_epoha_{epoch:03d}.png", dpi=100)
    plt.close()
    model.train()


def validacija():
    model.eval()
    ukupno = 0.0
    with torch.no_grad():
        for L, korpa in val_loader:
            L, korpa = L.to(device), korpa.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                ukupno += criterion(model(L), korpa).item()
    model.train()
    return ukupno / len(val_loader)


# ---- petlja ----
train_g, val_g = [], []
najbolji = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    ukupno = 0.0
    for L, korpa in train_loader:
        L, korpa = L.to(device), korpa.to(device)      # korpa: (B,H,W) celi brojevi
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = criterion(model(L), korpa)           # <-- promena 1: cross-entropy
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ukupno += loss.item()
    tr = ukupno / len(train_loader)
    val = validacija()
    train_g.append(tr); val_g.append(val)
    print(f"Epoha {epoch:2d}/{EPOCHS}  |  trening: {tr:.4f}  |  validacija: {val:.4f}")

    sacuvaj_primere(epoch)
    if val < najbolji:
        najbolji = val
        torch.save(model.state_dict(), "checkpoints/najbolji_klasifikacija_bal.pth")

# ---- kriva gubitka ----
plt.figure(figsize=(7, 4))
plt.plot(train_g, label="trening")
plt.plot(val_g, label="validacija")
plt.xlabel("Epoha"); plt.ylabel("Gubitak (cross-entropy)"); plt.title("Kriva gubitka — klasifikacija")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("rezultati_klas_bal/kriva_gubitka.png", dpi=100)
plt.close()

torch.save(model.state_dict(), "model_klasifikacija.pth")
print(f"Gotovo. Najbolji validacioni gubitak: {najbolji:.4f}")