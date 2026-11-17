import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from skimage import color
import matplotlib.pyplot as plt

from dataset import ColorizationDataset
from model import UNet

# ---- podesavanja ----
DATA_DIR = "train2017"       # <-- putanja do foldera sa COCO slikama
IMAGE_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 12
LR = 2e-4
VAL_UDEO = 0.1             # 10% slika za validaciju
device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = (device == "cuda")
print("Uredjaj:", device, "| mixed precision:", use_amp)

os.makedirs("rezultati", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# ---- podaci: podela na trening i validaciju ----
full = ColorizationDataset(DATA_DIR, image_size=IMAGE_SIZE)
full.paths = full.paths[:40000]   #koristimo prvih 40k slika
n_val = int(len(full) * VAL_UDEO)
n_train = len(full) - n_val
train_set, val_set = random_split(
    full, [n_train, n_val], generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
print(f"Trening: {n_train} slika | Validacija: {n_val} slika")

model = UNet().to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


def to_rgb(L, ab):
    L = L.cpu().numpy().transpose(1, 2, 0) * 100.0
    ab = ab.cpu().numpy().transpose(1, 2, 0) * 128.0
    return np.clip(color.lab2rgb(np.concatenate([L, ab], axis=2)), 0, 1)


def sacuvaj_primere(epoch, n=4):
    model.eval()
    fig, ax = plt.subplots(n, 3, figsize=(9, 3 * n))
    with torch.no_grad():
        for i in range(n):
            L, ab_true = val_set[i]
            ab_pred = model(L.unsqueeze(0).to(device))[0]
            ax[i, 0].imshow(to_rgb(L, ab_true));               ax[i, 0].set_title("Original")
            ax[i, 1].imshow(L.squeeze().numpy(), cmap="gray"); ax[i, 1].set_title("Ulaz (L)")
            ax[i, 2].imshow(to_rgb(L, ab_pred));               ax[i, 2].set_title("Model")
    for a in ax.ravel():
        a.axis("off")
    plt.tight_layout()
    plt.savefig(f"rezultati/val_epoha_{epoch:03d}.png", dpi=100)
    plt.close()
    model.train()


def validacija():
    model.eval()
    ukupno = 0.0
    with torch.no_grad():
        for L, ab in val_loader:
            L, ab = L.to(device), ab.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                ukupno += criterion(model(L), ab).item()
    model.train()
    return ukupno / len(val_loader)


# ---- petlja treniranja ----
train_gubici, val_gubici = [], []
najbolji_val = float("inf")

for epoch in range(1, EPOCHS + 1):
    model.train()
    ukupno = 0.0
    for L, ab in train_loader:
        L, ab = L.to(device), ab.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = criterion(model(L), ab)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ukupno += loss.item()
    tr = ukupno / len(train_loader)
    val = validacija()
    train_gubici.append(tr); val_gubici.append(val)
    print(f"Epoha {epoch:2d}/{EPOCHS}  |  trening: {tr:.4f}  |  validacija: {val:.4f}")

    sacuvaj_primere(epoch)
    if val < najbolji_val:
        najbolji_val = val
        torch.save(model.state_dict(), "checkpoints/najbolji_regresija.pth")

# ---- kriva gubitka ----
plt.figure(figsize=(7, 4))
plt.plot(train_gubici, label="trening")
plt.plot(val_gubici, label="validacija")
plt.xlabel("Epoha"); plt.ylabel("Gubitak (L1)"); plt.title("Kriva gubitka")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("rezultati/kriva_gubitka.png", dpi=100)
plt.close()

torch.save(model.state_dict(), "model_regresija.pth")
print(f"Gotovo. Najbolji validacioni gubitak: {najbolji_val:.4f}")