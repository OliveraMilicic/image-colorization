import numpy as np
import torch
from torch.utils.data import random_split
from PIL import Image
import matplotlib.pyplot as plt
from skimage import color, util

from dataset import ColorizationDataset

# 1. Ucitaj istu validacionu sliku (isti seed) kao u ostalim skriptama
DATA_DIR = "train2017"
IMAGE_SIZE = 256
full = ColorizationDataset(DATA_DIR, image_size=IMAGE_SIZE)
full.paths = full.paths[:40000]
n_val = int(len(full) * 0.1); n_train = len(full) - n_val
_, val_set = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(42))
slika_put = full.paths[val_set.indices[0]]

img = Image.open(slika_put).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
rgb = util.img_as_float(np.asarray(img))   # oblik (visina, sirina, 3), vrednosti 0-1
print("RGB oblik:", rgb.shape)

# 2. Konvertuj RGB -> Lab prostor boja
lab = color.rgb2lab(rgb)        # 3 kanala: L, a, b
L = lab[:, :, 0]                # svetlina   -> ULAZ u mrezu (crno-belo)
a = lab[:, :, 1]                # zeleno <-> crveno
b = lab[:, :, 2]                # plavo  <-> zuto   (a i b zajedno = boja -> IZLAZ mreze)

# 3. Ispisi opsege vrednosti (da osetis kako Lab radi)
print(f"L: {L.min():6.1f} do {L.max():6.1f}   (svetlina, ~0-100)")
print(f"a: {a.min():6.1f} do {a.max():6.1f}   (zeleno-crveno)")
print(f"b: {b.min():6.1f} do {b.max():6.1f}   (plavo-zuto)")

# 4. Prikazi original i tri kanala jedan pored drugog
fig, ax = plt.subplots(1, 4, figsize=(14, 4))
ax[0].imshow(rgb);             ax[0].set_title("Оригинал (RGB)")
ax[1].imshow(L, cmap="gray");  ax[1].set_title("L — улаз у мрежу")
ax[2].imshow(a, cmap="gray");  ax[2].set_title("a")
ax[3].imshow(b, cmap="gray");  ax[3].set_title("b")
for axi in ax:
    axi.axis("off")
plt.tight_layout()
plt.savefig("lab_test.png", dpi=100)
print("Slika sacuvana kao lab_test.png")