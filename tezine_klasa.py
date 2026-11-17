import numpy as np
import torch
from torch.utils.data import random_split

from kvantizacija import BROJ_KORPI
from dataset_klasifikacija import KlasifikacijaDataset

LAMBDA = 0.5
MAX_SLIKA = 1500        # dovoljno slika za procenu raspodele boja

full = KlasifikacijaDataset("val2017", image_size=256)
n_val = int(len(full) * 0.1); n_train = len(full) - n_val
train_set, _ = random_split(full, [n_train, n_val],
                            generator=torch.Generator().manual_seed(42))

# ---- prebroj koliko puta se svaka korpa javlja ----
brojac = np.zeros(BROJ_KORPI, dtype=np.int64)
n = min(MAX_SLIKA, len(train_set))
print(f"Racunam raspodelu boja na {n} slika (potraje minut-dva)...")
for i in range(n):
    _, korpa = train_set[i]
    brojac += np.bincount(korpa.numpy().ravel(), minlength=BROJ_KORPI)
    if (i + 1) % 300 == 0:
        print(f"  {i+1}/{n}")

# ---- Zhang-ove tezine ----
Q = BROJ_KORPI
p = brojac / brojac.sum()                       # empirijska raspodela boja
w = 1.0 / ((1 - LAMBDA) * p + LAMBDA / Q)        # retko -> velika tezina
w = w / (p * w).sum()                            # normalizuj: prosecna tezina = 1
w = np.clip(w, 0.0, 5.0)                          # ogranici da ekstremno retke boje ne poremete trening

np.save("tezine_klasa.npy", w.astype(np.float32))
print("Sacuvano: tezine_klasa.npy")
print(f"Vidjenih korpi: {(brojac > 0).sum()} / {Q}")
print(f"Tezine: min {w.min():.3f}, max {w.max():.3f}")
print(f"Najcesca (siva) korpa -> tezina {w[brojac.argmax()]:.3f}")