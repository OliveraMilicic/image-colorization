import numpy as np

# ---- definicija korpi (mreza preko ab prostora) ----
AB_MIN = -110.0
AB_MAX = 110.0
KORAK = 10.0                              # velicina korpe (u ab jedinicama)
N = int((AB_MAX - AB_MIN) / KORAK)        # broj korpi po osi (22)
BROJ_KORPI = N * N                        # ukupno (484)


def ab_u_korpu(ab):
    """ab: (..., 2) realne vrednosti -> indeks korpe (ceo broj)."""
    ab = np.clip(ab, AB_MIN, AB_MAX - 1e-3)
    idx = ((ab - AB_MIN) / KORAK).astype(np.int64)   # [ia, ib]
    return idx[..., 0] * N + idx[..., 1]             # spljosti u jedan broj


def korpa_u_ab(korpa):
    """indeks korpe -> ab vrednost (centar korpe)."""
    ia = korpa // N
    ib = korpa % N
    a = AB_MIN + (ia + 0.5) * KORAK
    b = AB_MIN + (ib + 0.5) * KORAK
    return np.stack([a, b], axis=-1)


# ---- test: koliko boje gubimo kvantizacijom? ----
if __name__ == "__main__":
    import torch
    from torch.utils.data import random_split
    from PIL import Image
    from skimage import color
    import matplotlib.pyplot as plt

    from dataset import ColorizationDataset

    print(f"Broj korpi: {BROJ_KORPI}  ({N} x {N})")

    # ista validaciona slika (isti seed) kao u ostalim skriptama, umesto astronauta
    DATA_DIR = "train2017"
    IMAGE_SIZE = 256
    full = ColorizationDataset(DATA_DIR, image_size=IMAGE_SIZE)
    full.paths = full.paths[:40000]
    n_val = int(len(full) * 0.1); n_train = len(full) - n_val
    _, val_set = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    slika_put = full.paths[val_set.indices[0]]

    img = Image.open(slika_put).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    rgb = np.asarray(img, dtype=np.float32) / 255.0
    lab = color.rgb2lab(rgb)
    L = lab[:, :, 0]
    ab = lab[:, :, 1:3]

    # ab -> korpa -> ab (centri): ovako ce izgledati kad mreza radi sa korpama
    korpe = ab_u_korpu(ab)
    ab_kvant = korpa_u_ab(korpe)

    lab_kvant = np.concatenate([L[:, :, None], ab_kvant], axis=2)
    rgb_kvant = np.clip(color.lab2rgb(lab_kvant), 0, 1)

    print("Broj razlicitih korpi u ovoj slici:", len(np.unique(korpe)))

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].imshow(rgb);        ax[0].set_title("Оригинал")
    ax[1].imshow(rgb_kvant);  ax[1].set_title(f"Квантизовано ({BROJ_KORPI} корпи)")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig("kvantizacija_test.png", dpi=100)
    print("Sacuvano: kvantizacija_test.png")