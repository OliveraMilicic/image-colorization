import os
from PIL import Image
import matplotlib.pyplot as plt

# Sastavlja poredjenje sa 3 kolone (Original / Regresija / Klasifikacija)
# za arhivske fotografije, koristeci vec izracunate slike iz
# porodicne_slike_test.py (bez ponovnog pokretanja mreza).

ULAZ = "porodicne_crnobele_slike"
REZ = "rezultati_porodicne"
IZLAZ = "rezultati_porodicne"

exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")
putanje = sorted(
    [f for f in os.listdir(ULAZ) if f.lower().endswith(exts)],
    key=lambda s: os.path.splitext(s)[0],
)

for f in putanje:
    ime = os.path.splitext(f)[0]
    p_reg = os.path.join(REZ, f"{ime}_regresija.png")
    p_kla = os.path.join(REZ, f"{ime}_klasifikacija.png")
    if not (os.path.exists(p_reg) and os.path.exists(p_kla)):
        print(f"  preskoceno (nedostaju rezultati): {ime}")
        continue

    original = Image.open(os.path.join(ULAZ, f)).convert("L")
    reg = Image.open(p_reg)
    kla = Image.open(p_kla)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.6))
    ax[0].imshow(original, cmap="gray"); ax[0].set_title("Оригинал (црно-бела)")
    ax[1].imshow(reg);                    ax[1].set_title("Регресија")
    ax[2].imshow(kla);                    ax[2].set_title("Класификација")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(IZLAZ, f"{ime}_poredjenje_3col.png"), dpi=110)
    plt.close()
    print("  gotovo:", ime)

print(f"Gotovo. Rezultati u folderu '{IZLAZ}/' (sufiks _poredjenje_3col.png)")
