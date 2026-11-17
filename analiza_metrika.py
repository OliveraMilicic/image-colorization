import pandas as pd
import matplotlib.pyplot as plt

# ---- ucitaj rezultate po slici (izracunato u evaluacija.py) ----
df = pd.read_csv("rezultati_eval/metrike.csv")

METODE = ["siva", "regresija", "klasifikacija"]
NAZIVI = {"siva": "Неутрална\n(сива) основа", "regresija": "Регресиони\nприступ", "klasifikacija": "Класификациони\nприступ"}
BOJE = {"siva": "#9e9e9e", "regresija": "#4c72b0", "klasifikacija": "#dd8452"}

# ---- 1) box plotovi: raspodela PSNR / SSIM / zasicenosti po metodi ----
mere = [("psnr", "PSNR (dB)"), ("ssim", "SSIM"), ("zasicenost", "Засићеност")]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
for ax, (mera, naslov) in zip(axes, mere):
    podaci = [df[f"{m}_{mera}"] for m in METODE]
    bp = ax.boxplot(podaci, tick_labels=[NAZIVI[m] for m in METODE], patch_artist=True, showfliers=False)
    for patch, m in zip(bp["boxes"], METODE):
        patch.set_facecolor(BOJE[m])
        patch.set_alpha(0.7)
    ax.set_title(naslov)
    ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("rezultati_eval/distribucije_metrika.png", dpi=130)
plt.close()
print("Sacuvano: rezultati_eval/distribucije_metrika.png")

# ---- 2) rasejanje: zasicenost naspram PSNR, po slici, obojeno po metodi ----
# (siva osnova ima ekstremne outlier-e >100 dB kod skoro monohromatskih
#  originala - y-osa je ogranicena radi citljivosti glavnog obrasca)
Y_MAX = 45
broj_odsecenih = int((df[[f"{m}_psnr" for m in METODE]] > Y_MAX).sum().sum())

plt.figure(figsize=(6.5, 5.5))
for m in METODE:
    plt.scatter(df[f"{m}_zasicenost"], df[f"{m}_psnr"], s=14, alpha=0.5,
                color=BOJE[m], label=NAZIVI[m].replace("\n", " "))
plt.xlabel("Засићеност")
plt.ylabel("PSNR (dB)")
plt.ylim(top=Y_MAX)
plt.title("Однос засићености и PSNR по слици")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("rezultati_eval/zasicenost_vs_psnr.png", dpi=130)
plt.close()
print(f"Sacuvano: rezultati_eval/zasicenost_vs_psnr.png ({broj_odsecenih} tacaka > {Y_MAX} dB van prikaza, sve iz 'siva')")

# ---- 3) sazeta tabela: srednja vrednost i std ----
red = []
for m in METODE:
    red.append({
        "metoda": m,
        "psnr_mean": df[f"{m}_psnr"].mean(), "psnr_std": df[f"{m}_psnr"].std(),
        "ssim_mean": df[f"{m}_ssim"].mean(), "ssim_std": df[f"{m}_ssim"].std(),
        "zasicenost_mean": df[f"{m}_zasicenost"].mean(), "zasicenost_std": df[f"{m}_zasicenost"].std(),
    })
tabela = pd.DataFrame(red)
tabela.to_csv("rezultati_eval/metrike_sazetak.csv", index=False, float_format="%.4f")
print("Sacuvano: rezultati_eval/metrike_sazetak.csv")
print(tabela.to_string(index=False))
