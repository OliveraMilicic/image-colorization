import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(20, 4.2))
ax.set_xlim(0, 28)
ax.set_ylim(0, 6)
ax.axis("off")


def kutija(xy, w, h, tekst, boja="#e8eef7", ivica="#2c3e50", fs=10.5):
    x, y = xy
    box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                          boxstyle="round,pad=0.02,rounding_size=0.12",
                          linewidth=1.6, edgecolor=ivica, facecolor=boja)
    ax.add_patch(box)
    ax.text(x, y, tekst, ha="center", va="center", fontsize=fs, linespacing=1.35)
    return x, y


def strelica(p1, p2, zakrivljenost=0.0, lw=1.6):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=lw,
                                 connectionstyle=f"arc3,rad={zakrivljenost}"))


Y = 3.6
W, H = 3.2, 1.5
razmak = 4.0

boje = ["#f4f4f4", "#f4f4f4", "#dbe7f7", "#cfe0f3", "#eaeaea", "#dbe7f7", "#d7e9d7"]
tekstovi = [
    "Улазна\nфотографија (RGB)",
    "Претварање\nRGB → Lab",
    "L канал\n(светлина)\nулаз мреже",
    "U-Net\n(енкодер–декодер,\nскип везе)",
    "Предвиђене\na, b компоненте",
    "Спајање L + (a,b)\n→ Lab → RGB",
    "Обојена\nизлазна слика",
]

centri = []
for i, (t, b) in enumerate(zip(tekstovi, boje)):
    x = 2 + i * razmak
    c = kutija((x, Y), W, H, t, boja=b)
    centri.append(c)

for i in range(len(centri) - 1):
    strelica((centri[i][0] + W / 2, Y), (centri[i + 1][0] - W / 2, Y))

# L kanal se ponovo koristi pri spajanju sa predvidjenim (a,b) -> petlja ispod reda,
# vodjena pod pravim uglom da ne prelazi preko kutija izmedju
c_l = centri[2]
c_spoj = centri[5]
y_dole = Y - H / 2 - 1.5
x_dole = c_spoj[0] - 0.7

ax.plot([c_l[0], c_l[0]], [Y - H / 2, y_dole], color="#2c3e50", lw=1.6)
ax.plot([c_l[0], x_dole], [y_dole, y_dole], color="#2c3e50", lw=1.6)
ax.annotate("", xy=(x_dole, Y - H / 2 - 0.05), xytext=(x_dole, y_dole),
            arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=1.6))
ax.text((c_l[0] + x_dole) / 2, y_dole - 0.45, "L канал се чува и поново користи при спајању",
        ha="center", va="center", fontsize=9, color="#555555", style="italic")

plt.tight_layout()
plt.savefig("pipeline_dijagram_v2.png", dpi=150)
plt.close()
print("Sacuvano: pipeline_dijagram_v2.png")
