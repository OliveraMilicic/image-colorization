import matplotlib.pyplot as plt

# Podaci prepisani iz konzolnog izlaza treninga (train.py + train_klasifikacija.py).

epohe = list(range(1, 13))

reg_trening = [0.0737, 0.0708, 0.0696, 0.0686, 0.0678, 0.0671, 0.0667, 0.0663, 0.0660, 0.0657, 0.0654, 0.0651]
reg_val     = [0.0715, 0.0706, 0.0687, 0.0700, 0.0672, 0.0665, 0.0660, 0.0665, 0.0673, 0.0656, 0.0649, 0.0648]

kla_trening = [4.3231, 4.0319, 3.9666, 3.9249, 3.8967, 3.8703, 3.8545, 3.8355, 3.8215, 3.8044, 3.7965, 3.7855]
kla_val     = [4.0809, 3.9452, 3.9079, 3.8622, 3.8251, 3.8274, 3.7994, 3.7812, 3.7748, 3.7734, 3.7561, 3.7412]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

ax[0].plot(epohe, reg_trening, marker="o", label="тренинг")
ax[0].plot(epohe, reg_val, marker="o", label="валидација")
ax[0].set_title("Регресиони приступ (L1 губитак)")
ax[0].set_xlabel("Епоха")
ax[0].set_ylabel("Губитак (L1)")
ax[0].set_xticks(epohe)
ax[0].grid(True, alpha=0.3)
ax[0].legend()

ax[1].plot(epohe, kla_trening, marker="o", label="тренинг")
ax[1].plot(epohe, kla_val, marker="o", label="валидација")
ax[1].set_title("Класификациони приступ (унакрсна ентропија)")
ax[1].set_xlabel("Епоха")
ax[1].set_ylabel("Губитак (cross-entropy)")
ax[1].set_xticks(epohe)
ax[1].grid(True, alpha=0.3)
ax[1].legend()

fig.suptitle("Ток губитка кроз епохе")
plt.tight_layout()
plt.savefig("rezultati_eval/kriva_gubitka_kombinovano.png", dpi=130)
plt.close()
print("Sacuvano: rezultati_eval/kriva_gubitka_kombinovano.png")
