import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("kv_results.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
bit_groups = [8192, 65536, 131072]
titles = ["8 192 bits", "65 536 bits", "131 072 bits"]

for ax, bits, title in zip(axes, bit_groups, titles):
    group = df[df["bits"] == bits]
    for label, sub in group.groupby("type"):
        ax.plot(sub["n"], sub["accuracy_mean"], marker="o", label=label)
        ax.fill_between(
            sub["n"],
            sub["accuracy_mean"] - sub["accuracy_std"],
            sub["accuracy_mean"] + sub["accuracy_std"],
            alpha=0.15,
        )
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("Bundle size N")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("HDC Associative Memory Capacity")
plt.tight_layout()
plt.savefig("capacity.png", dpi=150)
plt.show()
