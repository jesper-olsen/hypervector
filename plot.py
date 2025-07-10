import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(file, title):
    df = pd.read_csv(file, index_col=0)
    plt.figure(figsize=(8, 6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_side_by_side(hdv):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns
    obj_df = pd.read_csv(f"RESULTS/hdv_{hdv}_objects.csv", index_col=0)
    sent_df = pd.read_csv(f"RESULTS/hdv_{hdv}_sentences.csv", index_col=0)
    sns.heatmap(obj_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[0])
    axes[0].set_title(f"{hdv.capitalize()} HDV – Object Similarities")
    sns.heatmap(sent_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[1])
    axes[1].set_title(f"{hdv.capitalize()} HDV – Sentence Similarities")
    plt.tight_layout()
    plt.savefig(f"ASSETS/{hdv}_hdv_combined.png")  # Save as PNG for README
    plt.close()

for hdv in ["binary", "bipolar", "real", "complex"]: 
    #plot_heatmap(f"RESULTS/hdv_{hdv}_objects.csv", f"HDV: {hdv} – Object Similarities")
    #plot_heatmap(f"RESULTS/hdv_{hdv}_sentences.csv", f"HDV: {hdv} – Sentence Similarities")
    plot_side_by_side(hdv)

