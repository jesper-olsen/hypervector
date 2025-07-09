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

for hdv in ["binary", "bipolar", "real", "complex"]: 
    plot_heatmap(f"RESULTS/hdv_{hdv}_objects.csv", f"HDV: {hdv} – Object Similarities")
    plot_heatmap(f"RESULTS/hdv_{hdv}_sentences.csv", f"HDV: {hdv} – Sentence Similarities")

