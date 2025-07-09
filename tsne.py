import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

hdvs = np.loadtxt("RESULTS/model.csv", delimiter=",", dtype=np.float32)

labels=["af", "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hu", "it", "lt", "lv", "nl", "pl", "pt", "ro", "sk", "sl", "sv"]

assert hdvs.shape[0] == len(labels), "Mismatch between HDV rows and labels"

tsne = TSNE(n_components=2, random_state=42, perplexity=5)
coords = tsne.fit_transform(hdvs)

plt.figure(figsize=(10, 8))
for i, label in enumerate(labels):
    plt.scatter(*coords[i])
    plt.text(*coords[i], label, fontsize=12)
plt.title("Language space via HDV similarity")
plt.show()

