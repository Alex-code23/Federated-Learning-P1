import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Charger AG News via Hugging Face
dataset = load_dataset("ag_news")

# Dictionnaire des classes
label_names = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

train_ds = dataset["train"]  # ~120k exemples :contentReference[oaicite:1]{index=1}

texts = train_ds["text"]
labels = np.array(train_ds["label"])  # labels 0 à 3 :contentReference[oaicite:2]{index=2}

# 2. Créer 4 “workers” (1 classe par worker)
num_workers = 4
worker_texts = []
worker_labels = []

for c in range(num_workers):
    idx = np.where(labels == c)[0]
    worker_texts.append([texts[i] for i in idx])
    worker_labels.append([labels[i] for i in idx])

# 3. TF‑IDF
vectorizers = []
worker_features = []

for w_texts in worker_texts:
    vec = TfidfVectorizer(max_features=2000)  # plus de features si tu veux
    X = vec.fit_transform(w_texts).toarray()
    worker_features.append(X)
    vectorizers.append(vec)

# 4. PCA
pca_results = []
for X in worker_features:
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    pca_results.append(X_pca)

# 5. Visualisation
colors = ["red", "blue", "green", "orange"]
plt.figure(figsize=(8,6))
for i, X_pca in enumerate(pca_results):
    plt.scatter(X_pca[:,0], X_pca[:,1], label=f"Worker {i} (classe {label_names[i]})", alpha=0.5, color=colors[i])
plt.legend()
plt.title("PCA des textes AG News par classe")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("visualisations/AG_News/AG_News_data_visualisation.png")
plt.show()