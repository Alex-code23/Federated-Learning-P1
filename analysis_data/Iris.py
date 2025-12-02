from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 1. Charger le dataset Iris
iris = load_iris()
X = iris.data        # features
y = iris.target      # labels
target_names = iris.target_names

# 2. Standardisation des données (important pour la PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. PCA à 2 composantes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 4. Affichage
plt.figure(figsize=(8,6))

for target, name in enumerate(target_names):
    plt.scatter(
        X_pca[y == target, 0],
        X_pca[y == target, 1],
        label=name
    )

plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA of Iris Dataset")
plt.legend()
plt.savefig("visualisations/Iris/IRIS_data_visualisation.png")


plt.figure(figsize=(8,6))
plt.hist(y, bins=[-0.5, 0.5, 1.5, 2.5], width=0.9)
plt.xticks([0, 1, 2], target_names)
plt.xlabel("Espèce")
plt.ylabel("Nombre d'échantillons")
plt.title("Histogramme des espèces dans le dataset Iris")

# Enregistrer l'image
plt.savefig("visualisations/Iris/IRIS_data_histogramme.png")
plt.show()