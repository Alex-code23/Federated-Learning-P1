import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy as sc_entropy
import os

np.random.seed(42)

# --- Paramètres de la simulation ---
K = 10           # Nombre de classes
N_per_class = 400  # Échantillons par classe
D = 20           # Dimension des features (non utilisé pour l'entropie mais bon pour le contexte)
W = 10           # Nombre de travailleurs

# Bétas à explorer : on utilise une échelle logarithmique pour mieux voir l'évolution
betas = np.logspace(-2, 2, num=20)  # De 0.01 à 100

# --- Génération du jeu de données synthétique ---
# On a juste besoin des étiquettes 'y' pour cette analyse
y_list = []
for k in range(K):
    y_list.append(np.full(N_per_class, k, dtype=int))

y = np.concatenate(y_list)
N = y.shape[0]

# --- Fonctions de partitionnement et de métriques (simplifiées de task_1.py) ---

def dirichlet_partition(y, W, alpha):
    """Partitionne les données en utilisant une distribution de Dirichlet."""
    parts = [[] for _ in range(W)]
    classes = np.unique(y)
    
    for c in classes:
        idx = np.where(y == c)[0].copy()
        np.random.shuffle(idx)
        
        # Distribue les indices de cette classe selon les proportions de Dirichlet
        proportions = np.random.dirichlet([alpha] * W)
        
        # Calcule le nombre d'échantillons par travailleur pour cette classe
        counts = (proportions * len(idx)).astype(int)
        
        # Assigne les échantillons restants pour que le total corresponde
        diff = len(idx) - counts.sum()
        for i in range(diff):
            counts[i % W] += 1
            
        start = 0
        for w in range(W):
            count = counts[w]
            if count > 0:
                parts[w].extend(idx[start : start + count].tolist())
                start += count
    return parts

def class_distribution(parts, y, K):
    """Calcule la distribution des classes pour chaque travailleur."""
    dist = np.zeros((len(parts), K), dtype=float)
    for w, idx in enumerate(parts):
        if len(idx) == 0:
            continue
        counts = np.bincount(y[idx], minlength=K)
        dist[w] = counts / counts.sum()
    return dist

def worker_entropy(dist):
    """Calcule l'entropie pour la distribution de chaque travailleur."""
    ent = np.array([sc_entropy(p, base=K) for p in dist]) # Normalisé par log(K)
    # Remplace les NaN (pour les travailleurs sans données) par 0
    ent = np.nan_to_num(ent, nan=0.0)
    return ent

# --- Calcul de l'entropie pour chaque valeur de beta ---

mean_entropies = []
std_entropies = []

for beta in betas:
    # Partitionner les données pour le beta actuel
    parts = dirichlet_partition(y, W, alpha=beta)
    
    # Calculer la distribution des classes
    dist = class_distribution(parts, y, K)
    
    # Calculer l'entropie pour chaque travailleur
    entropies = worker_entropy(dist)
    
    # Stocker la moyenne et l'écart-type des entropies
    mean_entropies.append(np.mean(entropies))
    std_entropies.append(np.std(entropies))

# --- Visualisation ---

plt.figure(figsize=(10, 6))
mean_entropies = np.array(mean_entropies)
std_entropies = np.array(std_entropies)

plt.plot(betas, mean_entropies, marker='o', linestyle='-', label="Entropie moyenne normalisée")
plt.fill_between(betas, mean_entropies - std_entropies, mean_entropies + std_entropies,
                 alpha=0.2, label="Écart-type de l'entropie")

plt.xscale('log')
plt.xlabel("Beta (alpha) de la distribution de Dirichlet (échelle log)")
plt.ylabel("Entropie moyenne normalisée des travailleurs")
plt.title("Évolution de l'hétérogénéité des données (entropie) en fonction de Beta")
plt.grid(True, which="both", ls="--")
plt.legend()

# Enregistrer la figure
output_dir = "visualisations/partition_analysis"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "entropy_vs_beta.png")
plt.savefig(output_path)

print(f"Le graphique a été sauvegardé dans : {output_path}")
plt.show()