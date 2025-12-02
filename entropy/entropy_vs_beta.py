import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy as sc_entropy
import os
import json

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

# --- Chargement des données d'entropie réelles pour définir la zone "réaliste" ---

def load_real_world_entropies():
    """Charge les entropies depuis les fichiers JSON et les normalise."""
    real_entropies = []
    script_dir = os.path.dirname(__file__)
    
    # Fichier 1: Heart Disease
    hd_path = os.path.join(script_dir, 'heart_disease_entropy_analysis.json')
    if os.path.exists(hd_path):
        with open(hd_path, 'r') as f:
            hd_data = json.load(f)
        for worker_data in hd_data.values():
            real_entropies.append(worker_data['entropy'])

    # Fichier 2: Web Crawler
    web_path = os.path.join(script_dir, 'entropy_analysis.json')
    if os.path.exists(web_path):
        with open(web_path, 'r') as f:
            web_data = json.load(f)
        for site_data in web_data.values():
            raw_entropy = site_data['entropy']
            num_classes = len(site_data['counts'])
            if num_classes > 1:
                # Normaliser l'entropie (calculée en base 2) par log2(K)
                normalized_entropy = raw_entropy / np.log2(num_classes)
                real_entropies.append(normalized_entropy)
            elif raw_entropy > 0: # Cas étrange, mais pour être sûr
                real_entropies.append(1.0) # Non normalisable, mais clairement hétérogène

    return real_entropies

real_world_entropies = load_real_world_entropies()

# --- Visualisation ---

plt.figure(figsize=(10, 6))
mean_entropies = np.array(mean_entropies)
std_entropies = np.array(std_entropies)

plt.plot(betas, mean_entropies, marker='o', linestyle='-', color='blue', label="Entropie moyenne normalisée")
plt.fill_between(betas, mean_entropies - std_entropies, mean_entropies + std_entropies,
                 color='blue', alpha=0.2, label="Écart-type de l'entropie")

plt.xscale('log')
plt.xlabel("Beta (alpha) de la distribution de Dirichlet (échelle log)")
plt.ylabel("Entropie moyenne normalisée des travailleurs")
plt.title("Évolution de l'hétérogénéité des données (entropie) en fonction de Beta")
plt.grid(True, which="both", ls="--")

# --- Ajout de zones d'interprétation basées sur les données réelles ---
if real_world_entropies:
    # Définir la zone réaliste comme moyenne ± écart-type des entropies observées
    mean_real_entropy = np.mean(real_world_entropies)
    std_real_entropy = np.std(real_world_entropies)
    
    zone_min = max(0, mean_real_entropy - std_real_entropy)
    zone_max = min(1, mean_real_entropy + std_real_entropy)
    
    # Zone 1: Très hétérogène (plus que nos données réelles)
    plt.axhspan(0, zone_min, color='red', alpha=0.1, label='Très hétérogène (Pathologique)')
    plt.text(0.015, zone_min / 2, 'Scénario "Pathologique"\n(1-2 classes/client)', fontsize=9, color='darkred', verticalalignment='center')
    
    # Zone 2: Scénario réaliste (basé sur nos datasets)
    plt.axhspan(zone_min, zone_max, color='orange', alpha=0.15, label=f'Scénario Réaliste (Moyenne ± Écart-type)')
    plt.text(0.1, (zone_min + zone_max) / 2, f'Scénario "Réaliste"\n(Moyenne ± Écart-type: [{zone_min:.2f}, {zone_max:.2f}])', fontsize=9, color='darkorange', verticalalignment='center')
    
    # Zone 3: Quasi-IID (plus homogène que nos données réelles)
    plt.axhspan(zone_max, 1.0, color='green', alpha=0.1, label='Quasi-IID (Idéaliste)')
    plt.text(10, (zone_max + 1.0) / 2, 'Scénario "Idéaliste"\n(Proche de IID)', fontsize=9, color='darkgreen', verticalalignment='center')
else:
    print("Avertissement: Fichiers d'entropie non trouvés. Les zones du graphique sont illustratives.")

plt.legend()

# Enregistrer la figure
output_dir = "visualisations/partition_analysis"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "entropy_vs_beta.png")
plt.savefig(output_path)

print(f"Le graphique a été sauvegardé dans : {output_path}")
plt.show()