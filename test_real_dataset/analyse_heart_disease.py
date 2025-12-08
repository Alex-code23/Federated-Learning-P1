import json
import os
import pandas as pd
import numpy as np
from scipy.stats import entropy as sc_entropy

DATA_DIR = 'test_real_dataset/data_heartdisease/'
COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
]

def load_heart_disease_data(data_dir):
    """
    Charge les données du dataset Heart Disease depuis plusieurs fichiers,
    en considérant chaque fichier comme un worker.
    """
    all_data = []
    worker_indices = []
    worker_names = []
    current_index = 0

    # Lister les fichiers de données (ex: processed.cleveland.data, etc.)
    data_files = [f for f in os.listdir(data_dir) if f.startswith('processed.')]
    
    print(f"Fichiers détectés comme workers: {data_files}")

    for i, filename in enumerate(data_files):
        file_path = os.path.join(data_dir, filename)
        # Charger les données en remplaçant '?' par NaN
        df = pd.read_csv(file_path, header=None, names=COLUMN_NAMES, na_values='?')
        
        # Extraire le nom du worker (ex: 'cleveland')
        worker_names.append(filename.split('.')[1])
        
        # Garder une trace du nombre d'échantillons avant de supprimer les lignes
        n_samples_before = len(df)
        
        # Remplacer les valeurs manquantes (NaN) par 0 au lieu de supprimer les lignes.
        # Cela permet de conserver tous les échantillons, même ceux incomplets.
        df.fillna(0, inplace=True)
        
        n_samples_after = len(df)
        print(f"Worker {i} ({filename}): {n_samples_before} lignes -> {n_samples_after} lignes après suppression des NaN.")

        if n_samples_after == 0:
            print(f"ATTENTION : Le worker '{filename}' n'a plus aucun échantillon après nettoyage.")

        # Stocker les indices pour ce worker
        indices = list(range(current_index, current_index + n_samples_after))
        worker_indices.append(np.array(indices))
        current_index += n_samples_after

        all_data.append(df)

    # Concaténer toutes les données dans un seul DataFrame
    if not all_data:
        raise FileNotFoundError(f"Aucun fichier de données trouvé dans {data_dir}")
        
    full_df = pd.concat(all_data, ignore_index=True)

    # La cible 'num' correspond à la présence de maladie cardiaque. 0 = absence, 1-4 = présence.
    # Pour notre analyse, on binarise la cible : 0 pour "sain" et 1 pour "malade" (>0).
    y = (full_df['num'] > 0).astype(int)

    return y.values, worker_indices, worker_names

def analyze_partition_entropy(y, parts, worker_names):
    """
    Analyse l'entropie et d'autres métriques pour une partition de données donnée.
    """
    K = len(np.unique(y)) # Nombre de classes uniques (ici, 2)
    n_clients = len(parts)
    analysis_data = {}
    
    client_entropies = []
    client_unique_classes = []
    client_sample_counts = []

    for p in parts:
        client_sample_counts.append(len(p))
        if not p.any():
            client_entropies.append(0)
            client_unique_classes.append(0)
            continue

        labels_client = y[p]
        counts = np.bincount(labels_client, minlength=K)
        dist = counts / counts.sum()
        
        # Entropie normalisée (entre 0 et 1)
        client_entropies.append(sc_entropy(dist, base=K))
        client_unique_classes.append(len(np.unique(labels_client)))

    for i, worker_name in enumerate(worker_names):
        labels_client = y[parts[i]]
        counts = np.bincount(labels_client, minlength=K)
        analysis_data[worker_name] = {
            "counts": {k: int(v) for k, v in enumerate(counts)},
            "entropy": client_entropies[i]
        }

    print("\n--- Analyse de l'hétérogénéité du dataset Heart Disease ---")
    print(f"Nombre total de workers (locations): {n_clients}")
    print(f"Nombre total d'échantillons (après nettoyage): {len(y)}")
    print(f"Nombre de classes: {K}")
    
    print("\n--- Statistiques sur les échantillons par worker ---")
    print(f"Nombre moyen d'échantillons par worker: {np.mean(client_sample_counts):.2f}")
    print(f"Écart-type des échantillons par worker: {np.std(client_sample_counts):.2f}")
    print(f"Min/Max échantillons par worker: {np.min(client_sample_counts)} / {np.max(client_sample_counts)}")

    print("\n--- Statistiques sur la distribution des classes ---")
    print(f"Nombre moyen de classes uniques par worker: {np.mean(client_unique_classes):.2f}")
    print(f"Fraction de workers avec 1 seule classe: {np.mean(np.array(client_unique_classes) == 1):.2%}")
    
    print("\n--- Analyse de l'entropie ---")
    print(f"Entropie moyenne (normalisée) sur tous les workers: {np.mean(client_entropies):.4f}")
    print(f"Écart-type de l'entropie: {np.std(client_entropies):.4f}")
    print(f"Entropie Min/Max: {np.min(client_entropies):.4f} / {np.max(client_entropies):.4f}")

    # Sauvegarder les résultats dans le dossier entropy à la racine
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    entropy_dir = os.path.join(project_root, 'entropy')
    os.makedirs(entropy_dir, exist_ok=True)
    output_path = os.path.join(entropy_dir, 'heart_disease_entropy_analysis.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=4)
    
    print(f"\n💾 Données d'entropie sauvegardées dans : {output_path}")

if __name__ == "__main__":
    y_full, partitions, worker_names = load_heart_disease_data(DATA_DIR)
    analyze_partition_entropy(y_full, partitions, worker_names)