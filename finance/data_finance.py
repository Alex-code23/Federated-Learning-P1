import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os

# Définition des "Profils" de clients via des tickers spécifiques
# Chaque liste représente un worker avec une "histoire" différente
sectors = {
    "Worker_Tech_Volatile": ['NVDA', 'AMD', 'TSLA', 'BTC-USD'],
    "Worker_Stable_Consumer": ['KO', 'PG', 'JNJ', 'WMT'], # Coca, Procter & Gamble...
    "Worker_Banking_Cyclical": ['JPM', 'BAC', 'GS', 'C'],
    "Worker_Energy": ['XOM', 'CVX', 'SHEL', 'TTE'], # Exxon, Chevron, Shell, Total
    "Worker_Industrial": ['CAT', 'BA', 'GE', 'HON'], # Caterpillar, Boeing, General Electric
    "Worker_Healthcare": ['PFE', 'MRK', 'UNH', 'ABBV'], # Pfizer, Merck, UnitedHealth
    "Worker_Retail": ['AMZN', 'HD', 'TGT', 'COST'] # Amazon, Home Depot, Target, Costco
}

def get_client_data(tickers, start='2020-01-01', end='2023-01-01'):
    # Téléchargement
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)['Close']
    
    # On calcule les rendements (Log-returns sont préférables pour les stats)
    # Log return assure une meilleure symétrie théorique
    returns = np.log(data / data.shift(1)).dropna()
    
    # On "aplatit" tout en une seule série pour avoir la distribution globale du client
    # (Comme si le client avait un portefeuille de ces actifs)
    flat_series = returns.values.flatten()
    
    # Nettoyage des infinis/na
    flat_series = flat_series[~np.isnan(flat_series)]
    flat_series = flat_series[~np.isinf(flat_series)]
    
    return flat_series

def estimate_entropy(data, bins=100):
    """
    Estime l'entropie de Shannon d'une série de données continues en la discrétisant.
    H(X) = - sum(p(x) * log2(p(x)))
    """
    # 1. Créer un histogramme pour obtenir les fréquences des bins
    counts, _ = np.histogram(data, bins=bins)
    
    # 2. Convertir les fréquences en probabilités
    probabilities = counts / counts.sum()
    
    # 3. Filtrer les probabilités nulles (car log(0) est indéfini)
    probabilities = probabilities[probabilities > 0]
    
    # 4. Calculer l'entropie
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy

# --- Paramètres ---
BINS = 100

# Création des datasets des workers
clients_data = {}
entropy_results_for_json = {}

for name, tickers in sectors.items():
    print(f"Construction du {name}...")
    data = get_client_data(tickers)
    entropy = estimate_entropy(data, bins=BINS)
    
    # L'entropie est normalisée pour être entre 0 et 1, comme dans les autres scripts
    # L'entropie maximale pour K bins est log2(K)
    normalized_entropy = entropy / np.log2(BINS) if BINS > 1 else 0
    
    clients_data[name] = {"data": data, "entropy": entropy, "normalized_entropy": normalized_entropy}
    entropy_results_for_json[name] = {
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "bins": BINS
    }
    print(f"-> Entropie: {entropy:.4f} bits | Entropie Normalisée: {normalized_entropy:.4f}")

# --- Sauvegarde des résultats pour entropy_vs_beta.py ---
# Construire un chemin absolu vers le dossier 'entropy' à la racine du projet
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
output_dir = os.path.join(project_root, "entropy")
os.makedirs(output_dir, exist_ok=True)
json_path = os.path.join(output_dir, "finance_entropy_analysis.json")
with open(json_path, 'w') as f:
    json.dump(entropy_results_for_json, f, indent=4)
print(f"\nRésultats d'entropie sauvegardés dans : {json_path}")

plt.figure(figsize=(14, 8))

# On trace la densité de probabilité (KDE) pour chaque worker
for client_name, values in clients_data.items():
    label = f"{client_name} (Entropie Norm: {values['normalized_entropy']:.2f})"
    sns.kdeplot(values['data'], label=label, fill=True, alpha=0.3, common_norm=False)

plt.title("Visualisation du Non-IID : Densité des rendements et Entropie par client")
plt.xlabel("Log-Rendements Journaliers")
plt.xlim(-0.10, 0.10) # Zoom pour bien voir les différences centrales
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()