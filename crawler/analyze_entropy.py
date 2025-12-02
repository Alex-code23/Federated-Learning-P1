import os
import json
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt
def calculate_entropy(labels):
    """Calcule l'entropie de Shannon pour une liste d'étiquettes."""
    if not labels:
        return 0
    
    counts = Counter(labels)
    total_count = len(labels)
    entropy = 0.0
    
    print(f"  Distribution des classes : {dict(counts)}")
    
    for label in counts:
        probability = counts[label] / total_count
        if probability > 0:
            entropy -= probability * np.log2(probability)
            
    return entropy

def plot_single_site_distribution(site_name, counts, save_path):
    """Crée un graphique à barres pour la distribution des classes d'un seul site."""
    labels = list(counts.keys())
    values = list(counts.values())

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color='skyblue')
    plt.ylabel('Nombre de pages')
    plt.title(f'Distribution des classes pour le site : {site_name}')
    plt.xticks(rotation=15, ha="right")
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_all_sites_distribution(sites_data, save_path):
    """Crée un graphique à barres groupées pour comparer la distribution de tous les sites."""
    all_labels = sorted(list(set(label for data in sites_data.values() for label in data['counts'].keys())))
    site_names = list(sites_data.keys())
    
    x = np.arange(len(site_names))
    width = 0.8 / len(all_labels)
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    for i, label in enumerate(all_labels):
        counts = [sites_data[site].get('counts', {}).get(label, 0) for site in site_names]
        offset = width * i - (width * (len(all_labels) -1) / 2)
        rects = ax.bar(x + offset, counts, width, label=label)
        ax.bar_label(rects, padding=3, fontsize=8)

    ax.set_ylabel('Nombre de pages')
    ax.set_title('Comparaison de la distribution des classes par site')
    ax.set_xticks(x)
    ax.set_xticklabels(site_names, rotation=45, ha="right")
    ax.legend(title="Classes")

    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # Le chemin est relatif à l'emplacement du script
    SCRIPT_DIR = os.path.dirname(__file__)
    PROCESSED_DATA_DIR = os.path.join(SCRIPT_DIR, "processed_data")
    PLOTS_DIR = os.path.join(SCRIPT_DIR, "analysis_plots")
    # Le dossier entropy est à la racine du projet, donc on remonte d'un niveau
    ENTROPY_DATA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "entropy")
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(ENTROPY_DATA_DIR, exist_ok=True)

    print("\n--- Analyse de l'entropie des jeux de données ---")

    if not os.path.exists(PROCESSED_DATA_DIR):
        print(f"Le dossier '{PROCESSED_DATA_DIR}' n'existe pas. Exécutez d'abord process_crawled_data.py.")
    else:
        all_sites_data = {}
        for filename in os.listdir(PROCESSED_DATA_DIR):
            if filename.endswith(".json"):
                site_name = filename.replace(".json", "")
                filepath = os.path.join(PROCESSED_DATA_DIR, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                labels = data.get('y', [])
                entropy = calculate_entropy(labels)
                print(f"Entropie pour '{site_name}': {entropy:.4f}\n")
                
                all_sites_data[site_name] = {
                    'counts': Counter(labels),
                    'entropy': entropy
                }

        # Générer et sauvegarder les plots
        if all_sites_data:
            # Sauvegarder les données d'entropie dans un fichier JSON
            entropy_output_path = os.path.join(ENTROPY_DATA_DIR, "entropy_analysis.json")
            with open(entropy_output_path, 'w', encoding='utf-8') as f:
                json.dump(all_sites_data, f, indent=4)
            print(f"💾 Données d'entropie sauvegardées dans : {entropy_output_path}\n")

            plot_all_sites_path = os.path.join(PLOTS_DIR, "all_sites_distribution.png")
            plot_all_sites_distribution(all_sites_data, plot_all_sites_path)
            print(f"📊 Graphique comparatif sauvegardé dans : {plot_all_sites_path}")

            for site_name, data in all_sites_data.items():
                plot_site_path = os.path.join(PLOTS_DIR, f"distrib_{site_name}.png")
                plot_single_site_distribution(site_name, data['counts'], plot_site_path)
            print(f"📈 Graphiques individuels sauvegardés dans le dossier : {PLOTS_DIR}")