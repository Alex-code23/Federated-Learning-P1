# Retry with fixed plotting (use pivot from results_df)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy as sc_entropy
from sklearn.datasets import load_digits, load_iris, load_breast_cancer

# assume results_df exists in this cell's scope from previous computation;
# if not, recompute quickly by running the experiment functions again.
# For safety, recompute results_df here (functions are same as above).

from collections import defaultdict
def partition_iid(y, n_clients, rng):
    n = len(y)
    indices = np.arange(n)
    rng.shuffle(indices)
    sizes = [n // n_clients] * n_clients
    for i in range(n % n_clients):
        sizes[i] += 1
    parts = []
    start = 0
    for s in sizes:
        parts.append(indices[start:start+s])
        start += s
    return parts

def partition_dirichlet(y, n_clients, alpha, rng):
    labels = np.unique(y)
    label_indices = {lab: np.where(y == lab)[0] for lab in labels}
    client_indices = [[] for _ in range(n_clients)]
    for lab in labels:
        inds = label_indices[lab].copy()
        rng.shuffle(inds)
        props = rng.dirichlet(alpha=np.ones(n_clients) * alpha)
        # S'assurer qu'au moins 1 échantillon est attribué si possible, pour éviter les zéros dus à l'arrondi
        counts = (props * len(inds)).astype(int)
        diff = len(inds) - counts.sum()
        # Distribuer les échantillons restants (dus à la troncature)
        for i in range(int(diff)):
            counts[i % n_clients] += 1
        start = 0
        for c in range(n_clients):
            if counts[c] > 0:
                client_indices[c].extend(inds[start:start+counts[c]])
                start += counts[c]
    parts = [np.array(sorted(list(set(ci)))) for ci in client_indices]
    return parts

def partition_one_class_per_client(y, n_clients, rng):
    labels = np.unique(y)
    label_indices = {lab: np.where(y == lab)[0] for lab in labels}
    client_indices = [[] for _ in range(n_clients)]
    class_assignments = []
    for i in range(n_clients):
        class_assignments.append(labels[i % len(labels)])
    rng.shuffle(class_assignments)
    per_class_clients = defaultdict(list)
    for c_idx, lab in enumerate(class_assignments):
        per_class_clients[lab].append(c_idx)
    for lab, inds in label_indices.items():
        clients_for_lab = per_class_clients[lab]
        if len(clients_for_lab) == 0:
            continue
        k = len(clients_for_lab)
        sizes = [len(inds)//k] * k
        for i in range(len(inds) % k):
            sizes[i] += 1
        start = 0
        for i, c in enumerate(clients_for_lab):
            if sizes[i] > 0:
                client_indices[c].extend(inds[start:start+sizes[i]])
                start += sizes[i]
    parts = [np.array(sorted(list(set(ci)))) for ci in client_indices]
    return parts

def client_class_stats(parts, y, K):
    """Calcule les statistiques sur les classes pour chaque client."""
    unique_counts = []
    entropies = []
    for p in parts: # p est la liste des indices pour un client
        if len(p) == 0:
            unique_counts.append(0)
            entropies.append(0)
        else:
            labels_client = y[p]
            unique_counts.append(len(np.unique(labels_client)))
            # Calcul de l'entropie
            counts = np.bincount(labels_client, minlength=K)
            dist = counts / counts.sum()
            entropies.append(sc_entropy(dist, base=K)) # Normalisée par log(K)

    unique_counts = np.array(unique_counts)
    entropies = np.array(entropies)

    frac_one_class = np.mean(unique_counts == 1)
    mean_unique = unique_counts.mean()
    mean_entropy = np.mean(entropies)

    return frac_one_class, mean_unique, mean_entropy

def run_experiment(y, dataset_name, n_clients_list, alphas_to_test, n_trials=20, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    records = []
    K = len(np.unique(y))

    for n_clients in n_clients_list:
        for trial in range(n_trials):
            # IID
            parts_iid = partition_iid(y, n_clients, rng)
            frac_one, mean_unique, entropy = client_class_stats(parts_iid, y, K)
            records.append({
                "dataset": dataset_name, "n_clients": n_clients, "trial": trial,
                "partition_method": "IID", "frac_one_class": frac_one,
                "mean_unique_classes": mean_unique, "mean_entropy": entropy
            })

            # Dirichlet for various alphas
            for alpha in alphas_to_test:
                parts_dir = partition_dirichlet(y, n_clients, alpha=alpha, rng=rng)
                frac_one, mean_unique, entropy = client_class_stats(parts_dir, y, K)
                records.append({
                    "dataset": dataset_name, "n_clients": n_clients, "trial": trial,
                    "partition_method": f"Dirichlet (α={alpha})", "frac_one_class": frac_one,
                    "mean_unique_classes": mean_unique, "mean_entropy": entropy
                })

            # Forced one-class
            parts_forced = partition_one_class_per_client(y, n_clients, rng)
            frac_one, mean_unique, entropy = client_class_stats(parts_forced, y, K)
            records.append({
                "dataset": dataset_name, "n_clients": n_clients, "trial": trial,
                "partition_method": "Forcé (1 classe/client)", "frac_one_class": frac_one,
                "mean_unique_classes": mean_unique, "mean_entropy": entropy
            })
    return pd.DataFrame.from_records(records)

datasets = {"digits": load_digits(), "iris": load_iris(), "breast_cancer": load_breast_cancer()}
n_clients_list = [5, 10, 20, 50, 75, 100]
alphas_to_test = [10.0, 1.0, 0.5, 0.1]  # Différentes valeurs de Dirichlet à tester
results = []
for name, ds in datasets.items():
    print(f"Exécution de l'expérience pour : {name.upper()}")
    y = ds.target
    df_res = run_experiment(y, name, n_clients_list, alphas_to_test, n_trials=30, rng_seed=42)
    results.append(df_res)
results_df = pd.concat(results, ignore_index=True)

# pivot for plotting
pivot = results_df.groupby(['dataset', 'n_clients', 'partition_method']).mean().reset_index()


def plot_frac_one_class(pivot_df, ds_name, index):
    """Génère et sauvegarde un graphique montrant la fraction de clients avec une seule classe."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = pivot_df['partition_method'].unique()
    for method in methods:
        sub = pivot_df[(pivot_df['dataset'] == ds_name) & (pivot_df['partition_method'] == method)].sort_values('n_clients')
        if not sub.empty:
            ax.plot(sub['n_clients'], sub['frac_one_class'], marker='o', linestyle='-', label=method)
    
    ax.set_xlabel('Nombre de Clients', fontsize=12)
    ax.set_ylabel('Fraction de clients mono-classe', fontsize=12)
    ax.set_title(f'Impact de la méthode de partition sur l\'hétérogénéité ({ds_name.capitalize()})', fontsize=14, weight='bold')
    ax.legend(title='Méthode de Partition', fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 1.05) # Assurer que l'axe Y va jusqu'à 1

    fig.tight_layout()
    plt.savefig(f'visualisations/partition_analysis/probability_worker_partition_{index}.png')
    plt.close()

def plot_entropy_evolution(pivot_df, ds_name, index):
    """Génère et sauvegarde un graphique montrant l'évolution de l'entropie moyenne."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = pivot_df['partition_method'].unique()
    for method in methods:
        sub = pivot_df[(pivot_df['dataset'] == ds_name) & (pivot_df['partition_method'] == method)].sort_values('n_clients')
        if not sub.empty:
            ax.plot(sub['n_clients'], sub['mean_entropy'], marker='o', linestyle='-', label=method)

    # Titres et labels
    ax.set_xlabel('Nombre de Clients', fontsize=12)
    ax.set_ylabel('Entropie moyenne normalisée', fontsize=12)
    ax.set_title(f'Évolution de l\'entropie moyenne par client ({ds_name.capitalize()})', fontsize=14, weight='bold')
    ax.legend(title='Méthode de Partition', fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    plt.savefig(f'visualisations/partition_analysis/entropy_evolution_{index}.png')
    plt.close(fig)

def plot_mean_classes_evolution(pivot_df, ds_name, index):
    """Génère un graphique montrant l'évolution du nombre moyen de classes par client."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = pivot_df['partition_method'].unique()
    for method in methods:
        sub = pivot_df[(pivot_df['dataset'] == ds_name) & (pivot_df['partition_method'] == method)].sort_values('n_clients')
        if not sub.empty:
            ax.plot(sub['n_clients'], sub['mean_unique_classes'], marker='o', linestyle='-', label=method)
            
    # Titres et labels
    ax.set_xlabel('Nombre de Clients', fontsize=12)
    ax.set_ylabel('Nombre moyen de classes par client', fontsize=12)
    ax.set_title(f'Nombre moyen de classes vues par client ({ds_name.capitalize()})', fontsize=14, weight='bold')
    ax.legend(title='Méthode de Partition', fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    # L'axe Y max dépend du nombre de classes du dataset, on le laisse en auto

    fig.tight_layout()
    plt.savefig(f'visualisations/partition_analysis/mean_classes_evolution_{index}.png')
    plt.close(fig)

for index, ds in enumerate(datasets.keys()):
    plot_frac_one_class(pivot, ds, index)
    plot_entropy_evolution(pivot, ds, index)
    plot_mean_classes_evolution(pivot, ds, index)

# save outputs (again)
csv_path = "data_results/fl_class_partition_summary.csv"
results_df.to_csv(csv_path, index=False)
print(f"Saved detailed results to {csv_path}")
