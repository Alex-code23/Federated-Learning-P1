import math
import os
import random
from datetime import datetime


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer

from tools.models import SoftmaxModel
from simu import run_simulation
from tools.plot import plot_class_accuracy_evolution, plot_mean_ci_partitions, plot_xi_A_partitions_mean_ci, plot_partitions_aggregators, plot_xi_A_partitions

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n\n --------> Using device: {DEVICE} --------------\n\n')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if __name__ == '__main__':
    # --- Configuration de la simulation ---
    W = 4
    R = 3
    T = 400
    N = 1  # Nombre de simulations pour la moyenne

    partition_list = ['dirichlet', 'noniid']
    ATTACK = 'static'

    # --- Data Loading and Preprocessing with Hugging Face datasets and Scikit-learn ---
    print("Loading AG_NEWS dataset...")
    dataset = load_dataset("ag_news")

    train_texts = [item['text'] for item in dataset['train']]
    train_labels = [item['label'] for item in dataset['train']]
    test_texts = [item['text'] for item in dataset['test']]
    test_labels = [item['label'] for item in dataset['test']]

    print("Vectorizing text with TfidfVectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(train_labels, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(test_labels, dtype=torch.long)

    # Create TensorDatasets (compatible with existing simulation code)
    trainset = TensorDataset(X_train_tensor, y_train_tensor)
    testset = TensorDataset(X_test_tensor, y_test_tensor)

    # --- Modèle ---
    num_features = X_train.shape[1]
    num_class = len(set(train_labels))
    # Le modèle sera instancié dans run_simulation, ici on définit juste le type
    MODEL_TYPE = 'softmax' # Un modèle linéaire simple est parfait pour les features TF-IDF

    # --- Simulation ---
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M")
    DATASET_NAME = 'AG_NEWS'

    # Structures pour accumuler les résultats sur N runs
    results_all = {PART: {agg: {'accs': [], 'losses': [], 'xi': [], 'A': [], 'variance': [], 'per_class_accs': []}
                          for agg in ['Mean', 'TriMean', 'FABA', 'CC']} for PART in partition_list}

    for k in range(N):
        print(f"\n\n========== Simulation NLP: {k+1}/{N} ==========")
        for PARTITION in partition_list:
            print(f'\n\n=== Partition: {PARTITION} ===')
            for agg in results_all[PARTITION].keys():
                print('Running aggregator:', agg)
                
                stats = run_simulation(
                    trainset, testset,
                    W=W, R=R,
                    aggregator_name=agg,
                    partition=PARTITION,
                    attack_type=ATTACK,
                    model_type=MODEL_TYPE,
                    num_classes=num_class, # Passer le bon nombre de classes (4)
                    # model_factory et collate_fn ne sont plus nécessaires avec TF-IDF
                    T=T,
                    local_batch=64,
                    gamma=0.1, # Le learning rate peut nécessiter un ajustement pour le NLP
                    alpha=0.1,
                    verbose=True
                )

                # Stocker les résultats
                results_all[PARTITION][agg]['accs'].append(np.array(stats['accs']))
                results_all[PARTITION][agg]['losses'].append(np.array(stats['losses']))
                results_all[PARTITION][agg]['xi'].append(np.array(stats['xi']))
                results_all[PARTITION][agg]['A'].append(np.array(stats['A']))
                results_all[PARTITION][agg]['variance'].append(np.array(stats['variance']))
                results_all[PARTITION][agg]['per_class_accs'].append([np.array(class_acc) for class_acc in stats['per_class_accs']])

    # --- Calcul de la moyenne et de l'intervalle de confiance ---
    def mean_and_ci(list_of_arrays, ci_factor=1.96):
        arr = np.stack(list_of_arrays, axis=0)
        mean = np.mean(arr, axis=0)
        if arr.shape[0] <= 1:
            sem = np.zeros_like(mean)
        else:
            std = np.std(arr, axis=0, ddof=1)
            sem = std / math.sqrt(arr.shape[0])
        ci = ci_factor * sem
        return mean, ci

    results_mean = {PART: {} for PART in partition_list}
    results_ci = {PART: {} for PART in partition_list}

    for PART in partition_list:
        for agg, metrics in results_all[PART].items():
            results_mean[PART][agg] = {}
            results_ci[PART][agg] = {}
            for metric_name, list_of_arrays in metrics.items():
                if not list_of_arrays:
                    continue

                if metric_name == 'per_class_accs':
                    if not list_of_arrays or not list_of_arrays[0]:
                        continue
                    
                    num_classes = len(list_of_arrays[0])
                    mean_per_class = []
                    ci_per_class = []
                    for class_idx in range(num_classes):
                        class_accs_across_runs = [run_data[class_idx] for run_data in list_of_arrays]
                        
                        lengths = [a.shape[0] for a in class_accs_across_runs]
                        min_len = min(lengths) if lengths else 0
                        class_accs_across_runs = [a[:min_len] for a in class_accs_across_runs]
                        
                        mean_vec_j, ci_vec_j = mean_and_ci(class_accs_across_runs, ci_factor=1.96)
                        mean_per_class.append(mean_vec_j)
                        ci_per_class.append(ci_vec_j)
                    
                    results_mean[PART][agg][metric_name] = mean_per_class
                    results_ci[PART][agg][metric_name] = ci_per_class
                else:
                    lengths = [a.shape[0] for a in list_of_arrays]
                    min_len = min(lengths) if lengths else 0
                    list_of_arrays = [a[:min_len] for a in list_of_arrays]
                    mean_vec, ci_vec = mean_and_ci(list_of_arrays, ci_factor=1.96)
                    results_mean[PART][agg][metric_name] = mean_vec
                    results_ci[PART][agg][metric_name] = ci_vec

    # --- Sauvegarde et Plots ---
    FOLDER_PLOT = f'plots/{DATASET_NAME}/{dt_string}/{MODEL_TYPE}/'
    if not os.path.exists(FOLDER_PLOT):
        os.makedirs(FOLDER_PLOT, exist_ok=True)


    # Plot class accuracy evolution
    plot_class_accuracy_evolution(
        results_mean, # Pass results_mean as it contains the averaged per_class_accs
        partition_list,
        save_file=os.path.join(FOLDER_PLOT, f'{ATTACK}_class_accuracy_evolution_mean_AGNEWS.png'),
        title=f'Class Accuracy Evolution (Mean over {N} runs) (AGNEWS, attack={ATTACK})',
        dataset_name='AGNEWS'
    )


    if N > 1:
        plot_mean_ci_partitions(results_mean, results_ci, partition_list,
                                save_file=os.path.join(FOLDER_PLOT, f'{ATTACK}_aggregator_mean_ci_{DATASET_NAME}.png'),
                                title=f'Aggregator mean ± CI ({DATASET_NAME}, attack={ATTACK})')

        plot_xi_A_partitions_mean_ci(results_mean, results_ci, partition_list,
                                    save_file=os.path.join(FOLDER_PLOT, f'{ATTACK}_xi_A_variance_mean_ci_{DATASET_NAME}.png'),
                                    title=f'xi, A and Variance mean ± CI ({ATTACK})')
        
    else:
        # Pour N=1, il faut extraire les données de la liste pour correspondre au format attendu par les fonctions de plot.
        results_single_run = {PART: {} for PART in partition_list}
        for PART in partition_list:
            for agg, metrics in results_all[PART].items():
                # On prend le premier (et unique) élément de chaque liste de métriques.
                results_single_run[PART][agg] = {k: v[0] for k, v in metrics.items() if v}

        plot_xi_A_partitions(results_single_run, partition_list,
                             save_file=os.path.join(FOLDER_PLOT, f'{ATTACK}_xi_A_partitions_AGNEWS.png'),
                            title=f'xi, A and Variance across partitions (AGNEWS, attack={ATTACK})')
        plot_partitions_aggregators(results_single_run, partition_list,
                                    save_file=os.path.join(FOLDER_PLOT, f'{ATTACK}_aggregator_comparison_partitions_AGNEWS.png'),
                                    title=f'Aggregator comparison across partitions (AGNEWS, attack={ATTACK})')

    
    print('Done.')