import math
import os
import random

from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms


from aggregators import agg_mean, agg_trimmed_mean, agg_coord_median, agg_centered_clipping, agg_faba_simple, agg_lfighter_simple
from simu import run_simulation
from plot import plot_class_accuracy_evolution, plot_mean_ci_partitions, plot_xi_A_partitions_mean_ci, plot_partitions_aggregators, plot_xi_A_partitions




# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n\n --------> Using device: {DEVICE} --------------\n\n')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

AGGREGATORS = {
    'Mean': agg_mean,
    'TriMean': lambda m: agg_trimmed_mean(m, trim_ratio=0.2),
    # 'CoordMedian': agg_coord_median,
    'CC': lambda m: agg_centered_clipping(m, clip_threshold=1.0),
    'FABA': lambda m: agg_faba_simple(m, remove_frac=0.1),
    # 'LFighter': lambda m: agg_lfighter_simple(m, n_clusters=2)
}

# ---------------------- Simulation engine ----------------------


if __name__ == '__main__':
    # load MNIST (assume imports and datasets/transforms disponibles dans ton script principal)
    from torchvision import datasets, transforms
    from datetime import datetime

    transform = transforms.Compose([transforms.ToTensor()])
    DATASET = 'MNIST'
    if DATASET == 'MNIST':
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif DATASET == 'Fashion-MNIST':
        trainset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif DATASET == 'CIFAR-10':
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        print('[ERROR] DATASET')
        raise ValueError

    # quick demo
    W = 10         # total workers
    R = 6         # regular (non-poisoned) workers
    T = 2           # iterations (petit pour demo)

    partition_list = ['dirichlet','noniid'] # 'iid', 'dirichlet',
    # one attack and one model
    ATTACK = 'static'
    MODEL = 'softmax'

    # Number samples of simulations running
    N = 1

    # datetime H-J-M-Y
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M")

    # helper pour mean et IC à 95%
    def mean_and_ci(list_of_arrays, ci_factor=1.96):
        """
        list_of_arrays: list of 1D numpy arrays (mêmes longueurs)
        renvoie: mean (1D), ci (1D positive = rayon de l'IC)
        """
        arr = np.stack(list_of_arrays, axis=0)  # shape: (n_runs, T)
        mean = np.mean(arr, axis=0)
        # estimateur d'écart-type échantillonnal
        std = np.std(arr, axis=0, ddof=1)
        sem = std / math.sqrt(arr.shape[0])
        ci = ci_factor * sem
        return mean, ci

    # structures pour accumuler tous les résultats
    results_all = {PART: {agg: {'accs': [], 'losses': [], 'xi': [], 'A': [], 'variance': [], 'per_class_accs': []}
                          for agg in AGGREGATORS.keys()} for PART in partition_list}

    # boucle de N simulations (accumulation)
    for k in range(N):
        print(f"\n\n========== Simulation: {k} ==========")
        results_by_partition = {}

        for PARTITION in partition_list:
            print(f'\n\n=== Partition: {PARTITION} ===')
            results = {}
            for agg in AGGREGATORS.keys():
                print('Running aggregator:', agg)
                stats = run_simulation(
                    trainset, testset,
                    W=W, R=R,
                    aggregator_name=agg,
                    partition=PARTITION,
                    attack_type=ATTACK,
                    flip_prob=0.5,
                    model_type=MODEL,
                    T=T,
                    local_batch=64,
                    gamma=0.01,
                    alpha=0.1,
                    verbose=True  # moins de logs pendant N runs
                )
                results[agg] = stats

                # stocker les vecteurs (listes) pour calcul ultérieur
                # on convertit en numpy arrays pour plus de simplicité
                results_all[PARTITION][agg]['accs'].append(np.array(stats['accs']))
                results_all[PARTITION][agg]['losses'].append(np.array(stats['losses']))
                results_all[PARTITION][agg]['xi'].append(np.array(stats['xi']))
                results_all[PARTITION][agg]['A'].append(np.array(stats['A']))
                results_all[PARTITION][agg]['variance'].append(np.array(stats['variance']))
                
                # Store per_class_accs. Each element in stats['per_class_accs'] is a list for one class.
                # We convert these inner lists to numpy arrays for consistency.
                results_all[PARTITION][agg]['per_class_accs'].append([np.array(class_acc_list) for class_acc_list in stats['per_class_accs']])

            results_by_partition[PARTITION] = results

        print('Simulation', k, 'done.')

    # Après N runs: calculer moyenne et IC pour chaque metric / partition / aggregator
    results_mean = {PART: {} for PART in partition_list}
    results_ci = {PART: {} for PART in partition_list}

    for PART in partition_list:
        for agg, metrics in results_all[PART].items():
            results_mean[PART][agg] = {}
            results_ci[PART][agg] = {}
            for metric_name, list_of_arrays in metrics.items():
                # vérifier que l'on a au moins un run
                if len(list_of_arrays) == 0:
                    continue
                
                if metric_name == 'per_class_accs':
                    # list_of_arrays is like [ [c0_r1, c1_r1], [c0_r2, c1_r2] ]
                    # We need to transform it to [ [c0_r1, c0_r2], [c1_r1, c1_r2] ] for mean_and_ci
                    
                    if not list_of_arrays or not list_of_arrays[0]:
                        continue # Skip if no data or no classes in the first run
                    
                    num_classes = len(list_of_arrays[0])
                    
                    mean_per_class = []
                    ci_per_class = []
                    for class_idx in range(num_classes):
                        # Extract all runs' data for this specific class
                        class_j_accs_across_runs = [run_k_per_class_accs[class_idx] for run_k_per_class_accs in list_of_arrays]
                        
                        # Ensure all arrays for this class have the same length
                        lengths = [a.shape[0] for a in class_j_accs_across_runs]
                        if len(set(lengths)) != 1:
                            min_len = min(lengths)
                            class_j_accs_across_runs = [a[:min_len] for a in class_j_accs_across_runs]
                        
                        mean_vec_j, ci_vec_j = mean_and_ci(class_j_accs_across_runs, ci_factor=1.96)
                        mean_per_class.append(mean_vec_j)
                        ci_per_class.append(ci_vec_j)
                    
                    results_mean[PART][agg][metric_name] = mean_per_class
                    results_ci[PART][agg][metric_name] = ci_per_class
                else:
                    # For other metrics (accs, losses, xi, A, variance)
                    # s'assurer que toutes les longueurs sont identiques
                    lengths = [a.shape[0] for a in list_of_arrays]
                    if len(set(lengths)) != 1:
                        min_len = min(lengths)
                        list_of_arrays = [a[:min_len] for a in list_of_arrays]
                    mean_vec, ci_vec = mean_and_ci(list_of_arrays, ci_factor=1.96)
                    results_mean[PART][agg][metric_name] = mean_vec
                    results_ci[PART][agg][metric_name] = ci_vec

    # Sauvegarde CSV consolidé avec moyenne et IC (ex: accuracy_mean, accuracy_ci)
    FOLDER = f'data_results/{DATASET}/{dt_string}/{MODEL}/'
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER, exist_ok=True)
    csv_file = os.path.join(FOLDER, f'{ATTACK}_aggregators_mean_ci_{DATASET}.csv')
    with open(csv_file, 'w') as f: # Changed to 'w' to overwrite if exists
        f.write('Dataset,Model,Attack,Partition,Aggregator,Iteration,Metric,Mean,CI\n') # Header
        for PART in partition_list:
            for agg, metrics_mean in results_mean[PART].items():
                metrics_ci = results_ci[PART][agg]
                for metric_name, mean_data in metrics_mean.items():
                    ci_data = metrics_ci[metric_name]
                    if metric_name == 'per_class_accs':
                        # mean_data is a list of 1D arrays (one per class)
                        for class_idx, (class_mean_vec, class_ci_vec) in enumerate(zip(mean_data, ci_data)):
                            for t in range(len(class_mean_vec)):
                                f.write(f"{DATASET},{MODEL},{ATTACK},{PART},{agg},{t},{metric_name}_class_{class_idx},{class_mean_vec[t]},{class_ci_vec[t]}\n")
                    else:
                        # mean_data is a 1D array
                        for t in range(len(mean_data)):
                            f.write(f"{DATASET},{MODEL},{ATTACK},{PART},{agg},{t},{metric_name},{mean_data[t]},{ci_data[t]}\n")
    print(csv_file, 'saved.')

    # Tracer et sauvegarder figures
    FOLDER_PLOT = f'plots/{DATASET}/{dt_string}/{MODEL}/'
    if not os.path.exists(FOLDER_PLOT):
        os.makedirs(FOLDER_PLOT, exist_ok=True)

    # Plot class accuracy evolution
    plot_class_accuracy_evolution(
        results_mean, # Pass results_mean as it contains the averaged per_class_accs
        partition_list,
        save_file=os.path.join(FOLDER_PLOT, f'{ATTACK}_class_accuracy_evolution_mean_{DATASET}.png'),
        title=f'Class Accuracy Evolution (Mean over {N} runs) ({DATASET}, attack={ATTACK})',
        dataset_name=DATASET
    )

    if N > 1:


        plot_mean_ci_partitions(results_mean, results_ci, partition_list,
                                save_file=os.path.join(FOLDER_PLOT, f'{ATTACK}_aggregator_mean_ci_partitions_{DATASET}.png'),
                                title=f'Aggregator mean ± CI across partitions ({DATASET}, attack={ATTACK})',
                                show_loss=True)

        plot_xi_A_partitions_mean_ci(results_mean, results_ci, partition_list,
                                    save_file=os.path.join(FOLDER_PLOT, f'{ATTACK}_xi_A_variance_mean_ci_partitions_{DATASET}.png'),
                                    title=f'xi, A and Variance mean ± CI ({ATTACK})')
        
    else:
        plot_xi_A_partitions(results_by_partition, partition_list,
                            save_file=os.path.join(FOLDER_PLOT, f'{ATTACK}_xi_A_partitions_{DATASET}.png'),
                            title=f'xi, A and Variance across partitions ({DATASET}, attack={ATTACK})')
        
        plot_partitions_aggregators(results_by_partition, partition_list,
                                    save_file=os.path.join(FOLDER_PLOT, f'{ATTACK}_aggregator_comparison_partitions_{DATASET}.png'),
                                    title=f'Aggregator comparison across partitions ({DATASET}, attack={ATTACK})',
                                    show_loss=True)







    print('Done.')


    
