import os
import random

import numpy as np
import torch
from torchvision import datasets, transforms



from plot import plot_partitions_aggregators, plot_xi_A_partitions
from simu import run_simulation

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n\n --------> Using device: {DEVICE} --------------\n\n')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)



# ---------------------- Example usage (cell) ----------------------
if __name__ == '__main__':
    # load MNIST (assume imports and datasets/transforms disponibles dans ton script principal)
    from torchvision import datasets, transforms
    from datetime import datetime

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # quick demo
    W = 10         # total workers
    R = 8         # regular (non-poisoned) workers
    T = 200      # iterations (petit pour demo)

    partition_list = ['iid', 'dirichlet', 'noniid']
    attack_list = ['static', 'dynamic', 'targeted', 'partial', 'confidence', 'backdoor', 
                   'signflip', 'scale', 'stealth', 'modelreplace']
    model_list = ['softmax', 'mlp', 'cnn']


    # datetime H-J-M-Y
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M")

    for MODEL in model_list:
        print(f'\n\n========== Model type: {MODEL} ==========')

        for ATTACK in attack_list:
            print(f'\n\n===== Attack type: {ATTACK} =====')

            # results storage
            results_by_partition = {}

            for PARTITION in partition_list:
                print(f'\n\n=== Partition: {PARTITION} ===')

                results = {}
                for agg in ['Mean', 'TriMean', 'FABA', 'CC', 'LFighter']:
                    print('Running aggregator:', agg)
                    # run_simulation doit retourner un dict avec clés 'accs','losses','xi','A' (comme dans ton code original)
                    stats = run_simulation(
                        trainset, testset,
                        W=W, R=R,
                        aggregator_name=agg,
                        partition=PARTITION,
                        attack_type=ATTACK,
                        flip_prob=0.8,
                        model_type=MODEL,
                        T=T,
                        local_batch=64,
                        gamma=0.01,
                        alpha=0.1
                    )
                    results[agg] = stats

                results_by_partition[PARTITION] = results

                # Optionnel : plot xi & A pour un aggregator de référence (ex: Mean)
                # try:
                #     plot_xi_A(results['Mean'], save_file=f'plots/xi_A_mean_{PARTITION}.png', title=f'Mean: xi and A over iterations ({PARTITION})')
                # except Exception as e:
                #     print('plot_xi_A skipped or failed:', e)

            # Après avoir collecté tous les résultats, tracer les comparaisons côte-à-côte
            FOLDER_PLOT = f'plots/{dt_string}/{MODEL}/'
            if not os.path.exists(FOLDER_PLOT):
                os.makedirs(FOLDER_PLOT, exist_ok=True)

            plot_xi_A_partitions(
                results_by_partition,
                partition_list,
                save_file=f'{FOLDER_PLOT}/{ATTACK}_xi_A_comparison_partitions.png',
                title='xi, A and Variance comparison across partitions'
            )

            plot_partitions_aggregators(
                results_by_partition,
                partition_list,
                save_file=f'{FOLDER_PLOT}/{ATTACK}_aggregator_comparison_partitions_mnist.png',
                title=f'Aggregator comparison across partitions (MNIST, attack={ATTACK})',
                show_loss=True  # mettre False si tu veux uniquement l'accuracy
            )

            # Save results as csv (un seul fichier consolidé)
            FOLDER = f'data_results/{dt_string}/{MODEL}/'
            if not os.path.exists(FOLDER):
                os.makedirs(FOLDER, exist_ok=True)
            filename = f'{FOLDER}/{ATTACK}_all_partitions_mnist_agg_results.csv'
            with open(filename, 'w') as f:
                f.write('Model,Attack,Partition,Aggregator,Iteration,TestAccuracy,TestLoss,Variance,xi,A\n')
                for PARTITION, part_results in results_by_partition.items():
                    for agg, stats in part_results.items():
                        n = len(stats['accs'])
                        for t in range(n):
                            xi_t = stats['xi'][t] if 'xi' in stats and t < len(stats['xi']) else ''
                            A_t = stats['A'][t] if 'A' in stats and t < len(stats['A']) else ''
                            f.write(f"{MODEL},{ATTACK},{PARTITION},{agg},{t},{stats['accs'][t]},{stats['losses'][t]},{stats['variance'][t]},{xi_t},{A_t}\n")
            print(filename, 'saved.')

            print('\n\n End of attack type:', ATTACK)

    print('Done.')