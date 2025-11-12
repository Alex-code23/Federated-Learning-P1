import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



from data_partition import partition_iid, partition_dirichlet, partition_noniid_by_class
from worker import Worker
from aggregators import AGGREGATORS
from models import SoftmaxModel, TwoLayerMLP
from label_poisoning import dynamic_flip_batch
from plot import plot_results, plot_xi_A

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)



# ---------------------- Simulation engine ----------------------

def run_simulation(
    dataset_train,
    dataset_test,
    W=10,
    R=9,
    aggregator_name='Mean',
    partition='iid',
    dirichlet_alpha=1.0,
    attack_type='static',
    flip_prob=1.0,
    poison_frac=1.0, # fraction of samples at poisoned worker to flip (static) or placeholder
    model_type='softmax',
    T=200, # communication rounds
    local_batch=32,
    gamma=0.01,
    alpha=0.1,
    device=DEVICE,
    verbose=True
):
    # partition data
    if partition == 'iid':
        parts = partition_iid(dataset_train, W)
    elif partition == 'dirichlet':
        parts = partition_dirichlet(dataset_train, W, alpha=dirichlet_alpha)
    elif partition == 'noniid':
        parts = partition_noniid_by_class(dataset_train, W)
    else:
        raise ValueError('unknown partition')

    # create workers
    poisoned_workers = list(range(R, W)) if (W-R) > 0 else []
    workers = []
    for w in range(W):
        poisoned_flag = (w in poisoned_workers)
        workers.append(Worker(parts[w], dataset_train, device=device, poisoned=poisoned_flag, attack_type=attack_type, flip_prob=flip_prob))

    # create model
    if model_type == 'softmax':
        model = SoftmaxModel().to(device)
    elif model_type == 'mlp':
        model = TwoLayerMLP().to(device)
    else:
        raise ValueError('unknown model')

    loss_fn = nn.CrossEntropyLoss()

    # momentum states per worker
    D = sum(p.numel() for p in model.parameters())
    momenta = [None] * W

    agg_fn = AGGREGATORS[aggregator_name]

    accs = []
    losses = []
    heterogeneity_xi = []
    disturbance_A = []

    test_loader = DataLoader(dataset_test, batch_size=256, shuffle=False)

    for t in range(T):
        msgs = []
        local_losses = []
        grads_regular = []
        grads_poisoned = []
        # For dynamic attack, poisoned worker uses global model to determine least likely class
        for w, worker in enumerate(workers):
            x_batch, y_batch = worker.sample_batch(batch_size=local_batch)
            if worker.poisoned and attack_type == 'dynamic':
                # flip labels according to global model
                new_labels = dynamic_flip_batch(model, x_batch, y_batch, device)
                y_batch = torch.from_numpy(new_labels).long()
            m_prev = momenta[w]
            m, local_loss = worker.local_gradient(model, loss_fn, x_batch, y_batch, momentum_state=m_prev, alpha=alpha)
            momenta[w] = m
            msgs.append(m)
            local_losses.append(local_loss)
            # store grads for heterogeneity/disturbance computation
            if worker.poisoned:
                grads_poisoned.append(m)
            else:
                grads_regular.append(m)

        msgs = np.stack(msgs, axis=0)  # W x D
        # compute heterogeneity xi and disturbance A
        if len(grads_regular) > 0:
            ybar = np.mean(np.stack(grads_regular, axis=0), axis=0) # np.stack to handle single regular worker case
            xi_val = max(np.linalg.norm(g - ybar) for g in grads_regular)
        else:
            xi_val = 0.0
        if len(grads_poisoned) > 0:
            A_val = max(np.linalg.norm(g - np.mean(np.vstack((grads_regular, grads_poisoned)) if len(grads_regular)>0 else np.stack(grads_poisoned)), axis=0) for g in grads_poisoned)
        else:
            A_val = 0.0
        heterogeneity_xi.append(xi_val)
        disturbance_A.append(A_val)

        # aggregate
        agg_vec = agg_fn(msgs)
        # server update: x <- x - gamma * agg_vec
        # convert agg_vec to model parameter shapes and apply gradient step
        pointer = 0
        state_dict = model.state_dict()
        new_state = {}
        for name, param in model.named_parameters():
            numel = param.numel()   # number of elements
            grad_part = torch.from_numpy(agg_vec[pointer:pointer+numel]).view(param.shape).to(device)
            new_state[name] = param.data - gamma * grad_part
            pointer += numel
        # load new parameters
        for name, param in model.named_parameters():
            param.data.copy_(new_state[name])

        # evaluate on test set (accuracy)
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss_val = loss_fn(logits, yb).item()
                total_loss += loss_val * xb.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += xb.size(0)
        accs.append(correct / total)
        losses.append(total_loss / total)

        if verbose and (t % max(1, T//10) == 0):
            print(f"Iter {t}/{T}  test_acc={accs[-1]:.4f} test_loss={losses[-1]:.4f} xi={xi_val:.4e} A={A_val:.4e}")

    stats = {
        'accs': accs,
        'losses': losses,
        'xi': heterogeneity_xi,
        'A': disturbance_A
    }
    return stats



# # ---------------------- Example usage (cell) ----------------------
# if __name__ == '__main__':
#     # load MNIST
#     transform = transforms.Compose([transforms.ToTensor()])
#     trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
#     testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

#     # quick demo
#     W = 5         # total workers
#     R = 3           # regular (non-poisoned) workers
#     T = 100  # keep small for demo; increase for reproduction (paper used larger iterations 2000+)

#     partition_list = ['iid', 'dirichlet', 'noniid']
#     ATTACK = 'static'  # 'static' or 'dynamic'

#     for PARTITION in partition_list:
#         print(f'\n\n=== Partition: {PARTITION} ===')

#         results = {}
#         for agg in ['Mean', 'TriMean', 'FABA', 'CC', 'LFighter']:     #  
#             print('\nRunning aggregator:', agg)
#             stats = run_simulation(trainset, testset, W=W, R=R, aggregator_name=agg, partition=PARTITION, attack_type=ATTACK, flip_prob=0.8, model_type='softmax', T=T, local_batch=64, gamma=0.01, alpha=0.1)
#             results[agg] = stats

#         # Plot results
#         FOLDER_PLOT = 'plots'
#         plot_results(results, save_file=f'{FOLDER_PLOT}/aggregator_comparison_{PARTITION}_mnist.png', title=f'Aggregator comparison (MNIST, {PARTITION}, {ATTACK})')
#         # plot xi & A for mean
#         plot_xi_A(results['Mean'], save_file=f'{FOLDER_PLOT}/xi_A_mean_{PARTITION}.png', title='Mean: xi and A over iterations')

#         # Save results as csv
#         FOLDER = 'data_results'
#         filename = f'{FOLDER}/{PARTITION}_mnist_agg_results.csv'
#         folder = os.path.dirname(filename)
#         if folder and not os.path.exists(folder):
#             os.makedirs(folder, exist_ok=True)
#         with open(filename, 'w') as f:
#             f.write('Aggregator,Iteration,TestAccuracy,TestLoss,xi,A,Partition\n')
#             for agg, stats in results.items():
#                 for t in range(len(stats['accs'])):
#                     f.write(f"{agg},{t},{stats['accs'][t]},{stats['losses'][t]},{stats['xi'][t]},{stats['A'][t]},{PARTITION}\n")
#             print(filename, 'saved.')
        


#     print('Done.')

import os
import matplotlib.pyplot as plt

# ---------------------- Plotting helpers ----------------------

def plot_partitions_aggregators(results_by_partition, partition_list, save_file=None, title=None, show_loss=True):
    """
    Affiche une figure 2xlen(partition_list) (accuracy et loss par partition).
    results_by_partition : dict mapping partition -> dict mapping aggregator -> stats (stats['accs'], stats['losses'], ...)
    partition_list : ordre des partitions à afficher (ex: ['iid','dirichlet','noniid'])
    show_loss : si False n'affiche que la ligne d'accuracy (1xN)
    """
    n_part = len(partition_list)
    n_rows = 2 if show_loss else 1
    fig, axes = plt.subplots(n_rows, n_part, figsize=(5*n_part, 4*n_rows), squeeze=False)

    # Palette simple (matplotlib choisira automatiquement si nombre variable)
    for col, PART in enumerate(partition_list):
        if PART not in results_by_partition:
            continue
        results = results_by_partition[PART]  # dict: agg -> stats

        # Accuracy subplot (row 0)
        ax_acc = axes[0][col]
        for agg, stats in results.items():
            ax_acc.plot(stats['accs'], label=agg)
        ax_acc.set_xlabel('Communication rounds')
        ax_acc.set_ylabel('Test accuracy')
        ax_acc.set_title(f'Accuracy — {PART}')
        ax_acc.grid(True)
        if col == 0:
            # Afficher légende sur le premier plot pour éviter répétitions
            ax_acc.legend(loc='lower right', fontsize='small')

        # Loss subplot (row 1) if demandé
        if show_loss:
            ax_loss = axes[1][col]
            for agg, stats in results.items():
                ax_loss.plot(stats['losses'], label=agg)
            ax_loss.set_xlabel('Communication rounds')
            ax_loss.set_ylabel('Test loss')
            ax_loss.set_title(f'Loss — {PART}')
            ax_loss.grid(True)
            if col == 0:
                ax_loss.legend(loc='upper right', fontsize='small')

    plt.suptitle(title or '')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # laisser de la place pour le suptitle

    if save_file:
        folder = os.path.dirname(save_file)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_file)
        plt.close(fig)
    else:
        plt.show()


def plot_xi_A(stats, save_file=None, title='Heterogeneity and Disturbance'):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(stats['xi'])
    plt.title('Heterogeneity (xi)')
    plt.xlabel('Communication rounds')

    plt.subplot(1,2,2)
    plt.plot(stats['A'])
    plt.title('Disturbance (A)')
    plt.xlabel('Communication rounds')

    plt.suptitle(title)
    if save_file:
        folder = os.path.dirname(save_file)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_file)
        plt.close()
    else:
        plt.show()

def plot_xi_A_partitions(results_by_partition, partition_list, save_file=None, title=None):
    """
    Affiche 2xlen(partition_list) : xi (haut) et A (bas)
    results_by_partition : dict[partition][aggregator] = stats
    """
    n_part = len(partition_list)
    fig, axes = plt.subplots(2, n_part, figsize=(5*n_part, 8), squeeze=False)

    for col, PART in enumerate(partition_list):
        if PART not in results_by_partition:
            continue
        results = results_by_partition[PART]

        # Ligne 0 = xi
        ax_xi = axes[0][col]
        for agg, stats in results.items():
            if 'xi' in stats:
                ax_xi.plot(stats['xi'], label=agg)
        ax_xi.set_title(f'Heterogeneity (xi) — {PART}')
        ax_xi.set_xlabel('Rounds')
        ax_xi.set_ylabel('xi')
        ax_xi.grid(True)
        if col == 0:
            ax_xi.legend(loc='upper right', fontsize='small')

        # Ligne 1 = A
        ax_A = axes[1][col]
        for agg, stats in results.items():
            if 'A' in stats:
                ax_A.plot(stats['A'], label=agg)
        ax_A.set_title(f'Disturbance (A) — {PART}')
        ax_A.set_xlabel('Rounds')
        ax_A.set_ylabel('A')
        ax_A.grid(True)
        if col == 0:
            ax_A.legend(loc='upper right', fontsize='small')

    plt.suptitle(title or 'Comparison of xi and A across partitions')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_file:
        folder = os.path.dirname(save_file)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        plt.savefig(save_file)
        plt.close(fig)
    else:
        plt.show()


# ---------------------- Example usage (cell) ----------------------
if __name__ == '__main__':
    # load MNIST (assume imports and datasets/transforms disponibles dans ton script principal)
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # quick demo
    W = 10         # total workers
    R = 8         # regular (non-poisoned) workers
    T = 10       # iterations (petit pour demo)

    partition_list = ['iid', 'dirichlet', 'noniid']
    attack_list = ['static', 'dynamic']

    for ATTACK in attack_list:
        print(f'\n\n=== Attack type: {ATTACK} ===')

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
                    model_type='softmax',
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
        FOLDER_PLOT = 'plots'
        if not os.path.exists(FOLDER_PLOT):
            os.makedirs(FOLDER_PLOT, exist_ok=True)

        plot_xi_A_partitions(
            results_by_partition,
            partition_list,
            save_file=f'{FOLDER_PLOT}/{ATTACK}_xi_A_comparison_partitions.png',
            title='xi and A comparison across partitions'
        )

        plot_partitions_aggregators(
            results_by_partition,
            partition_list,
            save_file=f'{FOLDER_PLOT}/{ATTACK}_aggregator_comparison_partitions_mnist.png',
            title=f'Aggregator comparison across partitions (MNIST, attack={ATTACK})',
            show_loss=True  # mettre False si tu veux uniquement l'accuracy
        )

        # Save results as csv (un seul fichier consolidé)
        FOLDER = 'data_results'
        if not os.path.exists(FOLDER):
            os.makedirs(FOLDER, exist_ok=True)
        filename = f'{FOLDER}/{ATTACK}_all_partitions_mnist_agg_results.csv'
        with open(filename, 'w') as f:
            f.write('Attack,Partition,Aggregator,Iteration,TestAccuracy,TestLoss,xi,A\n')
            for PARTITION, part_results in results_by_partition.items():
                for agg, stats in part_results.items():
                    n = len(stats['accs'])
                    for t in range(n):
                        xi_t = stats['xi'][t] if 'xi' in stats and t < len(stats['xi']) else ''
                        A_t = stats['A'][t] if 'A' in stats and t < len(stats['A']) else ''
                        f.write(f"{ATTACK},{PARTITION},{agg},{t},{stats['accs'][t]},{stats['losses'][t]},{xi_t},{A_t}\n")
        print(filename, 'saved.')

        print('\n\n End of attack type:', ATTACK)

    print('Done.')