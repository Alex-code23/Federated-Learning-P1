import math
import os
import random

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms



from data_partition import partition_iid, partition_dirichlet, partition_noniid_by_class
from worker import Worker
from models import SoftmaxModel, TwoLayerMLP, CNNModel
from label_poisoning import dynamic_flip_batch, targeted_flip, partial_poisoning, confidence_based_flip_batch, backdoor_poisoning, static_flip, sign_flip_attack, scale_attack, stealthy_scaled_attack, craft_model_replacement_vector
from aggregators import agg_mean, agg_trimmed_mean, agg_coord_median, agg_centered_clipping, agg_faba_simple, agg_lfighter_simple


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
    'LFighter': lambda m: agg_lfighter_simple(m, n_clusters=2)
}

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
        workers.append(Worker(parts[w], dataset_train, device=device, poisoned=poisoned_flag, attack_type=attack_type, flip_prob=flip_prob, batch_size=local_batch))

    # create model
    if model_type == 'softmax':
        model = SoftmaxModel().to(device)
    elif model_type == 'mlp':
        model = TwoLayerMLP().to(device)
    elif model_type == 'cnn':
        model = CNNModel().to(device)
    else:
        raise ValueError('unknown model')

    loss_fn = nn.CrossEntropyLoss()

    # momentum states per worker
    D = sum(p.numel() for p in model.parameters())
    momenta = [None] * W

    agg_fn = AGGREGATORS[aggregator_name]

    accs = []
    losses = []
    variance = []
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
            x_batch, y_batch = worker.sample_batch()

            # ---------- Label-level attacks ----------
            if worker.poisoned:
                if attack_type == 'static':
                    y_batch = static_flip(y_batch, prob=flip_prob).long()
                elif attack_type == 'targeted':
                    y_batch = targeted_flip(y_batch, target_class=0, prob=flip_prob).long()
                elif attack_type == 'partial':
                    y_batch = partial_poisoning(y_batch, frac=0.5).long()
                elif attack_type == 'dynamic':
                    new_labels = dynamic_flip_batch(model, x_batch, y_batch, device)
                    y_batch = new_labels.long()
                elif attack_type == 'confidence':
                    new_labels = confidence_based_flip_batch(model, x_batch, y_batch, device, threshold=0.6)
                    y_batch = new_labels.long()
                elif attack_type == 'backdoor':
                    x_batch, y_batch = backdoor_poisoning(x_batch, y_batch, fraction=0.1, target_class=0)


            # put all on same device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # ---------- Local gradient computation ----------
            m_prev = momenta[w]
            # m, local_loss = worker.local_gradient(model, loss_fn, x_batch, y_batch, momentum_state=m_prev, alpha=alpha)
            m, local_loss = worker.local_gradient_on_model(model, loss_fn, x_batch, y_batch, alpha=alpha, momentum_state=m_prev)

            # ---------- Gradient-level attacks ----------
            if worker.poisoned:
                if attack_type == 'signflip':
                    m = sign_flip_attack(m, epsilon=1.0)
                elif attack_type == 'scale':
                    m = scale_attack(m, scale=5.0)
                elif attack_type == 'stealth':
                    target_dir = np.random.randn(*m.shape)
                    m = stealthy_scaled_attack(m, target_dir, scale=0.5)
                elif attack_type == 'modelreplace':
                    target_vec = np.random.randn(*m.shape)
                    m = craft_model_replacement_vector(m, target_vec, gamma)

            momenta[w] = m
            msgs.append(m)
            local_losses.append(local_loss)
            # store grads for heterogeneity/disturbance computation
            if worker.poisoned:
                grads_poisoned.append(m)
            else:
                grads_regular.append(m)

        # Keep messages on the device (GPU)
        msgs_tensor = torch.stack(msgs, dim=0)  # W x D tensor on device

        # compute heterogeneity xi and disturbance A
        # For metrics, we can move to CPU/numpy, as this is not part of the core training loop timing
        msgs_np = msgs_tensor.cpu().numpy()
        if len(grads_regular) > 0:
            grads_regular_np = torch.stack(grads_regular).cpu().numpy()
            ybar = np.mean(grads_regular_np, axis=0)
            xi_val = max(np.linalg.norm(g - ybar) for g in grads_regular_np)
        else:
            xi_val = 0.0
        if len(grads_poisoned) > 0:
            grads_poisoned_np = torch.stack(grads_poisoned).cpu().numpy()
            all_grads_np = np.vstack((grads_regular_np, grads_poisoned_np)) if len(grads_regular) > 0 else grads_poisoned_np
            A_val = max(np.linalg.norm(g - np.mean(all_grads_np, axis=0)) for g in grads_poisoned_np)
        else:
            A_val = 0.0
        heterogeneity_xi.append(xi_val)
        disturbance_A.append(A_val)

        # aggregate
        agg_vec_np = agg_fn(msgs_np)
        agg_vec = torch.from_numpy(agg_vec_np).to(device)

        # server update: x <- x - gamma * agg_vec
        # More efficient update by applying the flattened gradient vector directly
        pointer = 0
        with torch.no_grad():
            for param in model.parameters():
                numel = param.numel()
                param.data -= gamma * agg_vec[pointer:pointer+numel].view_as(param.data)
                pointer += numel

        # Evaluate periodically instead of every round
        if t % 3 == 0 or t == T - 1:
            model.eval()
            correct, total, total_loss = 0, 0, 0.0
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
        else: # if not evaluating, append previous value to keep lists aligned
            if accs: accs.append(accs[-1])
            if losses: losses.append(losses[-1])

        variance.append(np.var(msgs_np, axis=0).mean())

        if verbose and (t % max(1, T//10) == 0):
            print(f"Iter {t}/{T}  test_acc={accs[-1]:.4f} test_loss={losses[-1]:.4f} variance={variance[-1]:.4e} xi={xi_val:.4e} A={A_val:.4e}")

    stats = {
        'accs': accs,
        'losses': losses,
        'variance': variance,
        'xi': heterogeneity_xi,
        'A': disturbance_A
    }
    return stats
    


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
    T = 400      # iterations (petit pour demo)

    partition_list = ['iid', 'noniid']
    # one attack and one model
    ATTACK = 'static'
    MODEL = 'softmax'

    # Number samples of simulations running
    N = 10

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
    results_all = {PART: {agg: {'accs': [], 'losses': [], 'xi': [], 'A': [], 'variance': []}
                          for agg in AGGREGATORS.keys()}
                   for PART in partition_list}

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
                    flip_prob=0.8,
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
                # s'assurer que toutes les longueurs sont identiques
                lengths = [a.shape[0] for a in list_of_arrays]
                if len(set(lengths)) != 1:
                    # si mismatch (rare), tronquer à la longueur min
                    min_len = min(lengths)
                    list_of_arrays = [a[:min_len] for a in list_of_arrays]
                mean_vec, ci_vec = mean_and_ci(list_of_arrays, ci_factor=1.96)
                results_mean[PART][agg][metric_name] = mean_vec
                results_ci[PART][agg][metric_name] = ci_vec

    # fonctions de tracé avec bande d'incertitude
    def plot_mean_ci_partitions(results_mean, results_ci, partition_list, save_file=None, title=None, show_loss=True):
        n_part = len(partition_list)
        n_rows = 2 if show_loss else 1
        fig, axes = plt.subplots(n_rows, n_part, figsize=(5*n_part, 4*n_rows), squeeze=False)

        for col, PART in enumerate(partition_list):
            # Accuracy
            ax_acc = axes[0][col]
            acc_sup, acc_inf = 0,0
            for agg, stats_mean in results_mean[PART].items():
                if 'accs' not in stats_mean:
                    continue
                mean = stats_mean['accs']
                ci = results_ci[PART][agg]['accs']
                x = np.arange(len(mean))
                ax_acc.plot(x, mean, label=agg)
                ax_acc.fill_between(x, mean - ci, mean + ci, alpha=0.2)
                acc_sup = max(acc_sup, np.max(mean + ci))
                acc_inf = min(acc_inf, np.min(mean - ci))
            ax_acc.set_xlabel('Communication rounds')
            ax_acc.set_ylabel('Test accuracy')
            ax_acc.set_title(f'Accuracy — {PART}')
            ax_acc.set_ylim(acc_sup - acc_sup * 0.3 , acc_sup)
            ax_acc.grid(True)
            if col == 0:
                ax_acc.legend(loc='lower right', fontsize='small')

            # Loss
            if show_loss:
                ax_loss = axes[1][col]
                for agg, stats_mean in results_mean[PART].items():
                    if 'losses' not in stats_mean:
                        continue
                    mean = stats_mean['losses']
                    ci = results_ci[PART][agg]['losses']
                    x = np.arange(len(mean))
                    ax_loss.plot(x, mean, label=agg)
                    ax_loss.fill_between(x, mean - ci, mean + ci, alpha=0.2)
                ax_loss.set_xlabel('Communication rounds')
                ax_loss.set_ylabel('Test loss')
                ax_loss.set_title(f'Loss — {PART}')
                ax_loss.grid(True)
                if col == 0:
                    ax_loss.legend(loc='upper right', fontsize='small')

        plt.suptitle(title or '')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_file:
            folder = os.path.dirname(save_file)
            if folder and not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
            plt.savefig(save_file)
            plt.close(fig)
        else:
            plt.show()

    def plot_xi_A_partitions_mean_ci(results_mean, results_ci, partition_list, save_file=None, title=None):
        n_part = len(partition_list)
        fig, axes = plt.subplots(3, n_part, figsize=(5*n_part, 12), squeeze=False)

        for col, PART in enumerate(partition_list):
            # xi
            ax_xi = axes[0][col]
            for agg, stats_mean in results_mean[PART].items():
                if 'xi' not in stats_mean:
                    continue
                mean = stats_mean['xi']
                ci = results_ci[PART][agg]['xi']
                x = np.arange(len(mean))
                ax_xi.plot(x, mean, label=agg)
                ax_xi.fill_between(x, mean - ci, mean + ci, alpha=0.2)
            ax_xi.set_title(f'Heterogeneity (xi) — {PART}')
            ax_xi.set_xlabel('Rounds')
            ax_xi.set_ylabel('xi')
            ax_xi.grid(True)
            if col == 0:
                ax_xi.legend(loc='upper right', fontsize='small')

            # A
            ax_A = axes[1][col]
            for agg, stats_mean in results_mean[PART].items():
                if 'A' not in stats_mean:
                    continue
                mean = stats_mean['A']
                ci = results_ci[PART][agg]['A']
                x = np.arange(len(mean))
                ax_A.plot(x, mean, label=agg)
                ax_A.fill_between(x, mean - ci, mean + ci, alpha=0.2)
            ax_A.set_title(f'Disturbance (A) — {PART}')
            ax_A.set_xlabel('Rounds')
            ax_A.set_ylabel('A')
            ax_A.grid(True)
            if col == 0:
                ax_A.legend(loc='upper right', fontsize='small')

            # variance
            ax_var = axes[2][col]
            for agg, stats_mean in results_mean[PART].items():
                if 'variance' not in stats_mean:
                    continue
                mean = stats_mean['variance']
                ci = results_ci[PART][agg]['variance']
                x = np.arange(len(mean))
                ax_var.plot(x, mean, label=agg)
                ax_var.fill_between(x, mean - ci, mean + ci, alpha=0.2)
            ax_var.set_title(f'Variance of messages — {PART}')
            ax_var.set_xlabel('Rounds')
            ax_var.set_ylabel('Variance')
            ax_var.grid(True)
            if col == 0:
                ax_var.legend(loc='upper right', fontsize='small')

        plt.suptitle(title or 'Comparison of xi, A and Variance across partitions (mean ± CI)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if save_file:
            folder = os.path.dirname(save_file)
            if folder and not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)
            plt.savefig(save_file)
            plt.close(fig)
        else:
            plt.show()

    # Sauvegarde CSV consolidé avec moyenne et IC (ex: accuracy_mean, accuracy_ci)
    FOLDER = f'data_results/{dt_string}/{MODEL}/'
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER, exist_ok=True)
    csv_file = os.path.join(FOLDER, f'{ATTACK}_aggregators_mean_ci.csv')
    with open(csv_file, 'w') as f:
        # header
        f.write('Model,Attack,Partition,Aggregator,Iteration,Metric,Mean,CI\n')
        for PART in partition_list:
            for agg, metrics in results_mean[PART].items():
                for metric_name, mean_vec in metrics.items():
                    ci_vec = results_ci[PART][agg][metric_name]
                    for t in range(len(mean_vec)):
                        f.write(f"{MODEL},{ATTACK},{PART},{agg},{t},{metric_name},{mean_vec[t]},{ci_vec[t]}\n")
    print(csv_file, 'saved.')

    # Tracer et sauvegarder figures
    FOLDER_PLOT = f'plots/{dt_string}/{MODEL}/'
    if not os.path.exists(FOLDER_PLOT):
        os.makedirs(FOLDER_PLOT, exist_ok=True)

    plot_mean_ci_partitions(results_mean, results_ci, partition_list,
                            save_file=os.path.join(FOLDER_PLOT, f'{ATTACK}_aggregator_mean_ci_partitions.png'),
                            title=f'Aggregator mean ± CI across partitions (MNIST, attack={ATTACK})',
                            show_loss=True)

    plot_xi_A_partitions_mean_ci(results_mean, results_ci, partition_list,
                                 save_file=os.path.join(FOLDER_PLOT, f'{ATTACK}_xi_A_variance_mean_ci_partitions.png'),
                                 title=f'xi, A and Variance mean ± CI ({ATTACK})')



    print('Done.')


    
