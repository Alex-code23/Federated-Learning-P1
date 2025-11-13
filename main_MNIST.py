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
from models import SoftmaxModel, TwoLayerMLP, CNNModel
from label_poisoning import dynamic_flip_batch, targeted_flip, partial_poisoning, confidence_based_flip_batch, backdoor_poisoning, static_flip, sign_flip_attack, scale_attack, stealthy_scaled_attack, craft_model_replacement_vector
from plot import plot_results, plot_xi_A, plot_partitions_aggregators, plot_xi_A_partitions

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n\n --------> Using device: {DEVICE} --------------\n\n')
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
            x_batch, y_batch = worker.sample_batch(batch_size=local_batch)

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
        variance.append(np.var(msgs, axis=0).mean())

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