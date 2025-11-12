import math
import copy
import random
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ---------------------- Utilities ----------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {DEVICE}')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Flatten and simple softmax model for softmax regression
class SoftmaxModel(nn.Module):
    def __init__(self, input_dim=28*28, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)

# Two-layer MLP used in paper (50 units each layer)
class TwoLayerMLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden=50, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, num_classes)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# CIFAR / CNN omitted for brevity — user can extend.

# ---------------------- Data partition ----------------------

def partition_iid(dataset, W):
    n = len(dataset)
    idx = np.arange(n)
    np.random.shuffle(idx)
    parts = np.array_split(idx, W)
    return parts

def partition_dirichlet(dataset, W, alpha=1.0):
    # Dataset targets assumed available as dataset.targets
    targets = np.array(dataset.targets)
    K = targets.max() + 1
    n = len(dataset)
    # sample proportions for each worker from Dirichlet for each class
    class_idx = [np.where(targets == k)[0] for k in range(K)]
    parts = [[] for _ in range(W)]
    for k in range(K):
        nk = len(class_idx[k])
        if nk == 0:
            continue
        proportions = np.random.dirichlet([alpha]*W)
        # split indices of this class according to proportions
        split = (proportions * nk).astype(int)
        # fix rounding
        diff = nk - split.sum()
        for i in range(diff):
            split[i % W] += 1
        ptr = 0
        for w in range(W):
            cnt = split[w]
            if cnt > 0:
                sel = class_idx[k][ptr:ptr+cnt]
                parts[w].extend(sel.tolist())
            ptr += cnt
    parts = [np.array(p) for p in parts]
    return parts

def partition_noniid_by_class(dataset, W):
    # assign each class to one worker (paper: non-iid case)
    targets = np.array(dataset.targets)
    K = targets.max() + 1
    assert K <= W, "Need W >= num classes for pure class per worker partition"
    parts = [[] for _ in range(W)]
    for k in range(K):
        idx = np.where(targets == k)[0]
        parts[k] = idx
    for w in range(K, W):
        parts[w] = np.array([], dtype=int)
    return parts

# ---------------------- Label poisoning attacks ----------------------

def static_flip(labels, prob=1.0):
    # flip b -> 9 - b with probability prob
    labels = labels.copy()
    mask = np.random.rand(len(labels)) < prob
    labels[mask] = 9 - labels[mask]
    return labels


def dynamic_flip_batch(model, x_batch, labels, device):
    # flip to least probable label under current global model
    model.eval()
    with torch.no_grad():
        logits = model(x_batch.to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()
    least = np.argmin(probs, axis=1)
    return least

# ---------------------- Aggregators ----------------------

def agg_mean(msgs: np.ndarray) -> np.ndarray:
    return msgs.mean(axis=0)

def agg_trimmed_mean(msgs: np.ndarray, trim_ratio=0.2) -> np.ndarray:
    # coordinate-wise trimmed mean
    W, D = msgs.shape
    k = int(np.floor(trim_ratio * W))
    out = np.zeros(D)
    for d in range(D):
        col = np.sort(msgs[:, d])
        if k > 0:
            col = col[k:W-k]
        out[d] = col.mean()
    return out

def agg_coord_median(msgs: np.ndarray) -> np.ndarray:
    return np.median(msgs, axis=0)

def agg_centered_clipping(msgs: np.ndarray, clip_threshold=1.0) -> np.ndarray:
    # centered clipping as in Karimireddy et al.: clip each vector to have norm at most r around mean
    mean = msgs.mean(axis=0)
    out = []
    for v in msgs:
        diff = v - mean
        norm = np.linalg.norm(diff)
        if norm > clip_threshold:
            diff = diff / norm * clip_threshold
        out.append(mean + diff)
    return np.array(out).mean(axis=0)

def agg_faba_simple(msgs: np.ndarray, remove_frac=0.1) -> np.ndarray:
    # simple FABA: iteratively remove the worker whose message is furthest from the average
    W = msgs.shape[0]
    remove_count = int(np.floor(remove_frac * W))
    idxs = list(range(W))
    cur = msgs.copy()
    for _ in range(remove_count):
        mean = cur.mean(axis=0)
        dists = np.linalg.norm(cur - mean, axis=1)
        rm = dists.argmax()
        cur = np.delete(cur, rm, axis=0)
        if len(cur) == 0:
            break
    return cur.mean(axis=0)

def agg_lfighter_simple(msgs: np.ndarray, n_clusters=2) -> np.ndarray:
    # rough LFighter: cluster messages and keep the largest cluster centroid
    W, D = msgs.shape
    if W <= n_clusters:
        return msgs.mean(axis=0)
    kmeans = KMeans(n_clusters=n_clusters, random_state=SEED).fit(msgs)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    largest = labels[np.argmax(counts)]
    kept = msgs[kmeans.labels_ == largest]
    return kept.mean(axis=0)

AGGREGATORS = {
    'Mean': agg_mean,
    'TriMean': lambda m: agg_trimmed_mean(m, trim_ratio=0.2),
    'CoordMedian': agg_coord_median,
    'CC': lambda m: agg_centered_clipping(m, clip_threshold=1.0),
    'FABA': lambda m: agg_faba_simple(m, remove_frac=0.1),
    'LFighter': lambda m: agg_lfighter_simple(m, n_clusters=2)
}

# ---------------------- Worker & server simulation ----------------------

class Worker:
    def __init__(self, data_idx, dataset, device='cpu', poisoned=False, attack_type='static', flip_prob=1.0):
        self.idx = data_idx
        self.dataset = dataset
        self.poisoned = poisoned
        self.device = device
        self.attack_type = attack_type
        self.flip_prob = flip_prob
        self.samples = data_idx
        # Precompute local DataLoader (shuffle at each epoch)

    def sample_batch(self, batch_size=32):
        sel = np.random.choice(self.samples, size=min(batch_size, len(self.samples)), replace=False)
        data, labels = zip(*[self.dataset[i] for i in sel])
        x = torch.stack(data)
        y = np.array(labels)
        if self.poisoned and self.attack_type == 'static':
            y = static_flip(y, prob=self.flip_prob)
            y = torch.from_numpy(y).long()
        return x, y

    def local_gradient(self, model: nn.Module, loss_fn, x_batch, y_batch, momentum_state=None, alpha=0.1):
        # compute gradient of loss on this batch and return momentum vector as message
        model_local = copy.deepcopy(model).to(self.device)
        model_local.train()
        # set parameters to the same as server model
        model_local.load_state_dict(model.state_dict())
        x_batch = x_batch.to(self.device)
        if isinstance(y_batch, np.ndarray):
            y_batch = torch.from_numpy(y_batch)
        y_batch = y_batch.to(torch.long).to(self.device)
        logits = model_local(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        # gather gradient vector
        grad_list = []
        for p in model_local.parameters():
            if p.grad is None:
                grad_list.append(torch.zeros_like(p).view(-1))
            else:
                grad_list.append(p.grad.view(-1).cpu())
        grad_vec = torch.cat(grad_list).numpy()
        # momentum update: m_t = (1-alpha)*m_{t-1} + alpha * grad
        if momentum_state is None:
            m = grad_vec
        else:
            m = (1.0 - alpha) * momentum_state + alpha * grad_vec
        return m, loss.item()

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

# ---------------------- Plotting helpers ----------------------

def plot_results(results_dict, title='Accuracies'):
    plt.figure(figsize=(8,5))
    for label, stats in results_dict.items():
        plt.plot(stats['accs'], label=label)
    plt.xlabel('Communication rounds')
    plt.ylabel('Test accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_xi_A(stats, title='Heterogeneity and Disturbance'):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(stats['xi'])
    plt.title('Heterogeneity (xi)')
    plt.subplot(1,2,2)
    plt.plot(stats['A'])
    plt.title('Disturbance (A)')
    plt.suptitle(title)
    plt.show()

# ---------------------- Example usage (cell) ----------------------
if __name__ == '__main__':
    # load MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # quick demo
    W = 5         # total workers
    R = 2           # regular (non-poisoned) workers
    T = 10  # keep small for demo; increase for reproduction (paper used larger iterations 2000+)
    PARTITION = 'iid'  # 'iid', 'dirichlet', 'noniid'

    results = {}
    for agg in ['Mean', 'TriMean', 'FABA', 'CC', 'LFighter']:     #  
        print('\nRunning aggregator:', agg)
        stats = run_simulation(trainset, testset, W=W, R=R, aggregator_name=agg, partition=PARTITION, attack_type='static', flip_prob=0.8, model_type='softmax', T=T, local_batch=64, gamma=0.01, alpha=0.1)
        results[agg] = stats

    plot_results(results, title=f'Aggregator comparison (MNIST, {PARTITION}, static flip)')
    # plot xi & A for mean
    plot_xi_A(results['Mean'], title='Mean: xi and A over iterations')

    # Save results as csv
    filename = f'{PARTITION}_mnist_agg_results.csv'
    with open(filename, 'w') as f:
        f.write('Aggregator,Iteration,TestAccuracy,TestLoss,xi,A,Partition\n')
        for agg, stats in results.items():
            for t in range(len(stats['accs'])):
                f.write(f"{agg},{t},{stats['accs'][t]},{stats['losses'][t]},{stats['xi'][t]},{stats['A'][t]},{PARTITION}\n")
        print(filename, 'saved.')
        


    print('Done.')
