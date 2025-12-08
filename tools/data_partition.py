import numpy as np
import torch
from torch.utils.data import TensorDataset

# ---------------------- Data partition ----------------------

def _get_targets(dataset):
    """Helper function to get targets from torchvision dataset or TensorDataset."""
    if isinstance(dataset, TensorDataset):
        # For TensorDataset, the second tensor is usually the targets
        return dataset.tensors[1].cpu().numpy()
    elif hasattr(dataset, 'targets'):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            return targets.cpu().numpy()
        return np.array(targets)
    raise AttributeError("Dataset does not have a 'targets' attribute or is not a TensorDataset.")

def partition_iid(dataset, W):
    n = len(dataset)
    idx = np.arange(n)
    np.random.shuffle(idx)
    parts = np.array_split(idx, W)
    return parts

def partition_dirichlet(dataset, W, alpha=1.0):
    # Dataset targets assumed available as dataset.targets
    targets = _get_targets(dataset)
    K = int(targets.max()) + 1
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
    # MODIFIED: Distributes classes cyclically to workers.
    targets = _get_targets(dataset)
    K = int(targets.max()) + 1
    
    parts = [[] for _ in range(W)]
    class_indices = [np.where(targets == k)[0] for k in range(K)]

    for k in range(K):
        # Assign class k to worker k % W
        worker_idx = k % W
        parts[worker_idx].extend(class_indices[k])

    parts = [np.array(p, dtype=int) for p in parts]
    return parts

def partition_niid_pathological(dataset, W, shards_per_worker=4):
    """
    Cette méthode, inspirée par l'article original sur FedAvg (McMahan et al., 2017), 
    consiste à trier les données par classe, à les diviser en un certain nombre de "fragments" 
    (shards), puis à distribuer un petit nombre de ces fragments à chaque client. Le résultat 
    est que chaque client ne dispose que d'un nombre très limité de classes (par exemple, 2 
    pour MNIST). 
    """
    targets = _get_targets(dataset)
    n = len(dataset)
    
    # 1. Trier les indices de données par label
    sorted_indices = np.argsort(targets)
    
    # 2. Diviser les indices triés en shards
    num_shards = W * shards_per_worker
    shards = np.array_split(sorted_indices, num_shards)
    
    # 3. Assigner les shards aux workers
    shard_indices = np.arange(num_shards)
    np.random.shuffle(shard_indices)
    
    parts = [[] for _ in range(W)]
    for w in range(W):
        # Assigner `shards_per_worker` shards au worker `w`
        assigned_shard_indices = shard_indices[w * shards_per_worker : (w + 1) * shards_per_worker]
        for shard_idx in assigned_shard_indices:
            parts[w].extend(shards[shard_idx])
            
    parts = [np.array(p) for p in parts]
    return parts

def partition_noniid_by_class_count(dataset, W, classes_per_worker=2):
    targets = _get_targets(dataset)
    K = int(targets.max()) + 1
    
    class_indices = [np.where(targets == k)[0] for k in range(K)]

    # ---- Stage 1: ensure each worker gets at least 1 class ----
    worker_classes = [[] for _ in range(W)]
    class_list = list(range(K))
    np.random.shuffle(class_list)

    # First give each worker 1 class
    for w in range(W):
        k = class_list[w % K]
        worker_classes[w].append(k)

    # ---- Stage 2: add random classes until each worker has classes_per_worker ----
    for k in range(K):
        while sum(k in wc for wc in worker_classes) < classes_per_worker:
            w = np.random.randint(W)
            if k not in worker_classes[w]:
                worker_classes[w].append(k)

    # ---- Stage 3: assign samples ----
    parts = [[] for _ in range(W)]
    for w in range(W):
        for k in worker_classes[w]:
            parts[w].extend(class_indices[k])

    return [np.array(p, dtype=int) for p in parts]


if __name__ == '__main__':
    # test all partitions on simple dataset
    data = np.array([[i] for i in range(20000)])
    class SimpleDataset:
        def __init__(self, data):
            self.data = data
            self.targets = data.flatten() % 4  # 4 classes
        def __len__(self):
            return len(self.data)
        
    dataset = SimpleDataset(data)
    W = 4

    print("\n --- IID partition ---")
    print("[Explanation] In IID partition, each worker gets equal and random samples from all classes.")
    parts = partition_iid(dataset, W)
    for w in range(W):
        n = len(dataset.targets[parts[w]])
        label_proportion = [round(float(np.sum(dataset.targets[parts[w]] == c)/n), 3) for c in range(4)]
        print(f' Worker {w}:', len(parts[w]), ' Number for each labels:', label_proportion)

    print("\n --- Dirichlet partition ---")
    print("[Explanation] In Dirichlet partition, data is distributed based on Dirichlet distribution, leading to varying class distributions across workers.")
    print( " Dirichlet alpha=0.3")
    parts = partition_dirichlet(dataset, W, alpha=0.3)
    for w in range(W):
        n = len(dataset.targets[parts[w]])
        label_proportion = [round(float(np.sum(dataset.targets[parts[w]] == c)/n), 3) for c in range(4)]
        print(f' Worker {w}:', len(parts[w]), ' Number labels:', label_proportion)

    print(" Dirichlet alpha=10.0")
    parts = partition_dirichlet(dataset, W, alpha=10.0)
    for w in range(W):
        n = len(dataset.targets[parts[w]])
        label_proportion = [round(float(np.sum(dataset.targets[parts[w]] == c)/n), 3) for c in range(4)]
        print(f' Worker {w}:', len(parts[w]), ' Number labels:', label_proportion)

    print("\n --- Non-IID by class partition ---")
    print("[Explanation] In Non-IID by class partition, each worker is assigned data from only one class.")
    parts = partition_noniid_by_class(dataset, W)
    for w in range(W):
        n = len(dataset.targets[parts[w]])
        label_proportion = [round(float(np.sum(dataset.targets[parts[w]] == c)/n), 3) for c in range(4)]
        print(f' Worker {w}:',  len(parts[w]), ' Number labels:', label_proportion)

    print("\n --- Pathological Non-IID partition ---")
    print("[Explanation] In Pathological Non-IID partition, each worker gets a small number of shards, each shard containing data from a single class, leading to highly skewed class distributions.")
    parts = partition_niid_pathological(dataset, W, shards_per_worker=2)
    for w in range(W):
        n = len(dataset.targets[parts[w]])
        label_proportion = [round(float(np.sum(dataset.targets[parts[w]] == c)/n), 3) for c in range(4)]
        print(f' Worker {w}:', len(parts[w]), ' Number labels:', label_proportion)
    
    print("\n --- Non-IID by class count partition ---")
    print("[Explanation] In Non-IID by class count partition, each worker is assigned a fixed number of classes, ensuring diversity while maintaining non-IID characteristics.")
    parts = partition_noniid_by_class_count(dataset, W, classes_per_worker=2)
    for w in range(W):
        n = len(dataset.targets[parts[w]])
        label_proportion = [round(float(np.sum(dataset.targets[parts[w]] == c)/n), 3) for c in range(4)]
        print(f' Worker {w}:', len(parts[w]), ' Number labels:', label_proportion)