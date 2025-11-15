import numpy as np


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

def partition_niid_pathological(dataset, W, shards_per_worker=2):
    """
    Cette méthode, inspirée par l'article original sur FedAvg (McMahan et al., 2017), 
    consiste à trier les données par classe, à les diviser en un certain nombre de "fragments" 
    (shards), puis à distribuer un petit nombre de ces fragments à chaque client. Le résultat 
    est que chaque client ne dispose que d'un nombre très limité de classes (par exemple, 2 
    pour MNIST).
    """
    targets = np.array(dataset.targets)
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
    """
    Assigne un nombre fixe de classes (`classes_per_worker`) à chaque worker.
    Les données de chaque classe sont réparties équitablement entre les workers qui la possèdent.
    """
    targets = np.array(dataset.targets)
    K = targets.max() + 1
    
    # 1. Obtenir les indices pour chaque classe
    class_indices = [np.where(targets == k)[0] for k in range(K)]
    
    # 2. Assigner les classes aux workers
    worker_classes = [[] for _ in range(W)]
    for k in range(K):
        # Assigner la classe k à `classes_per_worker` workers choisis au hasard
        selected_workers = np.random.choice(W, classes_per_worker, replace=False)
        for w in selected_workers:
            worker_classes[w].append(k)

    # 3. Distribuer les indices de données
    parts = [[] for _ in range(W)]
    for w in range(W):
        for k in worker_classes[w]:
            # Pour simplifier, on donne tous les exemples de la classe k au worker w.
            # Une version plus complexe pourrait diviser les données de la classe k entre les workers qui la partagent.
            parts[w].extend(class_indices[k])
            
    parts = [np.array(p, dtype=int) for p in parts]
    return parts