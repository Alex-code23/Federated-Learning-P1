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