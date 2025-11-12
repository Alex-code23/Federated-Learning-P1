import numpy as np
from sklearn.cluster import KMeans

SEED = 42


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