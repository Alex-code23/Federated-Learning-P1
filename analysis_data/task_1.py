import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import entropy as sc_entropy
import pandas as pd
import os
np.random.seed(42)

# Parameters
K = 10           # number of classes
N_per_class = 400  # samples per class
D = 20           # feature dimension
W = 10           # number of workers
betas = [5, 1, 0.5, 0.1, 0.01]  # Dirichlet alphas to explore

# Generate synthetic dataset: K Gaussian blobs
X_list = []
y_list = []
for k in range(K):
    mean = np.random.randn(D) * 5  # separated means
    cov = np.eye(D) * 1.0
    Xk = np.random.multivariate_normal(mean, cov, size=N_per_class)
    yk = np.full(N_per_class, k, dtype=int)
    X_list.append(Xk)
    y_list.append(yk)

X = np.vstack(X_list)
y = np.concatenate(y_list)
N = X.shape[0]

# Utility: class indices
class_idx = {k: np.where(y == k)[0] for k in range(K)}

# Partitioning functions
def iid_partition(N, W):
    perm = np.random.permutation(N)
    sizes = [N // W + (1 if i < (N % W) else 0) for i in range(W)]
    parts = []
    start = 0
    for s in sizes:
        parts.append(perm[start:start+s].tolist())
        start += s
    return parts

def dirichlet_partition(y, W, alpha):
    parts = [[] for _ in range(W)]
    classes = np.unique(y)
    for c in classes:
        idx = np.where(y == c)[0].copy()
        props = np.random.dirichlet([alpha] * W)
        counts = (props * len(idx)).astype(int)
        diff = len(idx) - counts.sum()
        for i in range(diff):
            counts[i % W] += 1
        np.random.shuffle(idx)
        start = 0
        for w in range(W):
            cnt = counts[w]
            if cnt > 0:
                parts[w].extend(idx[start:start+cnt].tolist())
                start += cnt
    return parts

def one_class_per_worker(y, W):
    parts = [[] for _ in range(W)]
    classes = np.unique(y)
    for i, c in enumerate(classes):
        worker = i % W
        parts[worker].extend(class_idx[c].tolist())
    for w in range(W):
        if len(parts[w]) == 0:
            # steal one sample from first donor with >1 sample
            for donor in range(W):
                if len(parts[donor]) > 1:
                    parts[w].append(parts[donor].pop())
                    break
    return parts

# Metrics functions
def class_distribution(parts, y, K):
    dist = np.zeros((len(parts), K), dtype=float)
    for w, idx in enumerate(parts):
        if len(idx) == 0:
            continue
        counts = np.bincount(y[idx], minlength=K)
        dist[w] = counts / counts.sum()
    return dist

def worker_entropy(dist):
    ent = np.array([sc_entropy(p, base=2) for p in dist])
    # replace NaN for empty workers with 0
    ent = np.nan_to_num(ent, nan=0.0)
    return ent

def mean_js_divergence(dist, global_dist):
    def kl(p, q):
        mask = (p > 0)
        return np.sum(p[mask] * np.log2(p[mask] / q[mask]))
    js = []
    for p in dist:
        m = 0.5 * (p + global_dist)
        p_s = p.copy()
        q = global_dist.copy()
        eps = 1e-12
        p_s[p_s == 0] = eps
        q[q == 0] = eps
        m[m == 0] = eps
        val = 0.5 * kl(p_s, m) + 0.5 * kl(q, m)
        js.append(val)
    return np.array(js)

def num_nonzero_classes(dist):
    return np.sum(dist > 1e-12, axis=1)

# Analyze three canonical partitions
parts_iid = iid_partition(N, W)
parts_dir_beta1 = dirichlet_partition(y, W, alpha=1.0)
parts_oneclass = one_class_per_worker(y, W)

# global distribution
global_dist = np.bincount(y, minlength=K) / N

# compute metrics
def summarize(parts, name):
    dist = class_distribution(parts, y, K)
    ent = worker_entropy(dist)
    js = mean_js_divergence(dist, global_dist)
    nonzero = num_nonzero_classes(dist)
    summary = {
        'partition': name,
        'mean_entropy': float(np.nanmean(ent)),
        'std_entropy': float(np.nanstd(ent)),
        'mean_js': float(np.nanmean(js)),
        'std_js': float(np.nanstd(js)),
        'mean_nonzero_classes': float(np.nanmean(nonzero)),
        'std_nonzero_classes': float(np.nanstd(nonzero)),
    }
    return summary, dist, ent, js, nonzero

sum_iid, dist_iid, ent_iid, js_iid, nz_iid = summarize(parts_iid, 'IID')
sum_dir1, dist_dir1, ent_dir1, js_dir1, nz_dir1 = summarize(parts_dir_beta1, 'Dirichlet_beta1')
sum_one, dist_one, ent_one, js_one, nz_one = summarize(parts_oneclass, 'One_class_per_worker')

df_summary = pd.DataFrame([sum_iid, sum_dir1, sum_one]).set_index('partition')

# Also explore Dirichlet for different betas
rows = []
for b in betas:
    parts = dirichlet_partition(y, W, alpha=b)
    s, _, _, _, _ = summarize(parts, f'Dirichlet_beta_{b}')
    rows.append(s)
df_betas = pd.DataFrame(rows).set_index('partition')


# Plot 1: class distribution heatmaps (workers x classes)
fig1, ax1 = plt.subplots(figsize=(8, 4))
ax1.imshow(dist_iid, aspect='auto')
ax1.set_title('IID: worker x class distribution (rows=workers)')
ax1.set_xlabel('class')
ax1.set_ylabel('worker')
plt.tight_layout()
plt.show()

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.imshow(dist_dir1, aspect='auto')
fig2.suptitle('Dirichlet (beta=1): worker x class distribution')
ax2.set_xlabel('class')
ax2.set_ylabel('worker')
plt.tight_layout()
plt.show()

fig3, ax3 = plt.subplots(figsize=(8, 4))
ax3.imshow(dist_one, aspect='auto')
ax3.set_title('One-class-per-worker: worker x class distribution')
ax3.set_xlabel('class')
ax3.set_ylabel('worker')
plt.tight_layout()
plt.show()

# Plot 2: Entropy distributions as boxplots
fig4, ax4 = plt.subplots(figsize=(6,4))
ax4.boxplot([ent_iid, ent_dir1, ent_one], labels=['IID','Dirichlet β=1','One-class'])
ax4.set_ylabel('Entropy (bits) per worker label distribution')
ax4.set_title('Entropy across partitions')
plt.tight_layout()
plt.show()

# Plot 3: PCA visualization of features colored by true class and by worker (one-class case)
pca = PCA(n_components=2)
X2 = pca.fit_transform(X)

fig5, (ax5a, ax5b) = plt.subplots(1,2, figsize=(12,5))
for k in range(K):
    idx = np.where(y==k)[0]
    ax5a.scatter(X2[idx,0], X2[idx,1], s=8, label=str(k))
ax5a.set_title('PCA by true class (all data)')
ax5a.legend(loc='best', ncol=2, fontsize='small')

worker_of_idx = np.full(N, -1)
for w, idxs in enumerate(parts_oneclass):
    worker_of_idx[idxs] = w
for w in range(W):
    idx = np.where(worker_of_idx==w)[0]
    ax5b.scatter(X2[idx,0], X2[idx,1], s=8, label='w'+str(w))
ax5b.set_title('PCA colored by assigned worker (One-class-per-worker)')
ax5b.legend(loc='best', ncol=2, fontsize='small')
plt.tight_layout()
plt.show()

# Save key figures to files for download if desired
outdir = "./visualisations/partition_demo_outputs"
os.makedirs(outdir, exist_ok=True)
fig1.savefig(os.path.join(outdir, "dist_iid.png"))
fig2.savefig(os.path.join(outdir, "dist_dir1.png"))
fig3.savefig(os.path.join(outdir, "dist_oneclass.png"))
fig4.savefig(os.path.join(outdir, "entropy_boxplot.png"))
fig5.savefig(os.path.join(outdir, "pca_oneclass.png"))

print(f"Saved example figures to {outdir}")

text_summary = """
Observations:
- IID has high entropy per worker, each worker sees many classes.
- Dirichlet (β smaller => more heterogeneity) shows reduced entropy and more variation between workers.
- One-class-per-worker yields near-zero entropy (each worker sees 1 class), high JS divergence vs global,
  and is an extreme, artificial partition: real federated data rarely have zero class overlap.
- Visual PCA shows clear cluster-to-worker mapping in the one-class case: workers simply mirror classes.
"""
print(text_summary)
