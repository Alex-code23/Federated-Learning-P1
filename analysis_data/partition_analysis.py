#!/usr/bin/env python3
"""
partition_analysis.py

Analyse des partitions "client-level" (federated).
- Input formats supported:
  * LEAF .npz or .json format (keys: "users", "user_data" or similar)
  * CSV with at least columns: client_id,label  (one row per exemple)
  * Directory structure:
      - class_dir/<class_label>/<files...> where filenames start with clientID_...
      - or client_dir/<client_id>/<files...> (labels unknown in that case)
- Output:
  * DataFrame summary (per-client): num_samples, num_classes_local, entropy_bits, js_divergence, top_class_fractions
  * Heatmap client x class (saved)
  * Entropy boxplot (saved)
- Usage examples:
    python3 partition_analysis.py --input path/to/data.npz --format leaf --outdir ./outputs
    python3 partition_analysis.py --input clients.csv --format csv --outdir ./outputs
    python3 partition_analysis.py --input /path/to/clients_folder --format folder --outdir ./outputs
"""
import os, sys, argparse, json
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy as sc_entropy

def load_leaf_npz(path):
    """Load LEAF-style .npz or .json exported partitions."""
    if path.endswith('.npz'):
        d = np.load(path, allow_pickle=True)
        if 'users' in d and 'user_data' in d:
            users = d['users'].tolist()
            user_data = d['user_data'].tolist()
            client_labels = {}
            for u in users:
                data = user_data[u]
                if isinstance(data, dict) and 'y' in data:
                    client_labels[u] = list(data['y'])
                elif isinstance(data, (list, np.ndarray)):
                    ys = []
                    for item in data:
                        if isinstance(item, dict) and 'y' in item:
                            ys.append(int(item['y']))
                    client_labels[u] = ys
                else:
                    client_labels[u] = []
            return client_labels
        else:
            return {}
    elif path.endswith('.json'):
        with open(path, 'r') as f:
            j = json.load(f)
        if 'user_data' in j:
            client_labels = {}
            for u, d in j['user_data'].items():
                if isinstance(d, dict) and 'y' in d:
                    client_labels[u] = d['y']
                elif isinstance(d, dict) and 'labels' in d:
                    client_labels[u] = d['labels']
                else:
                    ys = []
                    for k2,v2 in d.items():
                        if isinstance(v2, dict) and 'y' in v2:
                            ys.extend(v2['y'])
                    client_labels[u] = ys
            return client_labels
        else:
            client_labels = {}
            for k,v in j.items():
                if isinstance(v, list) and all(isinstance(x, (int,str)) for x in v):
                    client_labels[k] = v
            return client_labels
    else:
        raise ValueError("Unsupported LEAF file extension: " + path)

def load_csv_client_label(path, client_col='client_id', label_col='label'):
    """CSV with one sample per row. Columns specifying client_id and label."""
    df = pd.read_csv(path)
    if client_col not in df.columns or label_col not in df.columns:
        candidates = df.columns.tolist()
        raise ValueError(f"CSV missing expected columns. Found columns: {candidates}")
    client_labels = defaultdict(list)
    for _, row in df.iterrows():
        client_labels[str(row[client_col])].append(int(row[label_col]))
    return client_labels

def load_folder_clients(path):
    """
    Two heuristics:
    - If subfolders appear to be class labels (few folders, numeric names), treat as class folders
      and try to parse client_id from filenames prefix (clientID_...).
    - If subfolders appear to be client ids (many folders), return empty mapping and warn user.
    """
    client_labels = defaultdict(list)
    subs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path,d))]
    numeric_names = all(name.isdigit() for name in subs) if subs else False
    if numeric_names or len(subs) < 200:
        for cls in subs:
            cls_path = os.path.join(path, cls)
            for fname in os.listdir(cls_path):
                parts = fname.split('_')
                client = parts[0]
                try:
                    cls_idx = int(cls)
                except:
                    cls_idx = cls
                client_labels[client].append(cls_idx)
        return client_labels
    else:
        print("Detected many subfolders (likely client folders). Labels unknown -> provide CSV mapping filename->label.")
        return client_labels

def compute_metrics(client_labels, nbins=None):
    clients = sorted(client_labels.keys())
    all_labels = []
    for u in clients:
        all_labels.extend(client_labels[u])
    if len(all_labels) == 0:
        raise ValueError("No labels found in any client.")
    unique_labels = sorted(list(set(all_labels)))
    label_to_idx = {int(l): i for i,l in enumerate(unique_labels)}
    K = len(unique_labels) if nbins is None else nbins
    dist = np.zeros((len(clients), K), dtype=float)
    counts = np.zeros(len(clients), dtype=int)
    for i,u in enumerate(clients):
        ys = [label_to_idx[int(y)] for y in client_labels[u] if int(y) in label_to_idx]
        counts[i] = len(ys)
        if len(ys) > 0:
            c = Counter(ys)
            for lab_idx, cnt in c.items():
                dist[i, lab_idx] = cnt
            dist[i] = dist[i] / dist[i].sum()
    global_dist = np.sum(dist * counts[:,None], axis=0)
    if global_dist.sum() > 0:
        global_dist = global_dist / global_dist.sum()
    ent = np.array([sc_entropy(p, base=2) if p.sum()>0 else 0.0 for p in dist])
    def js_divergence(p, q):
        eps = 1e-12
        p = p.copy(); q = q.copy()
        p[p==0] = eps; q[q==0] = eps
        m = 0.5*(p+q)
        def kl(a,b):
            mask = a>0
            return np.sum(a[mask]*np.log2(a[mask]/b[mask]))
        return 0.5*kl(p,m) + 0.5*kl(q,m)
    js = np.array([js_divergence(dist[i], global_dist) for i in range(len(clients))])
    nonzero = np.sum(dist>1e-12, axis=1)
    df = pd.DataFrame({
        'client': clients,
        'num_samples': counts,
        'num_classes_local': nonzero,
        'entropy_bits': ent,
        'js_divergence': js
    })
    topk = []
    for i in range(len(clients)):
        arr = dist[i]
        top_inds = np.argsort(arr)[::-1][:5]
        topk.append({str(j): float(arr[j]) for j in top_inds if arr[j]>0})
    df['top_class_fractions'] = topk
    return df, dist, global_dist, unique_labels

def save_heatmap(dist, labels, outpath):
    fig, ax = plt.subplots(figsize=(10,6))
    im = ax.imshow(dist, aspect='auto')
    ax.set_xlabel('class index')
    ax.set_ylabel('client index')
    ax.set_title('Client × Class distribution (rows=clients)')
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def save_entropy_boxplot(ent, outpath):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.boxplot(ent)
    ax.set_xticklabels(['entropy (bits)'])
    ax.set_ylabel('entropy (bits) per client label distribution')
    ax.set_title('Entropy across clients')
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Analyse partitions client-level")
    parser.add_argument('--input', required=True, help='path to input (npz/json/csv/folder)')
    parser.add_argument('--format', required=True, choices=['leaf','csv','folder'], help='input format')
    parser.add_argument('--client-col', default='client_id', help='CSV client column name (for csv format)')
    parser.add_argument('--label-col', default='label', help='CSV label column name (for csv format)')
    parser.add_argument('--outdir', default='./outputs', help='output directory to save results')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.format == 'leaf':
        client_labels = load_leaf_npz(args.input)
    elif args.format == 'csv':
        client_labels = load_csv_client_label(args.input, client_col=args.client_col, label_col=args.label_col)
    elif args.format == 'folder':
        client_labels = load_folder_clients(args.input)
    else:
        raise ValueError("unsupported format")

    print(f"Loaded {len(client_labels)} clients.")
    if len(client_labels) == 0:
        print("Aucune étiquette trouvée. Vérifie le format d'entrée ou fournis un mapping filename->label pour les dossiers.")
        return
    df, dist, global_dist, unique_labels = compute_metrics(client_labels)
    df.to_csv(os.path.join(args.outdir, 'client_summary.csv'), index=False)
    heatpath = os.path.join(args.outdir, 'client_class_heatmap.png')
    save_heatmap(dist, unique_labels, heatpath)
    boxpath = os.path.join(args.outdir, 'entropy_boxplot.png')
    save_entropy_boxplot(df['entropy_bits'].values, boxpath)
    gdf = pd.DataFrame({'class': unique_labels, 'fraction': global_dist})
    gdf.to_csv(os.path.join(args.outdir, 'global_distribution.csv'), index=False)
    print(f"Saved outputs to {args.outdir}")
    print("Summary (top rows):")
    print(df.head())

if __name__ == '__main__':
    main()
