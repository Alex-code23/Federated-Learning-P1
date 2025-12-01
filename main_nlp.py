import math
import os
import random
from datetime import datetime


import numpy as np
import torch
import torch.nn as nn
from torchtext.datasets import AG_NEWS 
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from simu import run_simulation
from plot import plot_mean_ci_partitions, plot_xi_A_partitions_mean_ci

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\n\n --------> Using device: {DEVICE} --------------\n\n')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- NLP Specifics ---

# 1. Tokenizer et vocabulaire
tokenizer = get_tokenizer('basic_english')
train_iter, _ = AG_NEWS(root='./data', split=('train', 'test'))

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x) - 1

# 2. Modèle de classification de texte
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

# 3. Fonction de collation pour le DataLoader
def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(DEVICE), text_list.to(DEVICE), offsets.to(DEVICE)

# Adapter le format pour la simulation (besoin de .targets)
class ListDataset:
    def __init__(self, list_data):
        self.data = list_data
        # AG_NEWS labels are 1-4, we map them to 0-3
        self.targets = [label_pipeline(item[0]) for item in list_data]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    # --- Configuration de la simulation ---
    W = 10
    R = 6
    T = 200
    N = 1 # Nombre de simulations pour la moyenne

    partition_list = ['dirichlet', 'noniid']
    ATTACK = 'static'
    
    # --- Chargement et préparation des données AG_NEWS ---
    # Les nouvelles versions de torchtext retournent des DataPipes.
    # On les convertit en listes pour pouvoir les manipuler facilement.
    train_iter, test_iter = AG_NEWS(root='./data', split=('train', 'test'))
    train_dataset = list(train_iter) # Convertit le DataPipe en liste
    test_dataset = list(test_iter)   # Convertit le DataPipe en liste

    trainset = ListDataset(train_dataset)
    testset = ListDataset(test_dataset)

    # --- Modèle ---
    num_class = len(set([label for (label, text) in train_dataset]))
    vocab_size = len(vocab)
    emsize = 64
    # Le modèle sera instancié dans run_simulation, ici on définit juste le type
    MODEL_TYPE = 'text_classification'

    # --- Simulation ---
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M")
    DATASET_NAME = 'AG_NEWS'

    # Structures pour accumuler les résultats sur N runs
    results_all = {PART: {agg: {'accs': [], 'losses': [], 'xi': [], 'A': [], 'variance': []}
                          for agg in ['Mean', 'TriMean', 'FABA', 'CC']} for PART in partition_list}

    for k in range(N):
        print(f"\n\n========== Simulation NLP: {k+1}/{N} ==========")
        for PARTITION in partition_list:
            print(f'\n\n=== Partition: {PARTITION} ===')
            for agg in results_all[PARTITION].keys():
                print('Running aggregator:', agg)
                
                # Note: `run_simulation` doit être modifié pour accepter `model_factory` et `collate_fn`
                stats = run_simulation(
                    trainset, testset,
                    W=W, R=R,
                    aggregator_name=agg,
                    partition=PARTITION,
                    attack_type=ATTACK,
                    model_type=MODEL_TYPE, # type spécial pour créer le bon modèle
                    model_factory=lambda: TextClassificationModel(vocab_size, emsize, num_class),
                    collate_fn=collate_batch, # Fonction de collation pour NLP
                    T=T,
                    local_batch=64,
                    gamma=0.1, # Le learning rate peut nécessiter un ajustement pour le NLP
                    alpha=0.1,
                    verbose=True
                )

                # Stocker les résultats
                results_all[PARTITION][agg]['accs'].append(np.array(stats['accs']))
                results_all[PARTITION][agg]['losses'].append(np.array(stats['losses']))
                results_all[PARTITION][agg]['xi'].append(np.array(stats['xi']))
                results_all[PARTITION][agg]['A'].append(np.array(stats['A']))
                results_all[PARTITION][agg]['variance'].append(np.array(stats['variance']))

    # --- Calcul de la moyenne et de l'intervalle de confiance ---
    def mean_and_ci(list_of_arrays, ci_factor=1.96):
        arr = np.stack(list_of_arrays, axis=0)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0, ddof=1)
        sem = std / math.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean)
        ci = ci_factor * sem
        return mean, ci

    results_mean = {PART: {} for PART in partition_list}
    results_ci = {PART: {} for PART in partition_list}

    for PART in partition_list:
        for agg, metrics in results_all[PART].items():
            results_mean[PART][agg] = {}
            results_ci[PART][agg] = {}
            for metric_name, list_of_arrays in metrics.items():
                if not list_of_arrays: continue
                lengths = [a.shape[0] for a in list_of_arrays]
                min_len = min(lengths) if lengths else 0
                list_of_arrays = [a[:min_len] for a in list_of_arrays]
                
                mean_vec, ci_vec = mean_and_ci(list_of_arrays, ci_factor=1.96)
                results_mean[PART][agg][metric_name] = mean_vec
                results_ci[PART][agg][metric_name] = ci_vec

    # --- Sauvegarde et Plots ---
    FOLDER_PLOT = f'plots/{DATASET_NAME}/{dt_string}/{MODEL_TYPE}/'
    if not os.path.exists(FOLDER_PLOT):
        os.makedirs(FOLDER_PLOT, exist_ok=True)

    if N > 1:
        plot_mean_ci_partitions(results_mean, results_ci, partition_list,
                                save_file=os.path.join(FOLDER_PLOT, f'{ATTACK}_aggregator_mean_ci_{DATASET_NAME}.png'),
                                title=f'Aggregator mean ± CI ({DATASET_NAME}, attack={ATTACK})')

        plot_xi_A_partitions_mean_ci(results_mean, results_ci, partition_list,
                                    save_file=os.path.join(FOLDER_PLOT, f'{ATTACK}_xi_A_variance_mean_ci_{DATASET_NAME}.png'),
                                    title=f'xi, A and Variance mean ± CI ({ATTACK})')
    
    print('Done.')