import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader




from data_partition import partition_iid, partition_dirichlet, partition_noniid_by_class, partition_niid_pathological, partition_noniid_by_class_count
from worker import Worker
from aggregators import AGGREGATORS
from models import SoftmaxModel, TwoLayerMLP, CNNModel
from label_poisoning import dynamic_flip_batch, targeted_flip, partial_poisoning, confidence_based_flip_batch, backdoor_poisoning, static_flip, sign_flip_attack, scale_attack, stealthy_scaled_attack, craft_model_replacement_vector

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
    W,
    R,
    aggregator_name='Mean',
    partition='iid',
    dirichlet_alpha=1.0,
    attack_type='static',
    flip_prob=1.0,
    poison_frac=1.0, # fraction of samples at poisoned worker to flip (static) or placeholder
    model_type='softmax',
    T=200, # communication rounds
    model_factory=None, # Pour les modèles complexes comme en NLP
    collate_fn=None, # Pour le dataloader de test en NLP
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
    elif partition == 'pathological': # Nouvelle option
        parts = partition_niid_pathological(dataset_train, W, shards_per_worker=4)
    elif partition == 'class_count': # Nouvelle option
        parts = partition_noniid_by_class_count(dataset_train, W, classes_per_worker=2)

    else:
        raise ValueError('unknown partition')

    # create workers
    poisoned_workers = list(range(R, W)) if (W-R) > 0 else []
    workers = []
    for w in range(W):
        poisoned_flag = (w in poisoned_workers)
        workers.append(Worker(parts[w], dataset_train, device=device, poisoned=poisoned_flag, attack_type=attack_type, flip_prob=flip_prob, batch_size=local_batch))

    # create model
    # get input dim and flatten
    if model_factory:
        model = model_factory().to(device)
    elif model_type in ['softmax', 'mlp', 'cnn']:
        input_dim = dataset_train[0][0].numel()
        if model_type == 'softmax':
            model = SoftmaxModel(input_dim=input_dim).to(device)
        elif model_type == 'mlp':
            model = TwoLayerMLP(input_dim=input_dim).to(device)
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
    # Store per-class accuracies
    if hasattr(dataset_test, 'classes'):
        num_classes = len(dataset_test.classes)
    else: # For MNIST which doesn't have .classes attribute
        num_classes = 10
    
    per_class_accs = [[] for _ in range(num_classes)]

    heterogeneity_xi = []
    disturbance_A = []

    if collate_fn:
        test_loader = DataLoader(dataset_test, batch_size=256, shuffle=False, collate_fn=collate_fn)
    else:
        test_loader = DataLoader(dataset_test, batch_size=256, shuffle=False)

    for t in range(T):
        msgs = []
        local_losses = []
        grads_regular = []
        grads_poisoned = []
        # For dynamic attack, poisoned worker uses global model to determine least likely class
        for w, worker in enumerate(workers):
            # Pour le NLP, le batch peut contenir plus que x et y
            batch_data = worker.sample_batch()
            if model_type == 'text_classification':
                y_batch, x_batch, offsets = batch_data
            else:
                x_batch, y_batch = batch_data

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


            # put all on same device (déjà fait dans collate_fn pour NLP)
            if model_type != 'text_classification':
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

            # ---------- Local gradient computation ----------
            m_prev = momenta[w]
            if model_type == 'text_classification':
                # Le worker doit appeler le modèle avec les bons arguments
                m, local_loss = worker.local_gradient_on_model(model, loss_fn, (x_batch, offsets), y_batch, alpha=alpha, momentum_state=m_prev)
            else:
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

        grads_poisoned_np = torch.stack(grads_poisoned).cpu().numpy() if len(grads_poisoned) > 0 else np.array([])
        grads_regular_np = torch.stack(grads_regular).cpu().numpy() if len(grads_regular) > 0 else np.array([])
        all_grads_np = np.vstack((grads_regular_np, grads_poisoned_np)) if (len(grads_regular) > 0 and len(grads_poisoned) > 0) else (grads_regular_np if len(grads_poisoned) == 0 else grads_poisoned_np)
        if len(grads_regular) > 0:
            # grads_regular_np = torch.stack(grads_regular).cpu().numpy()
            # all_grads_np = np.vstack((grads_regular_np, grads_poisoned_np)) if len(grads_regular) > 0 else grads_poisoned_np
            ybar = np.mean(all_grads_np, axis=0)
            xi_val = max(np.linalg.norm(g - ybar) for g in grads_regular_np)
        else:
            xi_val = 0.0
        if len(grads_poisoned) > 0:
            # grads_poisoned_np = torch.stack(grads_poisoned).cpu().numpy()
            # all_grads_np = np.vstack((grads_regular_np, grads_poisoned_np)) if len(grads_regular) > 0 else grads_poisoned_np
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
        if t % 10 == 0 or t == T - 1:
            model.eval()
            correct, total, total_loss = 0, 0, 0.0
            class_correct = list(0. for i in range(num_classes))
            class_total = list(0. for i in range(num_classes))

            with torch.no_grad():
                for batch in test_loader:
                    if model_type == 'text_classification':
                        yb, xb, offsets = batch
                        logits = model(xb, offsets)
                        total = yb.size(0)
                    else:
                        xb, yb = batch
                        xb, yb = xb.to(device), yb.to(device)
                    logits = model(xb)
                    loss_val = loss_fn(logits, yb).item()
                    total_loss += loss_val * xb.size(0)
                    preds = logits.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += xb.size(0)

                    # Per-class accuracy
                    c = (preds == yb).squeeze()
                    for i in range(yb.size(0)):
                        label = yb[i]
                        if label < num_classes:
                            class_correct[label] += c[i].item()
                            class_total[label] += 1

            accs.append(correct / total)
            losses.append(total_loss / total)
            for i in range(num_classes):
                if class_total[i] > 0:
                    per_class_accs[i].append(class_correct[i] / class_total[i])
                else:
                    per_class_accs[i].append(0.0)
        else: # if not evaluating, append previous value to keep lists aligned
            if accs: accs.append(accs[-1])
            if losses: losses.append(losses[-1])
            for i in range(num_classes):
                if per_class_accs[i]: per_class_accs[i].append(per_class_accs[i][-1])

        variance.append(np.var(msgs_np, axis=0).mean())

        if verbose and (t % max(1, T//10) == 0):
            print(f"Iter {t}/{T}  test_acc={accs[-1]:.4f} test_loss={losses[-1]:.4f} variance={variance[-1]:.4e} xi={xi_val:.4e} A={A_val:.4e}")

    stats = {
        'accs': accs,
        'losses': losses,
        'variance': variance,
        'per_class_accs': per_class_accs,
        'xi': heterogeneity_xi,
        'A': disturbance_A
    }
    return stats
