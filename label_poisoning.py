import numpy as np
import torch
import torch.nn.functional as F


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


def targeted_flip(labels, target_class=0, prob=1.0):
    labels = labels.copy()
    mask = np.random.rand(len(labels)) < prob
    labels[mask] = target_class
    return labels


def partial_poisoning(labels, frac=0.5, flip_fn=lambda y: 9 - y):
    labels = labels.copy()
    n = len(labels)
    idx = np.random.choice(n, size=int(frac * n), replace=False)
    labels[idx] = flip_fn(labels[idx])
    return labels


def confidence_based_flip_batch(model, x_batch, labels, device, threshold=0.6, flip_to='least'):
    model.eval()
    with torch.no_grad():
        logits = model(x_batch.to(device))
        probs = F.softmax(logits, dim=1).cpu().numpy()
    max_conf = probs.max(axis=1)
    mask = max_conf < threshold
    new_labels = labels.copy()
    if flip_to == 'least':
        least = np.argmin(probs, axis=1)
        new_labels[mask] = least[mask]
    else:
        new_labels[mask] = flip_to
    return new_labels


# ============================================================
#                  BACKDOOR ATTACKS
# ============================================================

def add_simple_trigger(x_batch, trigger_value=1.0, size=3, location=(0, 0)):
    xb = x_batch.clone().cpu()
    _, C, H, W = xb.shape
    y0, x0 = location
    x1 = min(x0 + size, W)
    y1 = min(y0 + size, H)
    xb[:, :, y0:y1, x0:x1] = trigger_value
    return xb


def backdoor_poisoning(x_batch, labels, fraction=0.1, trigger_fn=add_simple_trigger, target_class=0):
    # Sécurise les types
    if isinstance(x_batch, np.ndarray):
        x_batch = torch.from_numpy(x_batch).float()
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels).long()

    n = x_batch.size(0)
    k = int(fraction * n)
    idx = np.random.choice(n, k, replace=False)
    xb = x_batch.clone()
    xb[idx] = trigger_fn(xb[idx])
    yb = labels.clone()
    yb[idx] = target_class
    return xb, yb


# ============================================================
#                  GRADIENT-BASED ATTACKS
# ============================================================

def sign_flip_attack(m, epsilon=1.0):
    return -np.sign(m) * epsilon


def scale_attack(m, scale=5.0, max_norm=None):
    out = m * scale
    if max_norm is not None:
        norm = np.linalg.norm(out)
        if norm > max_norm:
            out = out * (max_norm / norm)
    return out


def stealthy_scaled_attack(m, target_dir, scale=1.0, max_delta=0.1):
    m = m.copy()
    delta = target_dir - m
    dnorm = np.linalg.norm(delta)
    if dnorm > 0:
        add = delta / dnorm * min(max_delta, scale)
        m = m + add
    return m


def craft_model_replacement_vector(current_model_vec, target_model_vec, gamma):
    return (current_model_vec - target_model_vec) / gamma