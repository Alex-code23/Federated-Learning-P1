import numpy as np
import torch
import torch.nn.functional as F

# ---------------------- Helpers ----------------------
def _to_tensor(x, dtype=None, device=None):
    """Convert numpy array or tensor-like to torch Tensor on device. If dtype given, cast."""
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    elif not isinstance(x, torch.Tensor):
        t = torch.tensor(x)
    else:
        t = x
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t

# ---------------------- Label poisoning attacks ----------------------

def static_flip(labels, prob=1.0, num_classes=10):
    """
    Flip label b -> 9-b with probability prob (works with tensors).
    Returns a torch.LongTensor on the same device as input.
    """
    labels = _to_tensor(labels)
    device = labels.device
    labels = labels.clone().long()

    n = labels.shape[0]
    if prob <= 0:
        return labels
    if prob >= 1.0:
        mask = torch.ones(n, dtype=torch.bool, device=device)
    else:
        mask = torch.rand(n, device=device) < float(prob)

    # flip: replace label by (num_classes - 1 - label)
    flipped = (num_classes - 1) - labels
    labels[mask] = flipped[mask]
    return labels.long()


def dynamic_flip_batch(model, x_batch, labels=None, device=None):
    """
    Return least-probable class indices as a torch.LongTensor on device.
    """
    # decide device
    if device is None:
        device = x_batch.device if isinstance(x_batch, torch.Tensor) else next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        xb = _to_tensor(x_batch, device=device)
        logits = model(xb)
        probs = F.softmax(logits, dim=1)          # tensor on device
        least = torch.argmin(probs, dim=1)
    return least.long().to(device)


def targeted_flip(labels, target_class=0, prob=1.0):
    """
    Replace with target_class with probability prob. Returns torch.LongTensor.
    """
    labels = _to_tensor(labels)
    device = labels.device
    labels = labels.clone().long()
    n = labels.shape[0]
    if prob <= 0:
        return labels
    if prob >= 1.0:
        mask = torch.ones(n, dtype=torch.bool, device=device)
    else:
        mask = torch.rand(n, device=device) < float(prob)
    labels[mask] = int(target_class)
    return labels.long()


def partial_poisoning(labels, frac=0.5, flip_fn=None, num_classes=10):
    """
    Flip a fraction `frac` of labels using flip_fn. If flip_fn is None, default to 9 - y.
    labels: Tensor or np.array. Returns torch.LongTensor.
    """
    labels = _to_tensor(labels)
    device = labels.device
    labels = labels.clone().long()
    n = labels.shape[0]
    if frac <= 0 or n == 0:
        return labels
    k = int(max(1, round(frac * n))) if frac > 0 else 0
    idx = torch.randperm(n, device=device)[:k]

    if flip_fn is None:
        flipped = (num_classes - 1) - labels
        labels[idx] = flipped[idx]
    else:
        # flip_fn must accept and return torch Tensor or numpy; try to call and convert result
        subset = labels[idx]
        out = flip_fn(subset)
        out = _to_tensor(out, dtype=torch.long, device=device)
        labels[idx] = out
    return labels.long()


def confidence_based_flip_batch(model, x_batch, labels, device=None, threshold=0.6, flip_to='least'):
    """
    Flip labels where model confidence < threshold.
    - flip_to='least' or integer target class.
    Returns torch.LongTensor of labels (on same device).
    """
    labels = _to_tensor(labels)
    device = device if device is not None else (x_batch.device if isinstance(x_batch, torch.Tensor) else next(model.parameters()).device)
    labels = labels.clone().long().to(device)

    model.eval()
    with torch.no_grad():
        xb = _to_tensor(x_batch, device=device)
        logits = model(xb)
        probs = F.softmax(logits, dim=1)  # on device
        max_conf, _ = torch.max(probs, dim=1)
        mask = max_conf < float(threshold)
        if torch.any(mask):
            if flip_to == 'least':
                least = torch.argmin(probs, dim=1)
                labels[mask] = least[mask]
            else:
                labels[mask] = int(flip_to)
    return labels.long()


# ============================================================
#                  BACKDOOR ATTACKS
# ============================================================

def add_simple_trigger(x_batch, trigger_value=1.0, size=3, location=(0, 0)):
    """
    Add small square trigger to images in a batch (tensor NxCxHxW).
    Returns tensor on same device as input.
    """
    xb = _to_tensor(x_batch)
    device = xb.device
    xb = xb.clone()
    if xb.dim() < 4:
        # assume Nx... reshape not handled
        return xb
    _, C, H, W = xb.shape
    y0, x0 = location
    x1 = min(x0 + size, W)
    y1 = min(y0 + size, H)
    xb[:, :, y0:y1, x0:x1] = float(trigger_value)
    return xb


def backdoor_poisoning(x_batch, labels, fraction=0.1, trigger_fn=add_simple_trigger, target_class=0):
    """
    Add trigger to a fraction of x_batch and set labels to target_class.
    Returns (xb_tensor, yb_tensor) on same device.
    """
    xb = _to_tensor(x_batch)
    yb = _to_tensor(labels, dtype=torch.long, device=xb.device)
    n = xb.size(0)
    if n == 0 or fraction <= 0:
        return xb, yb
    k = int(max(1, round(fraction * n))) if fraction > 0 else 0
    idx = torch.randperm(n, device=xb.device)[:k]
    xb = xb.clone()
    xb[idx] = trigger_fn(xb[idx])
    yb = yb.clone()
    yb[idx] = int(target_class)
    return xb, yb


# ============================================================
#                  GRADIENT-BASED ATTACKS 
# ============================================================

def sign_flip_attack(m, epsilon=1.0):
    """
    m: torch Tensor (any device). Return tensor same device.
    """
    m = _to_tensor(m)
    return -torch.sign(m) * float(epsilon)


def scale_attack(m, scale=5.0, max_norm=10.0, clip_value=1e3):
    """
    Scale gradient tensor by factor but clip to avoid overflow.
    Returns tensor on same device.
    """
    m = _to_tensor(m).clone().float()
    device = m.device
    out = m * float(scale)
    norm = torch.norm(out)
    if not torch.isfinite(norm):
        norm = torch.tensor(1e6, device=device)
    if max_norm is not None and norm > float(max_norm):
        out = out * (float(max_norm) / norm)
    out = torch.clamp(out, -float(clip_value), float(clip_value))
    return out


def stealthy_scaled_attack(m, target_dir, scale=1.0, max_delta=0.1):
    """
    Push m slightly toward target_dir (both tensors). Return tensor.
    """
    m = _to_tensor(m).clone().float()
    target_dir = _to_tensor(target_dir).float().to(m.device)
    delta = target_dir - m
    dnorm = torch.norm(delta)
    if dnorm > 0:
        add = delta / dnorm * min(float(max_delta), float(scale))
        m = m + add
    return m


def craft_model_replacement_vector(current_model_vec, target_model_vec, gamma):
    """
    Both inputs tensors; return attack vector tensor.
    """
    current_model_vec = _to_tensor(current_model_vec)
    target_model_vec = _to_tensor(target_model_vec).to(current_model_vec.device)
    return (current_model_vec - target_model_vec) / float(gamma)
