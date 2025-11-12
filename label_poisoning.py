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