import copy

import numpy as np
import torch
import torch.nn as nn

from label_poisoning import static_flip

# ---------------------- Worker & server simulation ----------------------

class Worker:
    def __init__(self, data_idx, dataset, device='cpu', poisoned=False, attack_type='static', flip_prob=1.0):
        self.idx = data_idx
        self.dataset = dataset
        self.poisoned = poisoned
        self.device = device
        self.attack_type = attack_type
        self.flip_prob = flip_prob
        self.samples = data_idx
        # Precompute local DataLoader (shuffle at each epoch)

    def sample_batch(self, batch_size=32):
        sel = np.random.choice(self.samples, size=min(batch_size, len(self.samples)), replace=False)
        data, labels = zip(*[self.dataset[i] for i in sel])
        x = torch.stack(data)
        y = np.array(labels)
        # if self.poisoned and self.attack_type == 'static':
        #     y = static_flip(y, prob=self.flip_prob)
        #     y = torch.from_numpy(y).long()
        return x, y

    def local_gradient(self, model: nn.Module, loss_fn, x_batch, y_batch, momentum_state=None, alpha=0.1):
        # compute gradient of loss on this batch and return momentum vector as message
        model_local = copy.deepcopy(model).to(self.device)
        model_local.train()
        # set parameters to the same as server model
        model_local.load_state_dict(model.state_dict())
        x_batch = x_batch.to(self.device)
        if isinstance(y_batch, np.ndarray):
            y_batch = torch.from_numpy(y_batch)
        y_batch = y_batch.to(torch.long).to(self.device)
        logits = model_local(x_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        # gather gradient vector
        grad_list = []
        for p in model_local.parameters():
            if p.grad is None:
                grad_list.append(torch.zeros_like(p).view(-1))
            else:
                grad_list.append(p.grad.view(-1).cpu())
        grad_vec = torch.cat(grad_list).numpy()
        # momentum update: m_t = (1-alpha)*m_{t-1} + alpha * grad
        if momentum_state is None:
            m = grad_vec
        else:
            m = (1.0 - alpha) * momentum_state + alpha * grad_vec
        return m, loss.item()