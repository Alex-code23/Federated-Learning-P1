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
        x = torch.stack(data).to(self.device, non_blocking=True)
        y = torch.tensor(labels, dtype=torch.long, device=self.device)
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
    
    def local_gradient_on_model(self, model: nn.Module, loss_fn, x_batch, y_batch, alpha=0.1, momentum_state=None, use_amp=False):
        """
        Compute gradients of loss on x_batch/y_batch w.r.t model parameters.
        Returns flattened grad vector (torch.Tensor on same device) and loss scalar.
        Does NOT modify model parameters (only uses .grad).
        """
        device = next(model.parameters()).device
        model.train()
        # ensure grads zero
        model.zero_grad(set_to_none=True)
        # optional mixed precision
        if use_amp:
            from torch.cuda.amp import autocast, GradScaler
            with autocast():
                logits = model(x_batch)
                loss = loss_fn(logits, y_batch)
            scaler = torch.cuda.amp.GradScaler()
            scaler.scale(loss).backward()
            scaler.unscale_(torch.optim.Optimizer(model.parameters(), []))  # dummy optimizer for unscale
            # now grads are in param.grad (unscaled) after scaler.step in normal flow; careful but we can extract grads using scaler.get_scale()
            # Simpler: skip scaler for now unless fully integrating optimizer
        else:
            logits = model(x_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()

        grad_list = []
        for p in model.parameters():
            g = p.grad
            if g is None:
                grad_list.append(torch.zeros_like(p).view(-1))
            else:
                grad_list.append(g.detach().clone().view(-1))
        grad_vec = torch.cat(grad_list)
        # momentum
        if momentum_state is None:
            m = grad_vec.cpu()   # keep momenta as numpy if you want, else keep torch
        else:
            # if momentum_state stored as numpy, convert: but for speed better store momenta as torch on device
            m = ((1.0 - alpha) * momentum_state.to(device) + alpha * grad_vec).cpu()
        # clear grads
        model.zero_grad(set_to_none=True)
        return m, loss.item()
