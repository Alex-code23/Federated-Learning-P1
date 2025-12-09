import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ----- Parameters -----
input_dim = 5
hidden_dim = 4
output_dim = 2
batch_size = 16
num_workers = 3
num_rounds = 500
lr = 0.2

# ----- Model -----
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

# Compute gradients
def compute_gradients(model, x, y):
    model.zero_grad()
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    grad_vec = torch.cat([p.grad.view(-1) for p in model.parameters()])
    return grad_vec, loss.item()


# ----- Données extrêmes pour worker 1 -----
torch.manual_seed(0)
workers_data = [torch.randn(batch_size, input_dim) for _ in range(num_workers)]

# Worker 1 a des données “explosives”
workers_data[1] = torch.randn(batch_size, input_dim) * 4.0
workers_labels = [torch.randint(0, output_dim, (batch_size,)) for _ in range(num_workers)]

# Tracking
grad_norm_poisoned = []
grad_norm_mean = []
loss_mean_hist = []
divergence = []

# Federated training loop
model = SimpleMLP()

for p in model.parameters():
    nn.init.normal_(p, mean=0.0, std=10.0)

for rnd in range(num_rounds):
    gradients = []
    losses = []
    for w in range(num_workers):
        g, loss = compute_gradients(model, workers_data[w], workers_labels[w])
        
        # STRONG POISONING: amplify malicious gradient heavily
        if w == 1:
            g = g * 20.0
            
        gradients.append(g)
        losses.append(loss)
    
    
    

    gradients = torch.stack(gradients)

    # Calcul de la moyenne SANS le méchant (worker 1)
    mean_grad = gradients.mean(dim=0)
    good_gradients = torch.stack([gradients[w] for w in range(num_workers) if w != 1])
    mean_grad_nice = good_gradients.mean(dim=0)
    
    # Record metrics
    grad_norm_poisoned.append(gradients[1].norm().item())
    grad_norm_mean.append(mean_grad_nice.norm().item())
    loss_mean_hist.append(sum(losses)/len(losses))
    divergence.append((gradients[1] - mean_grad).norm().item())
    
    # Apply update
    idx = 0
    for p in model.parameters():
        num = p.numel()
        p.data -= lr * mean_grad[idx:idx+num].view_as(p)
        idx += num

# ----- Plot -----
plt.figure(figsize=(8,5))
plt.plot(grad_norm_poisoned, label="Poisoned gradient norm")
plt.plot(grad_norm_mean, label="Mean gradient norm")
plt.plot(divergence, label=r"Divergence ‖$\nabla f$_poison - $\nabla f$‖")
plt.xlabel("Federated round")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig("plots/09-12-2025_explosion/divergence.png")
plt.show()
