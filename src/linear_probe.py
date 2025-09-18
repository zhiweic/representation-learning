import torch, torch.nn.functional as F

def split_idx(N, val=0.2, seed=0): # train test split
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(N, generator=g)
    n_val = int(N * val)
    return perm[n_val:], perm[:n_val]   # train_idx, val_idx

def train_linear_probe(Z, y, clf, *, epochs=200, lr=1e-2, wd=0.0, device=None):
    device = device or torch.device("cpu")
    Z = Z.to(device)
    y = y.to(device).long()
    W = clf.to(device)
    opt = torch.optim.AdamW(W.parameters(), lr=lr, weight_decay=wd)

    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(W(Z), y)
        loss.backward()
        opt.step()
    return W

@torch.no_grad()
def accuracy(W, Z, y, device=None):
    device = device or torch.device("cpu")
    logits = W(Z.to(device))
    pred = logits.argmax(dim=-1).cpu()
    return (pred == y.cpu().long()).float().mean().item()