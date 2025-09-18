import math, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset

# dataset + augs + losses + train + embed utils

# dataset
class TwoCropTSDataset(Dataset):
    """
    Time-series two-crop dataset.
    X: [N, T, C] (float)
    M: [N, T] bool pad mask (True=PAD). Pass None if no padding.
    view: callable that applies augmentations to a [T, C] tensor and returns [T, C]
    crop_len: fixed crop window length in timesteps
    pad_to_crop_len: if valid length < crop_len, right-pad with zeros and set mask=True on padded tail
    """
    def __init__(self, X: torch.Tensor, M: torch.Tensor | None, view, *,
                 crop_len: int = 128, pad_to_crop_len: bool = True):
        super().__init__()
        assert X.ndim == 3, "X must be [N,T,C]"
        self.X = X.float()
        self.M = M if M is None else M.bool()
        self.view = view
        self.crop_len = crop_len
        self.pad_to_crop_len = pad_to_crop_len

    def __len__(self): return self.X.size(0)

    def __getitem__(self, i):
        x = self.X[i]                         # [T, C]
        m = None if self.M is None else self.M[i]  # [T] bool (True=PAD)

        x1, m1 = self._crop_once(x, m)
        x2, m2 = self._crop_once(x, m)
        # return one sample [T, C]
        return (x1, m1), (x2, m2)

    def _crop_once(self, x: torch.Tensor, m: torch.Tensor | None):
        T = x.size(0)
        if m is None:
            # whole sequence is valid
            valid_T = T
            start_max = max(0, valid_T - self.crop_len)
            start = int(torch.randint(0, start_max + 1, (1,)))
            end = start + self.crop_len
            sub = x[start:end] if self.crop_len <= T else x
            if self.crop_len > T and self.pad_to_crop_len:
                sub, pad_mask = self._right_pad(sub, self.crop_len)
            else:
                pad_mask = torch.zeros(sub.size(0), dtype=torch.bool, device=x.device)
        else:
            # m: True=PAD, so valid = ~m
            valid_len = int((~m).sum().item())
            if valid_len == 0:
                # degenerate: make a single-timestep zero crop and pad
                sub = torch.zeros((0, x.size(1)), dtype=x.dtype, device=x.device)
                sub, pad_mask = self._right_pad(sub, self.crop_len)
                return self.view(sub), pad_mask

            w = min(valid_len, self.crop_len)
            start_max = max(0, valid_len - w)
            start = int(torch.randint(0, start_max + 1, (1,)))
            end = start + w
            # take from the *valid* prefix of length valid_len
            sub = x[:valid_len][start:end]     # [w, C]

            if w < self.crop_len and self.pad_to_crop_len:
                sub, pad_mask = self._right_pad(sub, self.crop_len)
            else:
                pad_mask = torch.zeros(sub.size(0), dtype=torch.bool, device=x.device)

        # apply view (augmentations) in [T, C] then return sub + mask
        sub = self.view(sub) if self.view is not None else sub
        return sub, pad_mask

    @staticmethod
    def _right_pad(sub: torch.Tensor, target_len: int):
        """Right-pad sub [t,C] to target_len with zeros; return padded sub and pad mask [target_len]."""
        t, C = sub.size(0), sub.size(1) if sub.ndim == 2 else (sub.size(0), 1)
        if t == target_len:
            pad_mask = torch.zeros(t, dtype=torch.bool, device=sub.device)
            return sub, pad_mask
        pad = torch.zeros((target_len - t, sub.size(1)), dtype=sub.dtype, device=sub.device)
        out = torch.cat([sub, pad], dim=0)
        mask = torch.zeros(target_len, dtype=torch.bool, device=sub.device)
        mask[t:] = True  # True=PAD
        return out, mask


# augmentation ops
class TSView:
    def __init__(self, jitter_std=0.02, scale_min=0.9, scale_max=1.1, cutout_p=0.3, cutout_len=12):
        self.jitter_std = jitter_std
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.cutout_p = cutout_p
        self.cutout_len = cutout_len

    def __call__(self, x: torch.Tensor):   # x: [T, C]
        # jitter
        x = x + torch.randn_like(x) * self.jitter_std
        # per-channel scaling
        scales = torch.empty(x.size(1), device=x.device).uniform_(self.scale_min, self.scale_max)
        x = x * scales
        # cutout (time masking) — zeros a contiguous block
        if torch.rand(()) < self.cutout_p and x.size(0) > self.cutout_len:
            start = int(torch.randint(0, x.size(0) - self.cutout_len + 1, (1,)))
            x[start:start+self.cutout_len] = 0.0
        return x
    
# loss
def nt_xent(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """
    z1, z2: [B, D], L2-normalized.
    Returns scalar loss (average over 2B positives).
    """
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)              # [2B, D]
    # Similarity matrix (cosine) scaled by temperature
    sim = z @ z.t()                              # cosine since z's are normalized -> [2B, 2B]
    # mask self-similarity
    diag = torch.eye(2*B, device=z.device, dtype=torch.bool)
    sim = sim / tau
    sim = sim - 1e9 * diag                       # remove self-pairs

    # positives: (i <-> i+B) and (i+B <-> i)
    # each anchor sees 2B−2 negatives
    targets = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)], dim=0).to(z.device)  # [2B]
    # Cross-entropy over rows (softmax over all 2B-1 others)
    loss = F.cross_entropy(sim, targets)
    return loss

def info_nce_two_way(z1, z2, tau=0.2):
    # z1,z2: [B,D], MUST be L2-normalized along dim=-1
    sim = (z1 @ z2.t()) / tau           # [B,B]
    y = torch.arange(z1.size(0), device=z1.device)
    # Each anchor sees only cross-view negatives (B−1 per anchor)
    return 0.5 * (F.cross_entropy(sim, y) + F.cross_entropy(sim.t(), y))

# training loop
def train_contrastive(model: nn.Module,loader,loss_fn, *,
                      epochs=20, lr=3e-4, tau=0.1, device=None):
    device = device or (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    for ep in range(1, epochs+1):
        model.train()
        total = 0.0
        for (ids1, m1), (ids2, m2) in loader:
            optim.zero_grad(set_to_none=True)
            ids1, m1 = ids1.to(device), m1.to(device)
            ids2, m2 = ids2.to(device), m2.to(device)

            z1 = model.encode(ids1)              # [B,d]
            z2 = model.encode(ids2)              # [B,d]
            loss = loss_fn(z1, z2, tau=tau)
            # loss = info_nce_two_way(z1, z2, tau=tau)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            g = sum((p.grad is not None and p.grad.abs().sum().item()) for p in model.parameters())
            assert g > 0, "no gradients flowed"
            optim.step()
            total += loss.item() * ids1.size(0)

        avg = total / len(loader.dataset)
        with torch.no_grad():
            # simple health checks
            z = torch.cat([z1, z2], 0)
            std = z.float().std(dim=0).mean().item()

            # intra-branch cosine structure (should be diag >> offdiag)
            sim_intra = (z1 @ z1.t()).float()
            B = sim_intra.size(0)
            diag = sim_intra.diag().mean().item()
            offd = (sim_intra.sum() - sim_intra.diag().sum()) / max(1, (B*B - B))
            # cross-branch diag/offdiag for the actual loss logits
            sim_cross = (z1 @ z2.t()).float()
            diag_x = sim_cross.diag().mean().item()
            offd_x = (sim_cross.sum() - sim_cross.diag().sum()) / max(1, (B*B - B))
        print(f"[epoch {ep:03d}] loss={avg:.4f} | z-std={std:.3f} |"
              f"intra diag/off={diag:.3f}/{offd:.3f} | cross diag/off={diag_x:.3f}/{offd_x:.3f}")

