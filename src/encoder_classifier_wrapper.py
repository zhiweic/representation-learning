
from typing import Callable, Optional, Literal, Iterable, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Encoder→Pool→Head wrapper
# -----------------------------

class EncoderClassifier(nn.Module):
    def __init__(self, *, vocab_size: int, d_model: int, pad_id: int,
                 num_layers: int, block_ctor: Callable[[int], nn.Module],
                 pool: Literal["mean","cls"] = "mean",
                 cls_id: Optional[int] = None,
                 posenc: Optional[nn.Module] = None,
                 final_norm: str = "ln"):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pool = pool
        self.cls_id = cls_id
        self.posenc = posenc
        self.layers = nn.ModuleList([block_ctor(d_model) for _ in range(num_layers)])
        self.final_ln = make_norm(final_norm, d_model)
        self.head = nn.Linear(d_model, 1)

    def masked_mean(self, x, mask):
        m = mask.float().unsqueeze(-1)
        return (x*m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor):
        # optional prepend CLS id
        if self.pool == "cls" and self.cls_id is not None:
            B = ids.size(0)
            cls_col = torch.full((B,1), self.cls_id, dtype=ids.dtype, device=ids.device)
            ids = torch.cat([cls_col, ids], dim=1)
            mask = torch.cat([torch.ones(B,1, dtype=torch.bool, device=mask.device), mask], dim=1)

        x = self.embed(ids)  # [B,T,D]
        if self.posenc is not None:
            x = self.posenc(x)  # your SinusoidalPE returns x+pe

        for blk in self.layers:
            x = blk(x, pad_mask=mask, causal=False)

        x = self.final_ln(x)
        z = self.masked_mean(x, mask) if self.pool == "mean" else x[:,0,:]
        logits = self.head(z).view(-1)  # [B]
        return logits
    

# ----- Norms -----
class RMSNorm(nn.Module):
    # LN: learnable scale + bias (per feature), with mean-centering and variance normalization.
    # RMSNorm: learnable scale only (per feature), with RMS normalization (no centering)
    def __init__(self, d, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps
    def forward(self, x):
        # normalize by root-mean-square over features
        ms = x.float().pow(2).mean(dim=-1, keepdim=True)
        # Use rsqrt (avoids a divide and is a bit more numerically stable)
        x_hat = x * torch.rsqrt(ms + self.eps)
        return self.scale * x_hat

def make_norm(kind: str, d_model: int):
    kind = kind.lower()
    if kind in ("ln", "layernorm"):
        return nn.LayerNorm(d_model)
    elif kind in ("rms", "rmsnorm"):
        return RMSNorm(d_model)
    else:
        raise ValueError(f"unknown norm: {kind}")