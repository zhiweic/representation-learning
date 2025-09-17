
from typing import Callable, Optional, Literal, Iterable, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.mha_block import make_norm


# --- Frontends ---
class TokenFrontend(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_id: int):
        super().__init__()
        self.pad_id = pad_id
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

    def forward(self, x, mask=None):  # x: [B, T] (Long)
        h = self.embed(x)             # [B, T, D]
        if mask is None:
            mask = (x == self.pad_id) # [B, T] True = pad
        return h, mask

class LinearFrontend(nn.Module):
    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)

    def forward(self, x, mask=None):  # x: [B, T, C] (Float)
        h = self.proj(x)  # -> [B, T, D]
        # if mask is None:
        #     # fixed-length crops typically have no padding
        #     B, T, _ = h.shape
        #     mask = torch.ones((B, T), dtype=torch.bool, device=h.device)
        return h, None
    

# -----------------------------
# Encoder→Pool→Head wrapper
# -----------------------------

class EncoderClassifier(nn.Module):
    def __init__(self, *, d_model: int,
                 num_layers: int, block_ctor: Callable[[int], nn.Module],
                 pool: Literal["mean","cls"] = "mean",
                 cls_id: Optional[int] = None,
                 posenc: Optional[nn.Module] = None,
                 final_norm: str = "ln",
                 final_norm_pos: str = "pre_pool",
                 frontend: nn.Module | None = None,
                 vocab_size: int | None = None,
                 pad_id: int | None = None,
                 proj_dim: int | None = None,
                 projector: nn.Module | None = None): 
        super().__init__()
        # Frontend selection (backwards compatible)
        if frontend is not None:
            self.frontend = frontend
        elif vocab_size is not None and pad_id is not None:
            self.frontend = TokenFrontend(vocab_size, d_model, pad_id)
        else:
            raise ValueError("Provide either `frontend` or (`vocab_size` and `pad_id`).")
        self.pool = pool
        self.cls_id = cls_id
        self.posenc = posenc
        self.layers = nn.ModuleList([block_ctor(d_model) for _ in range(num_layers)])
        self.final_ln = make_norm(final_norm, d_model)
        self.final_norm_pos = final_norm_pos
        self.head = nn.Linear(d_model, 1)

        # ---- projection head for contrastive (SimCLR-style) ----
        self.proj = None
        if proj_dim is not None:
            if projector is None:
                self.proj = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(inplace=True),                 # or GELU
                    nn.Linear(d_model, proj_dim),
                )
                nn.init.zeros_(self.proj[-1].bias)
            else:
                self.proj = projector

    def masked_mean(self, x, mask):
        if mask is None:
            return x.mean(dim=1)
        m = mask.float().unsqueeze(-1)
        return (x*m).sum(1) / m.sum(1).clamp(min=1.0)

    # ---- encode() returns pooled features, with optional projection+L2 norm ----
    def backbone(self, ids: torch.Tensor, mask: torch.Tensor | None = None):
        # optional prepend CLS id
        if self.pool == "cls" and self.cls_id is not None:
            B = ids.size(0)
            cls_col = torch.full((B,1), self.cls_id, dtype=ids.dtype, device=ids.device)
            ids = torch.cat([cls_col, ids], dim=1)
            mask = torch.cat([torch.ones(B,1, dtype=torch.bool, device=mask.device), mask], dim=1)

        x, mask = self.frontend(ids, mask)       # -> [B,T,D], [B,T]
        if self.posenc is not None:
            x = self.posenc(x)  # your SinusoidalPE returns x+pe
    
        for blk in self.layers:
            x = blk(x, pad_mask=mask, causal=False)

        if self.final_norm_pos == "pre_pool":
            x = self.final_ln(x)             # tokenwise LN: [B,T,D]

        z = self.masked_mean(x, mask) if self.pool == "mean" else x[:,0,:]

        if self.final_norm_pos == "post_pool":
            z = self.final_ln(z)              # vector LN: [B,D]
        return z
    
    @torch.no_grad()
    def features(self, x, mask=None):
        return self.backbone(x, mask)
    
    def encode(self, x, mask=None):
        h = self.backbone(x, mask)
        assert self.proj is not None, "proj_dim=None; set proj_dim to use projection head"
        z = self.proj(h)
        # z = F.normalize(z, p=2, dim=-1)
        z = F.normalize(z, dim=-1, eps=1e-8)
        return z
    
    def forward(self, ids: torch.Tensor, mask: torch.Tensor):
        z = self.backbone(x, mask)
        return self.head(z).view(-1) # [B]
    