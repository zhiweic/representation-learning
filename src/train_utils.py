import math, random
from dataclasses import dataclass
from typing import Callable, Optional, Literal, Iterable, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

# -----------------------------
# Utilities / config
# -----------------------------

@dataclass
class TrainConfig:
    device: Optional[torch.device] = None
    n_splits: int = 5
    epochs: int = 40
    batch_size: Optional[int] = None  # None = full-batch
    lr_enc: float = 3e-3
    lr_head: Optional[float] = None
    wd_enc: float = 0.0
    wd_head: float = 0.0
    patience: int = 4
    min_delta: float = 1e-3
    clip: float = 1.0
    smooth_eps: float = 0.0
    warmup_steps: int = 0
    use_adamw: bool = True
    seed: int = 42

def set_seed(seed: int, device: Optional[torch.device] = None):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device is None: device =(torch.device("mps") if torch.backends.mps.is_available()
                        else torch.device("cpu"))
    return device

# -----------------------------
# Optim groups / schedulers
# -----------------------------

def adamw_groups(model: nn.Module, lr_enc: float, lr_head: float, wd_enc: float, wd_head: float):
    decay, no_decay, head = [], [], []
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        if n.startswith("head."):
            head.append(p)
        elif n.endswith(".bias") or "norm" in n.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay,    "lr": lr_enc,  "weight_decay": wd_enc},
        {"params": no_decay, "lr": lr_enc,  "weight_decay": 0.0},
        {"params": head,     "lr": lr_head, "weight_decay": wd_head},
    ]

def linear_warmup(step: int, warmup_steps: int, base_lr: float):
    if warmup_steps <= 0: return base_lr
    if step >= warmup_steps: return base_lr
    return base_lr * (step / warmup_steps)

# -----------------------------
# Eval / Train
# -----------------------------

@torch.no_grad()
def eval_split(model: nn.Module, X: torch.Tensor, M: Optional[torch.Tensor], y_long: torch.Tensor, crit):
    model.eval()
    logits = (model(X, M) if M is not None else model(X)).view(-1)
    y_flt  = y_long.float()
    loss   = crit(logits, y_flt).item()
    probs  = torch.sigmoid(logits)
    preds  = (probs >= 0.5)
    acc    = (preds == y_long).float().mean().item()
    brier  = torch.mean((probs - y_flt)**2).item()
    margin = torch.mean(torch.abs(probs - 0.5)).item()
    return {"loss": loss, "acc": acc, "brier": brier, "margin": margin}

def kfold_train(
    X: torch.Tensor, y_float: torch.Tensor, y_long: torch.Tensor, mask: Optional[torch.Tensor],
    model_ctor: Callable[[], nn.Module],
    cfg: TrainConfig,
    groups_fn: Optional[Callable[..., list]] = adamw_groups,
):
    device = cfg.device or set_seed(cfg.seed)
    X = X.to(device)
    y_long = y_long.to(device)
    y_flt  = y_float.to(device)
    M = mask.to(device) if mask is not None else None
    if M is not None and M.dtype != torch.bool: M = M.bool()

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    y_np = y_long.detach().cpu().numpy()
    all_metrics = []

    for fold, (tr_np, va_np) in enumerate(skf.split(np.zeros(len(y_np)), y_np), 1):
        tr_idx = torch.tensor(tr_np, device=device)
        va_idx = torch.tensor(va_np, device=device)
        model = model_ctor().to(device)
        crit  = nn.BCEWithLogitsLoss()
        if groups_fn is None:
            param_groups = [{"params": model.parameters(), "lr": cfg.lr_enc, "weight_decay": cfg.wd_enc}]
        else:
            param_groups = groups_fn(model, lr_enc=cfg.lr_enc, lr_head=cfg.lr_head or cfg.lr_enc,
                                     wd_enc=cfg.wd_enc, wd_head=cfg.wd_head)
        Optim = torch.optim.AdamW if cfg.use_adamw else torch.optim.Adam
        opt = Optim(param_groups, betas=(0.9,0.98), eps=1e-9)
        for g in opt.param_groups: g.setdefault("base_lr", g["lr"])
        best, best_sd, wait, step = float("inf"), None, 0, 0

        for epoch in range(1, cfg.epochs+1):
            model.train()
            # build batches each epoch
            if cfg.batch_size is None or cfg.batch_size >= len(tr_idx):
                batches = [tr_idx]
            else:
                perm = tr_idx[torch.randperm(len(tr_idx), device=device)]
                batches = [perm[i:i+cfg.batch_size] for i in range(0, len(perm), cfg.batch_size)]

            total = 0.0
            for bi in batches:
                opt.zero_grad(set_to_none=True)
                logits = (model(X[bi], M[bi] if M is not None else None)).view(-1)
                targets = y_flt[bi]
                if cfg.smooth_eps and cfg.smooth_eps > 0:
                    targets = targets*(1-cfg.smooth_eps) + (1-targets)*cfg.smooth_eps
                loss = crit(logits, targets)
                loss.backward()
                if cfg.clip: nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
                step += 1
                if cfg.warmup_steps > 0:
                    for g in opt.param_groups:
                        g["lr"] = linear_warmup(step, cfg.warmup_steps, g["base_lr"])
                opt.step()
                total += loss.item() * len(bi)
            train_loss = total / len(tr_idx)

            val = eval_split(model, X[va_idx], M[va_idx] if M is not None else None, y_long[va_idx], crit)
            if epoch % 5 ==0:
                print(f"Fold {fold}, epoch {epoch}, {[k+': '+str(v) for k,v in val.items()]}")
            # early stop
            if best - val["loss"] < cfg.min_delta: wait += 1
            else: wait = 0
            if val["loss"] < best:
                best = val["loss"]
                best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if wait >= cfg.patience: break

        if best_sd is not None: model.load_state_dict(best_sd)
        all_metrics.append(eval_split(model, X[va_idx], M[va_idx] if M is not None else None, y_long[va_idx], crit))

    keys = all_metrics[0].keys()
    mean = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
    std  = {k: float(np.std( [m[k] for m in all_metrics])) for k in keys}
    return {"folds": all_metrics, "mean": mean, "std": std}
