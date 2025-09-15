from typing import Callable, Optional, Literal, Iterable, Dict, List, Tuple
import torch

# -----------------------------
# Text helpers (toy)
# -----------------------------

PAD_TOKEN = "<pad>"
CLS_TOKEN = "<cls>"
UNK_TOKEN = "<unk>"

def build_vocab(toks_list, min_freq=1, add_cls=False):
    from collections import Counter
    cnt = Counter(t for toks in toks_list for t in toks)
    itos = [PAD_TOKEN]
    if add_cls:
        itos.append(CLS_TOKEN)
    itos.append(UNK_TOKEN)
    for t, c in cnt.items():
        if c >= min_freq and t not in (PAD_TOKEN, CLS_TOKEN, UNK_TOKEN):
            itos.append(t)
    stoi = {t:i for i,t in enumerate(itos)}
    # “string-to-index”: a dict mapping each token (string) → its integer id
    # itos = “index-to-string”: a list (or dict) mapping each id → its token.
    return stoi, itos

def encode(sentences: Iterable[List[str]], stoi: Dict[str,int]):
    unk_id = len(stoi)  # optional: treat OOV as new id at end
    return [torch.tensor([stoi.get(t, unk_id) for t in toks], dtype=torch.long) for toks in sentences]

def pad_batch(batch_ids: List[torch.Tensor], pad_id: int):
    B = len(batch_ids)
    T = max((len(x) for x in batch_ids), default=0)
    X = torch.full((B, T), pad_id, dtype=torch.long)
    M = torch.zeros((B, T), dtype=torch.bool)
    for i, ids in enumerate(batch_ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        L = len(ids)
        if L:
            X[i, :L] = torch.tensor(ids, dtype=torch.long)
            M[i, :L] = True
    return X, M  # ids [B,T] long, mask [B,T] bool (True = real)

def tokenize(s): return s.strip().split()

def numericalize(tokens, stoi):
    return [stoi.get(t, stoi[UNK_TOKEN]) for t in tokens]

# dataset → tensors 
def make_tensors(sentences, labels, min_freq=1, add_cls=False):
    toks = [tokenize(s) for s in sentences]
    stoi, itos = build_vocab(toks, min_freq=min_freq, add_cls=add_cls)
    pad_id = stoi[PAD_TOKEN]
    cls_id = stoi.get(CLS_TOKEN, None)

    ids = [numericalize(t, stoi) for t in toks]
    X, M = pad_batch(ids, pad_id)

    y_long = torch.as_tensor(labels, dtype=torch.long)   # for CV/metrics
    y_float = y_long.float()                             # for BCEWithLogitsLoss

    return X, M, y_float, y_long, stoi, itos, pad_id, cls_id