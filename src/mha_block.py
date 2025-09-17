import math, torch, torch.nn as nn, torch.nn.functional as F

# ----- (A) tiny RoPE helper -----
def apply_rope(q, k, base=10000.0):
    # q,k: [B,H,T,Dh]; split half-dims
    Dh = q.size(-1); assert Dh % 2 == 0, "RoPE needs even head dim"
    half = Dh // 2
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    T = q.size(-2)
    device = q.device
    dtype = q.dtype

    pos = torch.arange(T, device=device, dtype=dtype)[:, None]                    # [T,1]
    inv_freqs = torch.exp(-math.log(base) * (torch.arange(0, half, device=device, dtype=dtype) / half))[None, :]   # [1, half]
    ang = pos * inv_freqs                                                    # [T,half]
    sin, cos = torch.sin(ang), torch.cos(ang)                                              # [T,half]
    # broadcast to [B,H,T,half]
    sin = sin[None, None, :, :]; cos = cos[None, None, :, :]

    # rotate (x1, x2) -> (x1*cos - x2*sin, x2*cos + x1*sin)
    def rot(x1, x2):
        return x1 * cos - x2 * sin, x2 * cos + x1 * sin

    q1r, q2r = rot(q1, q2); k1r, k2r = rot(k1, k2) # [B, H, T, half]
    q = torch.cat([q1r, q2r], dim=-1)
    k = torch.cat([k1r, k2r], dim=-1)
    return q, k

# ----- (B) clipped T5-style per-head relative bias -----
class ClippedRelPosBias(nn.Module):
    def __init__(self, num_heads, max_rel=128):
        super().__init__()
        self.max_rel = max_rel
        self.table = nn.Parameter(torch.zeros(num_heads, 2*max_rel - 1))  # [H, 2R-1]
    def forward(self, T, device=None):
        device = device or self.table.device
        q = torch.arange(T, device=device)[:, None] # [T,1]  (query indices i)
        k = torch.arange(T, device=device)[None, :] # [1, T] (key indices j)
        rel = (k - q).clamp(-self.max_rel+1, self.max_rel-1) # [T, T]
        idx = rel + (self.max_rel - 1)              # map to [0..2R-2]
        return self.table[:, idx]                   # [H, T, T]


# --- Sinusoidal positional encoding ---
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x):  # x: [B, T, D]
        return x + self.pe[: x.size(1)]

def large_neg(dtype):
    # Safe additive mask sentinels
    return -1e4 if dtype in (torch.float16, torch.bfloat16) else -1e9

def build_sdpa_mask(
    pad_mask: torch.Tensor | None,  # [B,T] bool, True = real token
    rel_bias: torch.Tensor | None,  # [H,T,T] float or None
    *, B: int, T: int, H: int, dtype: torch.dtype, device: torch.device,
) -> torch.Tensor | None:
    """
    Returns a FLOAT additive mask for SDPA, shape [B,H,T,T], or None.
    Combines key padding + optional relative bias. (Causal can be handled by SDPA's is_causal=True.)
    """
    attn_mask = None

    if pad_mask is not None and pad_mask.dtype == torch.bool:
        # key-only mask → 0 for allowed keys, large negative for PAD keys
        allow_keys = pad_mask[:, None, None, :]  # [B,1,1,T], True = allowed
        neg = large_neg(dtype)
        # convert boolean allow-mask to float additive: disallowed → a large finite negative
        # Can't use -inf, the MPS backend turn it into nan.
        float_mask = (~allow_keys).to(dtype=dtype) * neg

    if rel_bias is not None:
        # rel_bias should be [H,T,T]; broadcast to batch and match dtype
        bias = rel_bias.to(dtype=dtype, device=device).unsqueeze(0).expand(B, -1, -1, -1)  # [B,H,T,T]
        attn_mask = bias if attn_mask is None else (float_mask + bias)

    return attn_mask  # float additive mask


# ----- (C) Norms -----
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

# ----- (D) SDPA-based MHA (pre-proj qkv, optional RoPE + RelBias) -----
class SDPAMHA(nn.Module):
    def __init__(self, d_model, num_heads, p_drop=0.0, use_rope=False, rel_bias=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dh = d_model // num_heads
        # Use chunk later to separate: fewer kernel launches, better cache locality, and lower Python/autograd overhead.
        self.qkv = nn.Linear(d_model, 3*d_model, bias=True)
        self.o   = nn.Linear(d_model, d_model, bias=True)
        self.drop_p = p_drop
        self.use_rope = use_rope
        self.rel_bias = rel_bias  # nn.Module or None

    def split_heads(self, x):  # [B,T,D] -> [B,H,T,Dh]
        B,T,D = x.shape
        x = x.view(B,T,self.h,self.dh).permute(0,2,1,3) # [B,T,D] -> [B, T, H, Dh] -> [B, H, T, Dh]
        return x
    def merge_heads(self, x):  # [B,H,T,Dh] -> [B,T,D]
        B,H,T,Dh = x.shape
        return x.permute(0,2,1,3).reshape(B,T,H*Dh)
        # permute usually makes the tensor non-contiguous, .reshape() is equivalent to .contiguous().view()

    def forward(self, x, pad_mask=None, causal=False):
        # pad_mask: [B,T] bool, True=real token
        B,T,D = x.shape
        q,k,v = self.qkv(x).chunk(3, dim=-1)               # [B,T,D] each
        q,k,v = self.split_heads(q), self.split_heads(k), self.split_heads(v)  # [B,H,T,Dh]

        if self.use_rope:
            q, k = apply_rope(q, k)                        # rotate Q/K in-place

        # ---- Build SDPA attn_mask ----
        # SDPA accepts:
        #  * boolean mask: True = ALLOW attention (opposite of nn.MultiheadAttention)
        #  * float mask: added to logits
        
        # Right before calling the encoder, once per batch:
        assert pad_mask is None or (pad_mask.sum(dim=1) > 0).all(), "Empty sequence in batch; add CLS/UNK or drop it."

        bias = None
        if self.rel_bias is not None:
            bias = self.rel_bias(T, device=x.device)       # [H,T,T]
        attn_mask = build_sdpa_mask(pad_mask, bias, B=B, T=T, H=self.h, dtype=q.dtype, device=q.device)

        for name, t in {"q": q, "k": k, "v": v}.items():
            if not torch.isfinite(t).all():
                raise RuntimeError(f"{name} has non-finite values")

        if attn_mask is not None and attn_mask.dtype.is_floating_point:
            if not torch.isfinite(attn_mask[attn_mask > -1e30]).all():  # ignore our -inf sentinels
                raise RuntimeError("attn_mask has non-finite (non -inf) values")
            
        # ---- SDPA (handles scale, softmax, dropout, matmul) ----
        # SDPA expects [B,H,T,Dh]; attn_mask broadcastable to [B,H,T,T]
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=(self.drop_p if self.training else 0.0),
            is_causal=bool(causal)
        )                                                  # [B,H,T,Dh]
        x = self.merge_heads(out)                          # [B,T,D]
        x = self.o(x)                     # [B,T,D]
        if pad_mask is not None:
            x = x * pad_mask.unsqueeze(-1).to(x.dtype)  # zero padded query rows
        return x
    

class PreLNEncoderBlockSDPA(nn.Module):
    """
    Pre-LN Encoder block that *composes* your SDPAMHA attention module.
    Pre-LN + residual wiring + FFN
    x -> x + Drop( Attn( LN(x) ) )
       -> x + Drop( FFN( LN(x) ) )

    Expects:
      - attn: a module like SDPAMHA with signature:
              attn(x: [B,T,D], pad_mask: Optional[Bool[B,T]], causal: bool) -> [B,T,D]
      - pad_mask: Bool[B,T], True = real token (not PAD)
      - causal: usually False for encoders
    """
    def __init__(
        self,
        d_model: int,
        *,
        attn: nn.Module,          # <-- pass your SDPAMHA instance here
        ff_mult: int = 4,
        p_drop: float = 0.1,
        norm: str = "ln",
        resid_mode: str = "plain",  # {"plain","scaled","rezero"}
    ):
        super().__init__()
        self.attn = attn                       # <-- your SDPAMHA
        self.ln1  = make_norm(norm, d_model)
        self.drop1 = nn.Dropout(p_drop)

        self.ln2  = make_norm(norm, d_model)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.drop2 = nn.Dropout(p_drop)

        self.mode = resid_mode
        if resid_mode == "rezero":
            self.g = nn.Parameter(torch.zeros(1))  # learnable gate
        elif resid_mode == "scaled":
            self.alpha = 0.5                       # constant residual scale

    def _resid(self, x, h):
        # residual add with optional scaling/gating
        if self.mode == "plain":
            return x + h
        elif self.mode == "scaled":
            return x + self.alpha * h
        elif self.mode == "rezero":
            return x + self.g * h
        else:
            raise ValueError(f"unknown resid_mode={self.mode}")

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None, causal: bool = False):
        """
        x:        [B,T,D]
        pad_mask: Bool[B,T], True = real token (not PAD). Will be passed through to SDPAMHA.
        causal:   usually False for encoders
        """
        B, T, D = x.shape
        if pad_mask is not None:
            assert pad_mask.dtype == torch.bool and pad_mask.shape == (B, T), "pad_mask must be Bool[B,T]"

        # --- Attention branch (Pre-LN) ---
        a_in = self.ln1(x)
        a_out = self.attn(a_in, pad_mask=pad_mask, causal=causal)  # expects [B,T,D] from your SDPAMHA
        x = self._resid(x, self.drop1(a_out))

        # --- FFN branch (Pre-LN) ---
        f_in = self.ln2(x)
        f_out = self.ff(f_in)
        x = self._resid(x, self.drop2(f_out))

        return x  # [B,T,D]
