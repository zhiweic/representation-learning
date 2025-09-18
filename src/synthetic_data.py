import numpy as np
import torch

def synth_trips(
    N: int = 1200,
    T: int = 256,
    *,
    dt: float = 5.0,          # seconds between samples (fixed 5 s cadence)
    seed: int = 0,
    normalize: bool = True,   # dataset-level per-channel z-normalization
    use_accel: bool = False   # if True, also return accel channel (Δv / dt)
):
    """
    Generate synthetic telematics trips with speed measured every 5s.
    Context 1 (road_type): 0=urban, 1=suburban, 2=rural
    Context 2 (weather):   0=dry,   1=precipitation

    Returns:
      X         : torch.float32 [N, T, C]  (C=1 if speed only, C=2 if use_accel)
      road_type : torch.int64   [N]
      weather   : torch.int64   [N]
      t         : torch.float32 [T] timestamps in seconds (0, 5, 10, ...)
    """
    rng = np.random.default_rng(seed)

    # Road-type priors (m/s): means roughly ~20 mph, 31 mph, 43 mph
    road_cfg = {
        0: dict(v_mean= 9.0, v_std=3.2, stop_rate=0.065),  # urban: slower, more stops
        1: dict(v_mean=14.0, v_std=2.6, stop_rate=0.030),  # suburban
        2: dict(v_mean=19.0, v_std=2.2, stop_rate=0.015),  # rural: faster, fewer stops
    }

    # Weather modifiers applied multiplicatively/additively
    # precipitation lowers speeds a bit, raises variability & stop frequency
    def apply_weather(cfg, w):
        if w == 0:  # dry
            return dict(**cfg, v_mult=1.00, std_mult=1.00, stop_mult=1.00,
                        micro_slow_p=0.02, micro_slow_drop=0.75, micro_slow_len=(2, 5))
        else:       # precipitation
            return dict(**cfg, v_mult=0.85, std_mult=1.15, stop_mult=1.30,
                        micro_slow_p=0.06, micro_slow_drop=0.65, micro_slow_len=(3, 8))

    # Sample labels
    road_type = rng.integers(0, 3, size=N, dtype=np.int64)
    weather   = rng.integers(0, 2, size=N, dtype=np.int64)

    # Allocate (speed) and optional accel
    C = 2 if use_accel else 1
    X = np.zeros((N, T, C), dtype=np.float32)

    # AR(1)-like speed generator with stop blocks & micro-slowdowns
    # Keeps batch shapes fixed (T), which plays nicely with contrastive crops.
    for i in range(N):
        r, w = int(road_type[i]), int(weather[i])
        base = road_cfg[r]
        cfg  = apply_weather(base, w)

        v_mean = cfg["v_mean"] * cfg["v_mult"]
        v_std  = cfg["v_std"]  * cfg["std_mult"]
        stop_rate = cfg["stop_rate"] * cfg["stop_mult"]

        # Target speed can drift slightly over the trip
        target = v_mean
        v = np.empty(T, dtype=np.float32)
        v[0] = rng.normal(target, v_std)

        # Micro-slowdown state (e.g., puddles / cautious patches under precip)
        slow_rem = 0
        slow_factor = 1.0

        # Red-light / stop-block planning
        # We mark a few blocks where speed ~0 for several samples
        stop_mask = np.zeros(T, dtype=bool)
        t0 = 1
        while t0 < T:
            if rng.random() < stop_rate:
                L = rng.integers(2, 7)  # 10–35 s at 5 s cadence
                stop_mask[t0:t0+L] = True
                t0 += L
            else:
                t0 += 1

        # Speed evolution
        alpha = 0.25  # how fast speed returns to target
        for t in range(1, T):
            # occasional slowdowns (not full stops)
            if slow_rem == 0 and rng.random() < cfg["micro_slow_p"]:
                Lmin, Lmax = cfg["micro_slow_len"]
                slow_rem = int(rng.integers(Lmin, Lmax + 1))
                slow_factor = cfg["micro_slow_drop"]  # drop target for a short duration

            effective_target = target * (slow_factor if slow_rem > 0 else 1.0)

            # small wandering of target over time (route/traffic drift)
            target += rng.normal(0.0, v_std * 0.02)

            # AR(1) pull toward target + noise
            v[t] = (1 - alpha) * v[t-1] + alpha * effective_target + rng.normal(0.0, v_std * 0.5)

            # Apply stop if in a stop block (clip toward 0 with a bit of noise)
            if stop_mask[t]:
                v[t] = max(0.0, rng.normal(0.2, 0.2))  # near-zero speed during stop

            # decay the slowdown
            if slow_rem > 0:
                slow_rem -= 1
                if slow_rem == 0:
                    slow_factor = 1.0

        v = np.clip(v, 0.0, None)

        if use_accel:
            a = np.empty_like(v)
            a[0] = 0.0
            a[1:] = (v[1:] - v[:-1]) / dt  # m/s^2
            X[i, :, 0] = v
            X[i, :, 1] = a
        else:
            X[i, :, 0] = v

    # Dataset-level z-norm per channel (common for contrastive setups)
    if normalize:
        mu = X.reshape(-1, C).mean(axis=0, keepdims=True)
        sd = X.reshape(-1, C).std(axis=0, keepdims=True) + 1e-6
        X = (X - mu) / sd

    t = np.arange(T, dtype=np.float32) * dt

    return (
        torch.from_numpy(X),                    # [N, T, C]
        torch.from_numpy(road_type),            # [N]
        torch.from_numpy(weather),              # [N]
        torch.from_numpy(t)                     # [T]
    )