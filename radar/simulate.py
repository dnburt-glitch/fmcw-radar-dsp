"""Synthetic FMCW beat-signal generator."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .config import C, RadarConfig, Target


def simulate_beat_signal(
    cfg: RadarConfig,
    targets: Iterable[Target],
    noise_power: float = 1.0,
    rng: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Return a dechirped IQ data cube of shape (n_chirps, n_samples).

    The model is the exact mixer output tx * conj(rx) for a sawtooth chirp,
    including range migration within and across chirps:

        phase = 2*pi * (fc*tau + k*tau*t - k*tau**2 / 2),  tau = 2*R(t)/c
    """
    rng = np.random.default_rng(rng)
    t_fast = np.arange(cfg.n_samples) / cfg.fs                  # (N,)
    t_slow = np.arange(cfg.n_chirps)[:, None] * cfg.chirp_time  # (M, 1)

    data = np.zeros((cfg.n_chirps, cfg.n_samples), dtype=np.complex128)
    for tgt in targets:
        amplitude = np.sqrt(noise_power * 10.0 ** (tgt.snr_db / 10.0))
        r = tgt.range_m + tgt.velocity_mps * (t_slow + t_fast)
        tau = 2.0 * r / C
        phase = 2.0 * np.pi * (cfg.fc * tau + cfg.slope * tau * t_fast
                               - 0.5 * cfg.slope * tau**2)
        data += amplitude * np.exp(1j * phase)

    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(data.shape) + 1j * rng.standard_normal(data.shape)
    )
    return data + noise
