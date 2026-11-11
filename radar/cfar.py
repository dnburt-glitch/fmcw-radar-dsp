"""Two-dimensional cell-averaging CFAR."""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def ca_cfar_2d(
    power: np.ndarray,
    guard: tuple[int, int] = (2, 2),
    train: tuple[int, int] = (4, 8),
    pfa: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CA-CFAR on a square-law (linear power) range-Doppler map.

    guard / train are half-widths as (doppler, range) cells. The Doppler axis
    wraps (it is periodic); the range axis is mirrored at the edges.

    Returns (detection_mask, threshold, noise_estimate), all shaped like power.
    """
    gd, gr = guard
    td, tr = train
    hd, hr = gd + td, gr + tr

    kernel = np.ones((2 * hd + 1, 2 * hr + 1))
    kernel[td:td + 2 * gd + 1, tr:tr + 2 * gr + 1] = 0.0
    n_train = kernel.sum()

    padded = np.pad(power, ((hd, hd), (0, 0)), mode="wrap")
    padded = np.pad(padded, ((0, 0), (hr, hr)), mode="symmetric")
    noise = fftconvolve(padded, kernel, mode="valid") / n_train
    noise = np.maximum(noise, np.finfo(float).tiny)

    # Scaling factor for exponentially distributed (square-law) noise.
    alpha = n_train * (pfa ** (-1.0 / n_train) - 1.0)
    threshold = alpha * noise
    return power > threshold, threshold, noise
