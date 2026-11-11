"""Range-Doppler processing and detection extraction."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .config import RadarConfig


def range_doppler_map(data: np.ndarray) -> np.ndarray:
    """2D FFT with Hann windows. Returns complex map (doppler, range),
    zero velocity centred on the Doppler axis."""
    n_chirps, n_samples = data.shape
    win_r = np.hanning(n_samples)[None, :]
    win_d = np.hanning(n_chirps)[:, None]
    range_fft = np.fft.fft(data * win_r, axis=1)
    rd = np.fft.fft(range_fft * win_d, axis=0)
    return np.fft.fftshift(rd, axes=0)


def axes(cfg: RadarConfig) -> tuple[np.ndarray, np.ndarray]:
    """Range axis (m) and velocity axis (m/s) matching range_doppler_map."""
    range_axis = np.arange(cfg.n_samples) * cfg.range_bin
    velocity_axis = (np.arange(cfg.n_chirps) - cfg.n_chirps // 2) * cfg.velocity_resolution
    return range_axis, velocity_axis


@dataclass(frozen=True)
class Detection:
    range_m: float
    velocity_mps: float
    power_db: float
    snr_db: float  # peak power over the local CFAR noise estimate


def _parabolic_offset(left: float, centre: float, right: float) -> float:
    """Sub-bin peak offset from three log-power samples."""
    denom = left - 2.0 * centre + right
    if denom == 0.0:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denom, -0.5, 0.5))


def extract_detections(
    power: np.ndarray,
    mask: np.ndarray,
    noise: np.ndarray,
    cfg: RadarConfig,
) -> list[Detection]:
    """Keep CFAR hits that are local maxima of their 3x3 neighbourhood (so two
    targets in adjacent cells stay separate instead of merging into one
    cluster), then refine each peak with parabolic interpolation in both
    dimensions."""
    log_p = 10.0 * np.log10(power + 1e-30)
    local_max = power == ndimage.maximum_filter(power, size=3, mode="wrap")
    peaks = np.argwhere(mask & local_max)
    n_dop, n_rng = power.shape
    range_axis, velocity_axis = axes(cfg)

    detections = []
    for d, r in peaks:

        dr = 0.0
        if 0 < r < n_rng - 1:
            dr = _parabolic_offset(log_p[d, r - 1], log_p[d, r], log_p[d, r + 1])
        dd = _parabolic_offset(log_p[(d - 1) % n_dop, r], log_p[d, r],
                               log_p[(d + 1) % n_dop, r])

        velocity = float(velocity_axis[d] + dd * cfg.velocity_resolution)
        # Range-Doppler coupling: the Doppler shift adds to the beat frequency,
        # biasing range by v * fc / slope (0.2 m at 44 m/s here). Remove it.
        coupling = velocity * cfg.fc / cfg.slope
        detections.append(Detection(
            range_m=float(range_axis[r] + dr * cfg.range_bin - coupling),
            velocity_mps=velocity,
            power_db=float(log_p[d, r]),
            snr_db=float(log_p[d, r] - 10.0 * np.log10(noise[d, r])),
        ))
    return sorted(detections, key=lambda det: det.range_m)
