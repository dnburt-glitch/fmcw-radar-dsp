"""Radar and scenario configuration."""
from __future__ import annotations

from dataclasses import dataclass

C = 299_792_458.0  # speed of light, m/s


@dataclass(frozen=True)
class RadarConfig:
    """FMCW radar parameters (one frame = n_chirps sawtooth chirps).

    Defaults describe a generic 24 GHz short-range sensor with complex (IQ)
    sampling, so the full sampled band maps to positive ranges.
    """

    fc: float = 24.0e9          # carrier frequency, Hz
    bandwidth: float = 250e6    # chirp sweep bandwidth, Hz
    chirp_time: float = 50e-6   # chirp duration (= chirp repetition interval), s
    fs: float = 10.24e6         # ADC sample rate (complex), Hz
    n_samples: int = 512        # fast-time samples per chirp
    n_chirps: int = 128         # slow-time chirps per frame

    @property
    def slope(self) -> float:
        """Chirp slope, Hz/s."""
        return self.bandwidth / self.chirp_time

    @property
    def wavelength(self) -> float:
        return C / self.fc

    @property
    def range_resolution(self) -> float:
        return C / (2.0 * self.bandwidth)

    @property
    def range_bin(self) -> float:
        """Range spacing of one FFT bin, m."""
        return C * self.fs / (2.0 * self.slope * self.n_samples)

    @property
    def max_range(self) -> float:
        return self.range_bin * self.n_samples

    @property
    def velocity_resolution(self) -> float:
        return self.wavelength / (2.0 * self.n_chirps * self.chirp_time)

    @property
    def max_velocity(self) -> float:
        """Unambiguous radial velocity, +/- m/s."""
        return self.wavelength / (4.0 * self.chirp_time)


@dataclass(frozen=True)
class Target:
    """Point target. Positive velocity = moving away from the radar."""

    range_m: float
    velocity_mps: float
    snr_db: float  # per-sample SNR before any processing gain
