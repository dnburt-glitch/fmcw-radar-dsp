"""End-to-end demo: simulate a frame, process it, detect targets, plot results.

    python run_demo.py            # saves figures to docs/
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from radar import (RadarConfig, Target, axes, ca_cfar_2d, extract_detections,
                   range_doppler_map, simulate_beat_signal)

DOCS = Path(__file__).parent / "docs"

SCENARIO = [
    Target(range_m=42.0, velocity_mps=12.0, snr_db=-5.0),
    Target(range_m=95.0, velocity_mps=-7.5, snr_db=-12.0),
    Target(range_m=150.0, velocity_mps=25.0, snr_db=-10.0),
    Target(range_m=151.5, velocity_mps=25.0, snr_db=-13.0),   # close pair: 2.5 resolution cells
    Target(range_m=230.0, velocity_mps=-18.0, snr_db=-22.0),  # weak, far
]


def run(seed: int = 7):
    cfg = RadarConfig()
    data = simulate_beat_signal(cfg, SCENARIO, rng=seed)
    rd = range_doppler_map(data)
    power = np.abs(rd) ** 2
    mask, threshold, noise = ca_cfar_2d(power)
    detections = extract_detections(power, mask, noise, cfg)
    return cfg, data, power, mask, threshold, detections


def print_report(cfg: RadarConfig, detections) -> None:
    print("Radar: fc={:.1f} GHz, B={:.0f} MHz, {} chirps x {} samples".format(
        cfg.fc / 1e9, cfg.bandwidth / 1e6, cfg.n_chirps, cfg.n_samples))
    print("Range res {:.2f} m (max {:.0f} m) | velocity res {:.2f} m/s (max +/-{:.1f} m/s)\n".format(
        cfg.range_resolution, cfg.max_range, cfg.velocity_resolution, cfg.max_velocity))
    print(f"{'truth R':>8} {'truth v':>8} {'raw SNR':>8} | {'est R':>8} {'est v':>8} {'det SNR':>8} {'dR':>6} {'dv':>6}")
    for tgt in SCENARIO:
        best = min(detections, key=lambda d: abs(d.range_m - tgt.range_m) / cfg.range_bin
                   + abs(d.velocity_mps - tgt.velocity_mps) / cfg.velocity_resolution)
        print(f"{tgt.range_m:8.2f} {tgt.velocity_mps:8.2f} {tgt.snr_db:7.0f}dB | "
              f"{best.range_m:8.2f} {best.velocity_mps:8.2f} {best.snr_db:6.1f}dB "
              f"{best.range_m - tgt.range_m:6.2f} {best.velocity_mps - tgt.velocity_mps:6.2f}")
    print(f"\nTotal detections: {len(detections)} (targets: {len(SCENARIO)})")


def plot(cfg, data, power, mask, threshold, detections) -> None:
    DOCS.mkdir(exist_ok=True)
    rng_ax, vel_ax = axes(cfg)
    p_db = 10 * np.log10(power)
    p_db -= np.median(p_db)  # noise floor ~ 0 dB

    # 1. Raw beat signal vs processed range profile
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.5))
    t_us = np.arange(cfg.n_samples) / cfg.fs * 1e6
    a1.plot(t_us, data[0].real, lw=0.6, color="tab:gray")
    a1.set(title="Raw beat signal, chirp 0 (I channel): targets are below the noise",
           xlabel="Fast time, µs", ylabel="Amplitude")
    profile = p_db.max(axis=0)
    a2.plot(rng_ax, profile, lw=0.9)
    for tgt in SCENARIO:
        a2.axvline(tgt.range_m, color="tab:red", ls="--", lw=0.7, alpha=0.6)
    a2.set(title="Range profile after 2D FFT (max over Doppler); dashed = true ranges",
           xlabel="Range, m", ylabel="dB above median noise", xlim=(0, cfg.max_range))
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(DOCS / "01_range_profile.png", dpi=150)
    plt.close(fig)

    # 2. Range-Doppler map with detections
    fig, ax = plt.subplots(figsize=(10, 5.5))
    im = ax.imshow(p_db, aspect="auto", origin="lower", cmap="viridis", vmin=-5, vmax=45,
                   extent=[rng_ax[0], rng_ax[-1] + cfg.range_bin,
                           vel_ax[0], vel_ax[-1] + cfg.velocity_resolution])
    ax.scatter([d.range_m for d in detections], [d.velocity_mps for d in detections],
               s=120, facecolors="none", edgecolors="red", lw=1.4, label="CFAR detections")
    ax.scatter([t.range_m for t in SCENARIO], [t.velocity_mps for t in SCENARIO],
               marker="x", c="white", s=30, lw=1, label="Ground truth")
    ax.set(title="Range-Doppler map with 2D CA-CFAR detections",
           xlabel="Range, m", ylabel="Radial velocity, m/s")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="dB above noise floor")
    fig.tight_layout()
    fig.savefig(DOCS / "02_range_doppler_map.png", dpi=150)
    plt.close(fig)

    # 3. CFAR threshold cut through the close target pair
    d_idx = int(np.argmin(np.abs(vel_ax - 25.0)))
    offset = np.median(10 * np.log10(power))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rng_ax, 10 * np.log10(power[d_idx]) - offset, lw=0.9, label="Cell power")
    ax.plot(rng_ax, 10 * np.log10(threshold[d_idx]) - offset, lw=1.2, color="tab:orange",
            label="CA-CFAR threshold (Pfa = 1e-6)")
    hits = np.where(mask[d_idx])[0]
    ax.scatter(rng_ax[hits], 10 * np.log10(power[d_idx, hits]) - offset, color="red", s=12,
               zorder=3, label="Cells above threshold")
    ax.set(title=f"Range cut at v = {vel_ax[d_idx]:.1f} m/s: two targets 1.5 m apart are resolved",
           xlabel="Range, m", ylabel="dB above noise floor", xlim=(120, 180))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "03_cfar_cut.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    results = run()
    print_report(results[0], results[-1])
    plot(*results)
    print(f"Figures saved to {DOCS}/")
