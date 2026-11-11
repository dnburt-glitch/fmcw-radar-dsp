"""Monte Carlo performance check: probability of detection vs raw SNR,
and the measured false-alarm rate against the CFAR design value.

    python monte_carlo.py         # ~1 min, saves docs/04_pd_vs_snr.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from radar import (RadarConfig, Target, ca_cfar_2d, extract_detections,
                   range_doppler_map, simulate_beat_signal)

DOCS = Path(__file__).parent / "docs"
TRIALS = 100
SNRS_DB = np.arange(-42, -25, 1.0)
PFAS = (1e-4, 1e-6, 1e-8)


def detected(dets, tgt, cfg) -> bool:
    return any(abs(d.range_m - tgt.range_m) < 1.5 * cfg.range_bin
               and abs(d.velocity_mps - tgt.velocity_mps) < 1.5 * cfg.velocity_resolution
               for d in dets)


def main(seed: int = 0) -> None:
    cfg = RadarConfig()
    rng = np.random.default_rng(seed)
    pd = {pfa: [] for pfa in PFAS}
    false_alarms = {pfa: 0 for pfa in PFAS}

    for snr in SNRS_DB:
        hits = {pfa: 0 for pfa in PFAS}
        for _ in range(TRIALS):
            tgt = Target(range_m=rng.uniform(20, 280), velocity_mps=rng.uniform(-50, 50), snr_db=snr)
            power = np.abs(range_doppler_map(simulate_beat_signal(cfg, [tgt], rng=rng))) ** 2
            for pfa in PFAS:
                mask, _, noise = ca_cfar_2d(power, pfa=pfa)
                dets = extract_detections(power, mask, noise, cfg)
                ok = detected(dets, tgt, cfg)
                hits[pfa] += ok
                false_alarms[pfa] += len(dets) - int(ok)
        for pfa in PFAS:
            pd[pfa].append(hits[pfa] / TRIALS)
        print(f"SNR {snr:6.1f} dB: " + "  ".join(f"Pd@{p:.0e}={pd[p][-1]:.2f}" for p in PFAS))

    frames = TRIALS * len(SNRS_DB)
    cells = cfg.n_chirps * cfg.n_samples
    print("\nFalse alarms per frame (peaks, measured):")
    for pfa in PFAS:
        print(f"  Pfa design {pfa:.0e}: {false_alarms[pfa] / frames:.3f}"
              f"  (cell-level design value x cells = {pfa * cells:.3f})")

    gain_db = 10 * np.log10(cells)
    DOCS.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for pfa in PFAS:
        ax.plot(SNRS_DB, pd[pfa], marker="o", ms=3, label=f"CA-CFAR, Pfa = {pfa:.0e}")
    ax.set(title=f"Probability of detection vs per-sample SNR ({TRIALS} trials per point, "
                 f"2D FFT gain = {gain_db:.1f} dB)",
           xlabel="Per-sample SNR before processing, dB", ylabel="Pd", ylim=(-0.02, 1.02))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS / "04_pd_vs_snr.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved {DOCS / '04_pd_vs_snr.png'}")


if __name__ == "__main__":
    main()
