import numpy as np
import pytest

from radar import (RadarConfig, Target, ca_cfar_2d, extract_detections,
                   range_doppler_map, simulate_beat_signal)

CFG = RadarConfig()


def process(targets, seed=0, pfa=1e-6):
    power = np.abs(range_doppler_map(simulate_beat_signal(CFG, targets, rng=seed))) ** 2
    mask, _, noise = ca_cfar_2d(power, pfa=pfa)
    return extract_detections(power, mask, noise, CFG)


def matches(dets, tgt):
    return [d for d in dets
            if abs(d.range_m - tgt.range_m) < CFG.range_bin
            and abs(d.velocity_mps - tgt.velocity_mps) < CFG.velocity_resolution]


def test_derived_parameters():
    assert CFG.range_resolution == pytest.approx(0.5996, rel=1e-3)
    assert CFG.range_bin == pytest.approx(CFG.range_resolution)  # n_samples = fs * chirp_time
    assert CFG.max_velocity == pytest.approx(62.46, rel=1e-3)
    assert CFG.velocity_resolution == pytest.approx(0.976, rel=1e-3)


@pytest.mark.parametrize("r, v", [(10.0, 0.0), (77.3, -31.0), (199.9, 44.4), (280.0, 5.5)])
def test_single_target_accuracy(r, v):
    tgt = Target(r, v, snr_db=-15.0)
    found = matches(process([tgt], seed=1), tgt)
    assert len(found) == 1
    # sub-bin interpolation: error well inside one resolution cell
    assert abs(found[0].range_m - r) < 0.35 * CFG.range_bin
    assert abs(found[0].velocity_mps - v) < 0.35 * CFG.velocity_resolution


def test_close_targets_are_resolved():
    a, b = Target(150.0, 25.0, -10.0), Target(151.5, 25.0, -13.0)
    dets = process([a, b], seed=7)
    assert matches(dets, a) and matches(dets, b)


def test_noise_only_false_alarm_rate():
    # Pfa 1e-6 over 65536 cells -> ~0.07 per frame at design; allow margin.
    false_alarms = sum(len(process([], seed=s)) for s in range(30))
    assert false_alarms / 30 < 0.5


def test_weak_target_below_detection_limit_is_missed():
    tgt = Target(120.0, 10.0, snr_db=-45.0)
    assert not matches(process([tgt], seed=3), tgt)
