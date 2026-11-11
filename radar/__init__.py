"""FMCW radar signal-processing pipeline: simulation, range-Doppler, CA-CFAR."""
from .cfar import ca_cfar_2d
from .config import RadarConfig, Target
from .processing import Detection, axes, extract_detections, range_doppler_map
from .simulate import simulate_beat_signal

__all__ = [
    "RadarConfig", "Target", "Detection",
    "simulate_beat_signal", "range_doppler_map", "axes",
    "ca_cfar_2d", "extract_detections",
]
