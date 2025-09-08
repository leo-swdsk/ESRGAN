import torch
from .presets import WINDOW_PRESETS
from .types import WindowCfg


def hu_to_m11_window(x: torch.Tensor, wl: float, ww: float) -> torch.Tensor:
    x = x.to(torch.float32)
    ww = float(ww)
    wl = float(wl)
    if ww <= 0:
        raise ValueError(f"Window Width must be > 0, got {ww}")
    min_val = wl - ww / 2.0
    max_val = wl + ww / 2.0
    x = x.clamp(min_val, max_val)
    x01 = (x - min_val) / (max_val - min_val)
    return x01 * 2.0 - 1.0


def get_preset(name: str) -> WindowCfg:
    cfg = WINDOW_PRESETS.get(name, WINDOW_PRESETS['default'])
    return WindowCfg(center=float(cfg['center']), width=float(cfg['width']))


