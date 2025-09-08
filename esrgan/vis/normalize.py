import torch


def hu_to_m11_global(x: torch.Tensor, lo: float = -1000.0, hi: float = 2000.0) -> torch.Tensor:
    x = x.to(torch.float32).clamp(lo, hi)
    x01 = (x - lo) / (hi - lo)
    return x01 * 2.0 - 1.0


def m11_to_hu_global(x: torch.Tensor, lo: float = -1000.0, hi: float = 2000.0) -> torch.Tensor:
    x = x.to(torch.float32)
    x01 = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
    return x01 * (hi - lo) + lo


