import torch


def center_crop_to_multiple(volume: torch.Tensor, scale: int) -> torch.Tensor:
    D, C, H, W = volume.shape
    target_H = (H // scale) * scale
    target_W = (W // scale) * scale
    if target_H <= 0 or target_W <= 0:
        return volume
    if target_H == H and target_W == W:
        return volume
    y0 = (H - target_H) // 2
    x0 = (W - target_W) // 2
    return volume[:, :, y0:y0+target_H, x0:x0+target_W]


