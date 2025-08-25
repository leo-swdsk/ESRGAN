# downsample_tensor_volume.py

import torch
import torch.nn.functional as F

def downsample_tensor(tensor, scale_factor=2):
    """
    Funktioniert für:
    - Einzelne Slices: [1, H, W]
    - Volumen: [N, 1, H, W]
    """
    if tensor.ndim == 3:
        # Einzelbild: [1, H, W] → [1, 1, H, W]
        tensor = tensor.unsqueeze(0)
        downsampled = F.interpolate(tensor, scale_factor=1/scale_factor, mode='bilinear', align_corners=False, antialias=True)
        return downsampled.squeeze(0)  # zurück zu [1, H', W']

    elif tensor.ndim == 4:
        # Volumen: [N, 1, H, W]
        downsampled = F.interpolate(tensor, scale_factor=1/scale_factor, mode='bilinear', align_corners=False, antialias=True)
        return downsampled  # [N, 1, H', W']

    else:
        raise ValueError("Tensor muss [1,H,W] oder [N,1,H,W] sein")
