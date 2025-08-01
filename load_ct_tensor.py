# load_ct_tensor.py

import pydicom
import numpy as np
import torch
from window_presets import WINDOW_PRESETS

def load_ct_as_tensor(dicom_path, preset="soft_tissue"):
    # Fensterwerte laden
    window = WINDOW_PRESETS.get(preset, WINDOW_PRESETS["default"])
    wl, ww = window["center"], window["width"]

    # DICOM laden
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(np.float32)

    # Fensterung anwenden
    min_val = wl - ww / 2
    max_val = wl + ww / 2
    img = np.clip(img, min_val, max_val)

    # Normalisierung auf [-1, 1]
    img = (img - min_val) / (max_val - min_val)  # [0, 1]
    img = img * 2 - 1                             # [-1, 1]

    # Als Tensor [1, H, W] zurückgeben (1 Kanal)
    tensor = torch.tensor(img).unsqueeze(0)  # Shape: [1, H, W]
    return tensor
