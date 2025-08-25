# load_ct_tensor.py

import pydicom
import numpy as np
import torch
from window_presets import WINDOW_PRESETS
from pydicom.pixel_data_handlers.util import apply_modality_lut

def load_ct_as_tensor(dicom_path, preset="soft_tissue", normalization='global', hu_clip=(-1000, 2000)):
    # DICOM laden und in HU umrechnen (Modality LUT)
    ds = pydicom.dcmread(dicom_path, force=True)
    arr = ds.pixel_array
    try:
        hu = apply_modality_lut(arr, ds).astype(np.float32)
    except Exception:
        hu = arr.astype(np.float32)

    if normalization == 'window':
        window = WINDOW_PRESETS.get(preset, WINDOW_PRESETS["default"])
        wl, ww = window["center"], window["width"]
        min_val = wl - ww / 2
        max_val = wl + ww / 2
        img = np.clip(hu, min_val, max_val)
        img = (img - min_val) / (max_val - min_val)
        img = img * 2 - 1
    else:
        lo, hi = hu_clip
        img = np.clip(hu, lo, hi)
        img = (img - lo) / (hi - lo)
        img = img * 2 - 1

    tensor = torch.tensor(img).unsqueeze(0)  # [1, H, W]
    return tensor
