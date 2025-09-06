import torch
import torch.nn.functional as F
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import math
import warnings

# Optional metrics: LPIPS (full-reference) and Perceptual Index (PI = 0.5*((10-Ma)+NIQE))
_lpips_model = None
_pyiqa_lpips = None
_piqa_niqe = None
_piqa_ma = None
_warned_lpips = False
_warned_pi = False
try:
    import lpips as _lpips
except Exception:
    _lpips = None
try:
    import pyiqa as _pyiqa
except Exception:
    _pyiqa = None
try:
    from skimage.metrics import niqe as _skimage_niqe
except Exception:
    _skimage_niqe = None

def upsample_interpolation(lr_tensor, target_size, method="bilinear"):
    mode = {
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC
    }.get(method, cv2.INTER_LINEAR)

    img_np = lr_tensor.squeeze(0).cpu().numpy()  # [H, W]
    img_up = cv2.resize(img_np, dsize=(target_size[1], target_size[0]), interpolation=mode)
    return torch.tensor(img_up).unsqueeze(0)

def _ensure_lpips_model():
    """Prefer pyiqa's LPIPS (newer API), fallback to lpips package."""
    global _pyiqa_lpips, _lpips_model, _warned_lpips
    if _pyiqa is not None:
        if _pyiqa_lpips is None:
            try:
                _pyiqa_lpips = _pyiqa.create_metric('lpips')
                _pyiqa_lpips.eval()
            except Exception:
                _pyiqa_lpips = None
        if _pyiqa_lpips is not None:
            return _pyiqa_lpips
    # fallback to original lpips
    if _lpips_model is not None:
        return _lpips_model
    if _lpips is None:
        if not _warned_lpips:
            print("[Metrics] LPIPS not available (pip install pyiqa or lpips). LPIPS will be NaN.")
            _warned_lpips = True
        return None
    try:
        _lpips_model = _lpips.LPIPS(net='alex')
        _lpips_model.eval()
    except Exception:
        _lpips_model = None
        if not _warned_lpips:
            print("[Metrics] Failed to initialize LPIPS(alex). LPIPS will be NaN.")
            _warned_lpips = True
    return _lpips_model

def _compute_lpips(sr_tensor, hr_tensor):
    # Inputs: [1,H,W] in [-1,1]
    model = _ensure_lpips_model()
    if model is None:
        return float('nan')
    try:
        with torch.no_grad():
            if _pyiqa is not None and model is _pyiqa_lpips:
                # pyiqa expects [B,C,H,W] in [0,1] (it will internally normalize)
                def to01_3(x):
                    x01 = (x + 1.0) * 0.5
                    x3 = x01.unsqueeze(0).repeat(1,3,1,1)  # [1,3,H,W]
                    return x3
                d = model(to01_3(sr_tensor), to01_3(hr_tensor))
            else:
                # lpips package expects [-1,1] 3ch
                def to_m11_3(x):
                    x3 = x.clone().repeat(3, 1, 1)
                    return x3.unsqueeze(0)
                d = model(to_m11_3(sr_tensor), to_m11_3(hr_tensor))
        return float(d.item())
    except Exception:
        return float('nan')

def _ensure_pi_metrics():
    global _piqa_niqe, _piqa_ma, _warned_pi
    if _piqa_niqe is not None and _piqa_ma is not None:
        return _piqa_niqe, _piqa_ma
    if _pyiqa is not None:
        try:
            _piqa_niqe = _pyiqa.create_metric('niqe')
            _piqa_ma = _pyiqa.create_metric('nrqm')  # Ma-Score in pyiqa heißt 'nrqm'
        except Exception:
            _piqa_niqe = None
            _piqa_ma = None
    if (_piqa_niqe is None or _piqa_ma is None) and not _warned_pi:
        msg = "[Metrics] PI requires NIQE and Ma."
        if _pyiqa is None:
            msg += " Install pyiqa for NIQE/Ma (pip install pyiqa)."
        else:
            msg += " pyiqa not fully available; PI will be NaN."
        if _skimage_niqe is None:
            msg += " skimage.niqe also not available."
        print(msg)
        _warned_pi = True
    return _piqa_niqe, _piqa_ma

def _compute_pi(grayscale_tensor_m1_1):
    # PI = 0.5*((10 - Ma) + NIQE). Compute on the reconstructed image only.
    # Input: [1,H,W] in [-1,1]
    # Convert to [0,1] and 3ch for Ma; NIQE often uses grayscale or color [0,1]
    img01 = ((grayscale_tensor_m1_1.detach().cpu().numpy() + 1.0) / 2.0).clip(0.0, 1.0)
    niqe_val = float('nan')
    ma_val = float('nan')
    # Try pyiqa first (resize to typical backbone input to avoid NaNs)
    piqa_niqe, piqa_ma = _ensure_pi_metrics()
    try:
        if piqa_niqe is not None:
            # NIQE: compute on native grayscale size [B,1,H,W] in [0,1]
            x_gray = torch.tensor(img01, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                niqe_val = float(piqa_niqe(x_gray).item())
        elif _skimage_niqe is not None:
            niqe_val = float(_skimage_niqe(img01.astype(np.float64)))
    except Exception:
        pass
    try:
        if piqa_ma is not None:
            x = torch.tensor(img01, dtype=torch.float32).unsqueeze(0)
            x3 = x.repeat(1,3,1,1)
            with torch.no_grad():
                x3 = F.interpolate(x3, size=(224, 224), mode='bilinear', align_corners=False)
                ma_val = float(piqa_ma(x3).item())
    except Exception:
        # When 'ma' not present in this pyiqa build, leave as NaN so PI becomes NaN (not NIQE)
        pass
    if math.isfinite(niqe_val) and math.isfinite(ma_val):
        return float(0.5 * ((10.0 - ma_val) + niqe_val))
    return float('nan')

def evaluate_metrics(sr_tensor, hr_tensor):
    sr_np = sr_tensor.squeeze().cpu().numpy()
    hr_np = hr_tensor.squeeze().cpu().numpy()

    # Rescale from [-1,1] to [0,1] for metrics; otherwise SSIM and my own PSNR calculation does not work correctly
    sr_np = ((sr_np + 1) / 2).clip(0, 1)
    hr_np = ((hr_np + 1) / 2).clip(0, 1)

    # Robust MSE/PSNR without warnings for perfect matches
    diff = hr_np.astype(np.float64) - sr_np.astype(np.float64)
    mse_val = float(np.mean(diff * diff))
    rmse_val = float(math.sqrt(mse_val))
    mae_val = float(np.mean(np.abs(diff)))  
    
    if mse_val <= 0.0:
        psnr_val = float('inf')
    else:
        psnr_val = float(10.0 * math.log10(1.0 / mse_val))
    # Choose an adaptive odd win_size <= min(H,W) to avoid errors on small images
    # Support [H,W] or [1,H,W]
    if hr_np.ndim == 3 and hr_np.shape[0] == 1:
        hr_np = hr_np[0]
        sr_np = sr_np[0]
    h, w = hr_np.shape
    min_side = max(1, min(h, w))
    # cap at 11 (common default upper bound); ensure odd and >=3
    ws = min(11, min_side)
    if ws % 2 == 0:
        ws = max(3, ws - 1)
    ws = max(3, ws)
    try:
        ssim_val = float(ssim(hr_np, sr_np, data_range=1.0, win_size=ws))
    except Exception:
        # Fallback: if still failing, return 0.0 to keep evaluation running
        ssim_val = 0.0

    # LPIPS on [-1,1] 3ch
    try:
        lpips_val = _compute_lpips(sr_tensor, hr_tensor)
    except Exception:
        lpips_val = float('nan')

    # PI on SR only
    try:
        pi_val = _compute_pi(sr_tensor)
    except Exception:
        pi_val = float('nan')

    return {
        "MSE": mse_val,
        "RMSE": rmse_val,
        "MAE": mae_val,  
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "LPIPS": lpips_val,
        "PI": pi_val
    }

def compare_methods(lr_tensor, hr_tensor, model):
    model.eval()
    with torch.no_grad():
        # Ensure model input is on the same device as model
        device = next(model.parameters()).device
        lr_batched = lr_tensor.unsqueeze(0).to(device)
        sr_model = model(lr_batched).squeeze(0).cpu()

        # Interpolation Upscaling on CPU tensors
        target_size = hr_tensor.shape[1:]  # [H, W]
        sr_linear = upsample_interpolation(lr_tensor, target_size, method="bilinear")
        sr_cubic = upsample_interpolation(lr_tensor, target_size, method="bicubic")

        # Metrics berechnen
        results = {
            "Model (RRDB)": evaluate_metrics(sr_model, hr_tensor),
            "Interpolation (Linear)": evaluate_metrics(sr_linear, hr_tensor),
            "Interpolation (Bicubic)": evaluate_metrics(sr_cubic, hr_tensor)
        }

        return results

# Beispiel:
if __name__ == "__main__":
    # Dummydaten
    hr = torch.rand(1, 512, 512) * 2 - 1  # [-1, 1]
    lr = F.interpolate(hr.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)

    from rrdb_ct_model import RRDBNet_CT
    model = RRDBNet_CT()

    # Vergleich durchführen
    result = compare_methods(lr, hr, model)
    for method, metrics in result.items():
        print(f"{method}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
