import torch
import torch.nn.functional as F
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import math

def upsample_interpolation(lr_tensor, target_size, method="bilinear"):
    mode = {
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC
    }.get(method, cv2.INTER_LINEAR)

    img_np = lr_tensor.squeeze(0).cpu().numpy()  # [H, W]
    img_up = cv2.resize(img_np, dsize=(target_size[1], target_size[0]), interpolation=mode)
    return torch.tensor(img_up).unsqueeze(0)

def evaluate_metrics(sr_tensor, hr_tensor):
    sr_np = sr_tensor.squeeze().cpu().numpy()
    hr_np = hr_tensor.squeeze().cpu().numpy()

    # Rescale from [-1,1] to [0,1] for metrics
    sr_np = ((sr_np + 1) / 2).clip(0, 1)
    hr_np = ((hr_np + 1) / 2).clip(0, 1)

    # Robust MSE/PSNR without warnings for perfect matches
    diff = hr_np.astype(np.float64) - sr_np.astype(np.float64)
    mse_val = float(np.mean(diff * diff))
    rmse_val = float(math.sqrt(mse_val))
    if mse_val <= 0.0:
        psnr_val = float('inf')
    else:
        psnr_val = float(10.0 * math.log10(1.0 / mse_val))
    ssim_val = float(ssim(hr_np, sr_np, data_range=1.0))

    return {
        "MSE": mse_val,
        "RMSE": rmse_val,
        "PSNR": psnr_val,
        "SSIM": ssim_val
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
