import numpy as np
import torch
import torch.nn.functional as F


def _gaussian_kernel_2d(sigma: float, kernel_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    half = (k - 1) // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    g1 = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    g1 = g1 / g1.sum()
    kernel = torch.outer(g1, g1)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, k, k)


def _kernel_size_from_sigma(sigma: float) -> int:
    k = int(max(3, round(6.0 * float(sigma))))
    if k % 2 == 0:
        k += 1
    return k


def degrade_hr_to_lr(hr_volume: torch.Tensor, scale: int, *, degradation: str = 'blurnoise', blur_sigma_range=None,
                     blur_kernel: int = None, noise_sigma_range_norm=(0.001, 0.003), dose_factor_range=(0.25, 0.5),
                     antialias_clean: bool = True, rng=None):
    device = hr_volume.device
    dtype = hr_volume.dtype
    used = {'blur_sigma': None, 'blur_kernel_k': None, 'noise_sigma': None, 'dose': None}
    if degradation in ('blur', 'blurnoise'):
        if blur_sigma_range is None:
            base_sigma = 0.8 if scale == 2 else (1.2 if scale == 4 else 0.8)
            jitter = 0.1 if scale == 2 else 0.15
            sig_lo, sig_hi = max(1e-6, base_sigma - jitter), base_sigma + jitter
        else:
            sig_lo, sig_hi = float(blur_sigma_range[0]), float(blur_sigma_range[1])
        rng = np.random.default_rng() if rng is None else rng
        sigma = float(rng.uniform(sig_lo, sig_hi))
        k = blur_kernel if blur_kernel is not None else _kernel_size_from_sigma(sigma)
        kernel = _gaussian_kernel_2d(max(1e-6, sigma), k, device, dtype)
        used['blur_sigma'] = float(sigma)
        used['blur_kernel_k'] = int(k)
        pad = (k // 2, k // 2, k // 2, k // 2)
        x = F.pad(hr_volume, pad, mode='reflect')
        hr_blur = F.conv2d(x, kernel)
    else:
        hr_blur = hr_volume

    if degradation == 'clean':
        lr = F.interpolate(hr_blur, scale_factor=(1.0/scale, 1.0/scale), mode='bilinear', align_corners=False, antialias=antialias_clean)
    else:
        lr = F.interpolate(hr_blur, scale_factor=(1.0/scale, 1.0/scale), mode='bilinear', align_corners=False, antialias=False)

    if degradation == 'blurnoise':
        n_lo, n_hi = float(noise_sigma_range_norm[0]), float(noise_sigma_range_norm[1])
        d_lo, d_hi = float(dose_factor_range[0]), float(dose_factor_range[1])
        rng = np.random.default_rng() if rng is None else rng
        noise_sigma = float(rng.uniform(n_lo, n_hi))
        dose = float(rng.uniform(min(d_lo, d_hi), max(d_lo, d_hi)))
        noise_eff = noise_sigma / max(1e-6, dose) ** 0.5
        noise_t = torch.randn_like(lr, device=lr.device, dtype=lr.dtype) * noise_eff
        lr = torch.clamp(lr + noise_t, -1.0, 1.0)
        used['noise_sigma'] = float(noise_sigma)
        used['dose'] = float(dose)
    return lr, used


def build_lr_volume_from_hr(hr_volume, scale=2, *, degradation='blurnoise', blur_sigma_range=None, blur_kernel=None,
						 noise_sigma_range_norm=(0.001, 0.003), dose_factor_range=(0.25, 0.5), antialias_clean=True, rng=None):
	return degrade_hr_to_lr(hr_volume, scale,
		degradation=degradation,
		blur_sigma_range=blur_sigma_range,
		blur_kernel=blur_kernel,
		noise_sigma_range_norm=noise_sigma_range_norm,
		dose_factor_range=dose_factor_range,
		antialias_clean=antialias_clean,
		rng=rng)


