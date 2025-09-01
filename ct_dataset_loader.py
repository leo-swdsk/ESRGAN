import os
import torch
import pydicom
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
from pydicom.pixel_data_handlers.util import apply_modality_lut
#WGanze Slices laden bringt meistens kaum einen Vorteil und füllt den Speicher unnötig, deshalb kleinere zufällige Patches
def random_aligned_crop(hr_tensor, lr_tensor, hr_patch=128, scale=2):
    # hr_tensor: [1, H, W], lr_tensor: [1, H/2, W/2] bei scale=2
    _, H, W = hr_tensor.shape
    assert hr_patch % scale == 0, "hr_patch muss Vielfaches von scale sein"
    lr_patch = hr_patch // scale

    # sichere Grenzen
    max_y_hr = H - hr_patch
    max_x_hr = W - hr_patch
    if max_y_hr < 0 or max_x_hr < 0:
        # Falls das Bild kleiner als der Patch ist: auf ganze Slice zurückfallen
        return lr_tensor, hr_tensor

    # zufällige, scale-ausgerichtete Startpunkte
    y_hr = random.randint(0, max_y_hr)
    x_hr = random.randint(0, max_x_hr)
    # LR-Koordinaten entsprechend skaliert
    y_lr = y_hr // scale
    x_lr = x_hr // scale

    hr_crop = hr_tensor[:, y_hr:y_hr+hr_patch, x_hr:x_hr+hr_patch]
    lr_crop = lr_tensor[:, y_lr:y_lr+lr_patch, x_lr:x_lr+lr_patch]
    return lr_crop, hr_crop


def apply_window(img, center, width):
    min_val = center - width / 2
    max_val = center + width / 2
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val)  # [0,1]
    img = img * 2 - 1  # [-1,1]
    return img.astype(np.float32)


def is_ct_image_dicom(path):
    """
    Returns True only for CT image storage objects. Skips DICOM SEG and any non-CT modalities.
    Uses header-only read for speed and robustness.
    """
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        modality = getattr(ds, 'Modality', '')
        if modality != 'CT':
            return False
        sop_class = str(getattr(ds, 'SOPClassUID', ''))
        allowed_sops = {
            '1.2.840.10008.5.1.4.1.1.2',    # CT Image Storage
            '1.2.840.10008.5.1.4.1.1.2.1',  # Enhanced CT Image Storage
        }
        # If SOPClassUID missing but modality CT, still accept to be lenient
        return sop_class in allowed_sops or sop_class == ''
    except Exception:
        return False


def find_dicom_files_recursively(base_folder):
    dicom_files = []
    ignore_dirs = {'ctkDICOM-Database', '.git', '__pycache__'}
    print(f"[CT-Loader] Scanning DICOMs under: {base_folder}")
    for root, dirs, files in os.walk(base_folder):
        # prune ignored directories in-place to avoid descending into them
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for f in files:
            if f.lower().endswith('.dcm'):
                path = os.path.join(root, f)
                if is_ct_image_dicom(path):
                    dicom_files.append(path)
    print(f"[CT-Loader] Found {len(dicom_files)} CT image files")
    return sorted(dicom_files)

def load_dicom_as_tensor(path, normalization='global', hu_clip=(-1000, 2000), window_center=40, window_width=400):
    """
    Load a DICOM slice as a normalized tensor [1,H,W].
    - Applies Modality LUT (RescaleSlope/Intercept) to obtain HU when present.
    - normalization='global': clip to hu_clip (default [-1000,2000]) and scale to [-1,1].
    - normalization='window': apply window_center/width to scale to [-1,1] (legacy behavior).
    """
    ds = pydicom.dcmread(path, force=True)
    arr = ds.pixel_array
    try:
        hu = apply_modality_lut(arr, ds).astype(np.float32)
    except Exception:
        hu = arr.astype(np.float32)

    if normalization == 'window':
        img = apply_window(hu, window_center, window_width)
    else: #bei global normalization wird der HU-Wert zwischen -1000 und 2000 geclippt und dann auf [-1,1] skaliert
        lo, hi = hu_clip
        img = np.clip(hu, lo, hi)
        img = (img - lo) / (hi - lo)  # [0,1]
        img = img * 2 - 1             # [-1,1]
        img = img.astype(np.float32)

    tensor = torch.tensor(img).unsqueeze(0)  # [1, H, W]
    return tensor

def downsample_tensor(tensor, scale_factor=2, *, antialias=True):
    tensor = tensor.unsqueeze(0)  # [1, 1, H, W]
    ds = F.interpolate(tensor, scale_factor=1/scale_factor, mode='bilinear', align_corners=False, antialias=antialias)
    return ds.squeeze(0)  # [1, H/s, W/s]


def _gaussian_kernel_2d(sigma: float, kernel_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # ensure odd kernel size
    k = int(kernel_size)
    if k % 2 == 0:
        k = k + 1
    half = (k - 1) // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    gauss_1d = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    gauss_1d = gauss_1d / gauss_1d.sum()
    kernel_2d = torch.outer(gauss_1d, gauss_1d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.view(1, 1, k, k)


def gaussian_blur_2d(tensor_1chw: torch.Tensor, sigma: float, kernel_size: int) -> torch.Tensor:
    # tensor_1chw: [1,H,W]
    device = tensor_1chw.device
    dtype = tensor_1chw.dtype
    kernel = _gaussian_kernel_2d(max(1e-6, float(sigma)), kernel_size, device, dtype)
    # pad reflect to preserve size
    k = kernel.shape[-1]
    pad = (k // 2, k // 2, k // 2, k // 2)
    x = tensor_1chw.unsqueeze(0)  # [1,1,H,W]
    x = F.pad(x, pad, mode='reflect')
    out = F.conv2d(x, kernel)
    return out.squeeze(0)  # [1,H,W]


def _compute_kernel_size_from_sigma(sigma: float) -> int:
    # common heuristic: k ~ 6*sigma rounded to nearest odd
    k = int(max(3, round(6.0 * float(sigma))))
    if k % 2 == 0:
        k += 1
    return k

class CT_Dataset_SR(Dataset):
    def __init__(self, dicom_folder, window_center=40, window_width=400, scale_factor=2, max_slices=None,
                 do_random_crop=True, hr_patch=128, normalization='global', hu_clip=(-1000, 2000),
                 degradation='blurnoise', blur_sigma_range=None, blur_kernel=None,
                 noise_sigma_range_norm=(0.001, 0.003), dose_factor_range=(0.25, 0.5), antialias_clean=True):
        self.paths = find_dicom_files_recursively(dicom_folder)
        if max_slices:
            self.paths = self.paths[:max_slices]
        self.wc = window_center
        self.ww = window_width
        self.scale = scale_factor
        self.do_random_crop = do_random_crop
        self.hr_patch = hr_patch
        self.normalization = normalization  # 'global' (default) or 'window'
        self.hu_clip = hu_clip
        # degradation settings
        self.degradation = degradation  # 'clean' | 'blur' | 'blurnoise'
        # default sigma ranges based on scale if not provided
        if blur_sigma_range is None:
            base_sigma = 0.8 if self.scale == 2 else (1.2 if self.scale == 4 else 0.8)
            jitter = 0.1 if self.scale == 2 else 0.15
            self.blur_sigma_range = (max(1e-6, base_sigma - jitter), base_sigma + jitter)
        else:
            self.blur_sigma_range = tuple(blur_sigma_range)
        self.blur_kernel = blur_kernel  # if None, derive per-sample from sigma
        self.noise_sigma_range_norm = tuple(noise_sigma_range_norm)
        self.dose_factor_range = tuple(dose_factor_range)
        self.antialias_clean = bool(antialias_clean)
        if self.normalization == 'global':
            norm_desc = f"global_HU_clip={self.hu_clip}"
        else:
            norm_desc = f"window=({self.wc},{self.ww})"
        print(f"[CT-Loader] Dataset ready: {len(self.paths)} slices | scale={self.scale} | norm={norm_desc} | random_crop={self.do_random_crop}")
        print(f"[CT-Loader] Degradation='{self.degradation}' | blur_sigma_range={self.blur_sigma_range} | blur_kernel={self.blur_kernel} | noise_sigma_range_norm={self.noise_sigma_range_norm} | dose_factor_range={self.dose_factor_range}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        hr_full = load_dicom_as_tensor(self.paths[idx], normalization=self.normalization, hu_clip=self.hu_clip,
                                       window_center=self.wc, window_width=self.ww)   # [1, H, W]
        # choose HR region
        if self.do_random_crop and self.hr_patch is not None:
            _, H, W = hr_full.shape
            if H >= self.hr_patch and W >= self.hr_patch:
                max_y = H - self.hr_patch
                max_x = W - self.hr_patch
                y0 = random.randint(0, max_y) if max_y > 0 else 0
                x0 = random.randint(0, max_x) if max_x > 0 else 0
                hr = hr_full[:, y0:y0+self.hr_patch, x0:x0+self.hr_patch]
            else:
                hr = hr_full
        else:
            # center-crop to multiples of scale for full-slice evaluation
            hr = hr_full
            _, H, W = hr.shape
            target_H = (H // self.scale) * self.scale
            target_W = (W // self.scale) * self.scale
            if target_H > 0 and target_W > 0 and (target_H != H or target_W != W):
                y0 = (H - target_H) // 2
                x0 = (W - target_W) // 2
                hr = hr[:, y0:y0+target_H, x0:x0+target_W]

        # sample jitter parameters per item
        if self.degradation in ('blur', 'blurnoise'):
            sig_lo, sig_hi = self.blur_sigma_range
            blur_sigma = random.uniform(sig_lo, sig_hi)
            k = self.blur_kernel if self.blur_kernel is not None else _compute_kernel_size_from_sigma(blur_sigma)
            hr_for_lr = gaussian_blur_2d(hr, sigma=blur_sigma, kernel_size=k)
        else:
            hr_for_lr = hr

        # downsample
        if self.degradation == 'clean':
            lr = downsample_tensor(hr_for_lr, self.scale, antialias=self.antialias_clean)
        else:
            # disable antialias to respect explicit blur kernel
            lr = downsample_tensor(hr_for_lr, self.scale, antialias=False)

        # optional noise on LR
        if self.degradation == 'blurnoise':
            n_lo, n_hi = self.noise_sigma_range_norm
            noise_sigma = random.uniform(n_lo, n_hi)
            d_lo, d_hi = self.dose_factor_range
            dose = random.uniform(min(d_lo, d_hi), max(d_lo, d_hi))
            noise_eff = noise_sigma / max(1e-6, float(dose)) ** 0.5
            lr = lr + torch.randn_like(lr) * noise_eff
            lr = torch.clamp(lr, -1.0, 1.0)

        return lr, hr


# Selbst-test-Beispiel:
if __name__ == "__main__":
    dataset = CT_Dataset_SR(
        #r"C:\AA_Leonard\A_Studium\Bachelorarbeit Superresolution\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG", #Laptop
        r"C:\BachelorarbeitLeo\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG", #RTX4080 Super
        #max_slices=20  # wir nehmen erstmal nur 20 Slices
    )
    print(f"Anzahl Bilder: {len(dataset)}")
    lr, hr = dataset[0]
    print(f"LR-Shape: {lr.shape}, HR-Shape: {hr.shape}")

