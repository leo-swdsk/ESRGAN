import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pydicom
import math
from pydicom.pixel_data_handlers.util import apply_modality_lut

from window_presets import WINDOW_PRESETS
from ct_dataset_loader import is_ct_image_dicom
from rrdb_ct_model import RRDBNet_CT
from skimage.metrics import structural_similarity as ssim
import io, contextlib


def apply_window_np(img, center, width):
    min_val = center - width / 2.0
    max_val = center + width / 2.0
    img = np.clip(img.astype(np.float32), min_val, max_val)
    img = (img - min_val) / (max_val - min_val)
    img = img * 2.0 - 1.0
    return img.astype(np.float32)


def load_ct_volume(folder_path, preset="soft_tissue", override_window=None):
    window = WINDOW_PRESETS.get(preset, WINDOW_PRESETS["default"])
    if override_window is not None:
        wl, ww = override_window
    else:
        wl, ww = window["center"], window["width"]

    slice_list = []
    slice_paths = []
    
    # Sammle alle DICOM-Dateien und ihre Pfade
    for root, _, files in os.walk(folder_path):
        for f in sorted(files):
            if not f.lower().endswith('.dcm'):
                continue
            path = os.path.join(root, f)
            if not is_ct_image_dicom(path):
                continue
            slice_paths.append(path)
    
    # Sortiere die Pfade, um konsistente Reihenfolge zu gewährleisten
    slice_paths.sort()
    
    # Lade die Slices in der sortierten Reihenfolge
    for path in slice_paths:
        try:
            ds = pydicom.dcmread(path, force=True)
            arr = ds.pixel_array
            hu = apply_modality_lut(arr, ds).astype(np.float32)
            if hu.ndim == 2:
                img = apply_window_np(hu, wl, ww)
                slice_list.append(torch.tensor(img).unsqueeze(0))
            elif hu.ndim == 3:
                for k in range(hu.shape[0]):
                    img = apply_window_np(hu[k], wl, ww)
                    slice_list.append(torch.tensor(img).unsqueeze(0))
        except Exception:
            continue

    if len(slice_list) == 0:
        raise RuntimeError(f"No CT image DICOM files found under {folder_path}")
    
    H, W = slice_list[0].shape[-2:]
    slice_list = [s for s in slice_list if s.shape[-2:] == (H, W)]
    
    # Kehre die Schichtreihenfolge um: erste geladene Schicht wird höchster Index
    slice_list.reverse()
    
    # Erstelle das Volumen - erste geladene Schicht wird höchster Index (wie in Slicer 3D)
    vol = torch.stack(slice_list, dim=0)
    
    print(f"[CT-Loader] Loaded {vol.shape[0]} slices with dimensions {vol.shape[1:]} (Index 0 = last loaded slice, Index {vol.shape[0]-1} = first loaded slice)")
    return vol


def read_series_metadata(folder_path):
    row_mm = None; col_mm = None; slice_thickness = None; patient_id = ''
    for root, _, files in os.walk(folder_path):
        for f in sorted(files):
            if not f.lower().endswith('.dcm'):
                continue
            path = os.path.join(root, f)
            if not is_ct_image_dicom(path):
                continue
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
                ps = getattr(ds, 'PixelSpacing', None)
                if ps is not None and len(ps) >= 2:
                    row_mm = float(ps[0]); col_mm = float(ps[1])
                st = getattr(ds, 'SliceThickness', None)
                if st is not None:
                    slice_thickness = float(st)
                patient_id = str(getattr(ds, 'PatientID', ''))
                return (row_mm, col_mm), slice_thickness, patient_id
            except Exception:
                continue
    return (row_mm, col_mm), slice_thickness, patient_id

def to_display(img_tensor):
    img = img_tensor.detach().cpu().numpy()
    img = ((img + 1.0) / 2.0).clip(0.0, 1.0)
    return img


def extract_slice(volume, index):
    D, _, _, _ = volume.shape
    index = int(np.clip(index, 0, D - 1))
    return volume[index, 0, :, :], D, index


def map_index_between_hr_lr(hr_index, hr_shape, lr_shape):
    D_hr, _, _, _ = hr_shape
    D_lr, _, _, _ = lr_shape
    # Beide Volumen haben die gleiche Anzahl Schichten, Index 0-basiert
    return int(np.clip(hr_index, 0, min(D_hr, D_lr) - 1))


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
                     blur_kernel: int = None, noise_sigma_range_norm=(0.001, 0.003), dose_factor_range=(1.0, 1.0),
                     antialias_clean: bool = True) -> torch.Tensor:
    device = hr_volume.device
    dtype = hr_volume.dtype
    if degradation in ('blur', 'blurnoise'):
        if blur_sigma_range is None:
            base_sigma = 0.8 if scale == 2 else (1.2 if scale == 4 else 0.8)
            jitter = 0.1 if scale == 2 else 0.15
            sig_lo, sig_hi = max(1e-6, base_sigma - jitter), base_sigma + jitter
        else:
            sig_lo, sig_hi = float(blur_sigma_range[0]), float(blur_sigma_range[1])
        sigma = float(np.random.uniform(sig_lo, sig_hi))
        k = blur_kernel if blur_kernel is not None else _kernel_size_from_sigma(sigma)
        kernel = _gaussian_kernel_2d(max(1e-6, sigma), k, device, dtype)
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
        noise_sigma = float(np.random.uniform(n_lo, n_hi))
        dose = float(np.random.uniform(min(d_lo, d_hi), max(d_lo, d_hi)))
        noise_eff = noise_sigma / max(1e-6, dose) ** 0.5
        lr = torch.clamp(lr + torch.randn_like(lr) * noise_eff, -1.0, 1.0)
    return lr


def build_lr_volume_from_hr(hr_volume, scale=2, *, degradation='blurnoise', blur_sigma_range=None, blur_kernel=None,
							 noise_sigma_range_norm=(0.001, 0.003), dose_factor_range=(1.0, 1.0), antialias_clean=True):
	return degrade_hr_to_lr(hr_volume, scale,
		degradation=degradation,
		blur_sigma_range=blur_sigma_range,
		blur_kernel=blur_kernel,
		noise_sigma_range_norm=noise_sigma_range_norm,
		dose_factor_range=dose_factor_range,
		antialias_clean=antialias_clean)


def center_crop_to_multiple(volume: torch.Tensor, scale: int) -> torch.Tensor:
    # volume: [D,1,H,W]; crop H,W to nearest lower multiple of scale (centered)
    D, C, H, W = volume.shape
    target_H = (H // scale) * scale
    target_W = (W // scale) * scale
    if target_H <= 0 or target_W <= 0:
        return volume
    if target_H == H and target_W == W:
        return volume
    y0 = (H - target_H) // 2
    x0 = (W - target_W) // 2
    cropped = volume[:, :, y0:y0+target_H, x0:x0+target_W]
    print(f"[Vis] Center-cropped HR from ({H},{W}) -> ({target_H},{target_W}) to match scale={scale}")
    return cropped


def build_sr_volume_from_lr(lr_volume, model, batch_size: int = 8):
	device = next(model.parameters()).device
	model.eval()
	slices = lr_volume.shape[0]
	outs = []
	with torch.no_grad():
		for i in range(0, slices, max(1, batch_size)):
			batch = lr_volume[i:i+batch_size]  # [B,1,h,w]
			if device.type == 'cuda':
				with torch.amp.autocast('cuda'):
					y = model(batch.to(device))  # [B,1,H,W]
			else:
				y = model(batch.to(device))  # [B,1,H,W]
			outs.append(y.cpu())
	return torch.cat(outs, dim=0)  # [D,1,H,W]


class ViewerLRSRHR:
    def __init__(self, lr_volume, sr_volume, hr_volume, scale=2, lin_volume=None, bic_volume=None, *,
                 dicom_folder=None, preset_name="soft_tissue", model=None, device=None,
                 pixel_spacing_mm=None, slice_thickness_mm=None, patient_id=''):
        self.lr = lr_volume
        self.sr = sr_volume
        self.hr = hr_volume
        self.lin = lin_volume
        self.bic = bic_volume
        self.scale = scale
        self.dicom_folder = dicom_folder
        self.preset_name = preset_name
        self.model = model
        self.device = device
        # metadata
        self.ps_row_mm = None if pixel_spacing_mm is None else pixel_spacing_mm[0]
        self.ps_col_mm = None if pixel_spacing_mm is None else pixel_spacing_mm[1]
        self.slice_thickness_mm = slice_thickness_mm
        self.patient_id = patient_id
        D, _, _, _ = self.hr.shape
        self.index = 0  # Start bei Index 0 (erste Schicht)
        print(f"[Viewer] init: D={D} | LR={tuple(self.lr.shape)} SR={tuple(self.sr.shape)} HR={tuple(self.hr.shape)}")
        if self.lin is not None:
            print(f"[Viewer] LIN={tuple(self.lin.shape)}")
        if self.bic is not None:
            print(f"[Viewer] BIC={tuple(self.bic.shape)}")

        # Axes: LR | Linear | Bicubic | SR | HR
        self.fig, self.axes = plt.subplots(1, 5, figsize=(22, 6))
        self.ax_lr, self.ax_lin, self.ax_bic, self.ax_sr, self.ax_hr = self.axes
        self.ax_lr.set_title('LR')
        self.ax_lin.set_title('Linear x{}'.format(scale))
        self.ax_bic.set_title('Bicubic x{}'.format(scale))
        self.ax_sr.set_title('SR (model)')
        self.ax_hr.set_title('HR')
        for ax in self.axes:
            ax.axis('off')
            ax.set_aspect('equal')

        self.im_lr = None
        self.im_lin = None
        self.im_bic = None
        self.im_sr = None
        self.im_hr = None
        self.text = self.fig.text(0.5, 0.02, '', ha='center', va='bottom')
        self.text_stats = self.fig.text(0.5, 0.97, '', ha='center', va='top')
        # info top, metrics will be placed below with spacing
        self.text_info = self.fig.text(0.02, 0.995, '', ha='left', va='top', family='monospace')
        self.text_roi = self.fig.text(0.02, 0.10, '', ha='left', va='top', family='monospace')
        self.metric_texts = []

        self.roi = None  # (x0,y0,x1,y1) in HR coordinate space
        self.selector = None
        self._is_syncing = False  # guard to prevent recursive axis callbacks
        self._images_ready = False  # defer axis sync until images initialized

        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.enable_selector()
        # Custom Reset ROI button (bottom-left)
        from matplotlib.widgets import Button, TextBox, RadioButtons
        axbtn = self.fig.add_axes([0.01, 0.02, 0.12, 0.06])
        self.btn_roi = Button(axbtn, 'Reset ROI')
        def reset_roi_click(event):
            self.roi = None
            self.hide_roi_overlay()
            print('[ROI Button] Reset ROI')
            self.update()
        self.btn_roi.on_clicked(reset_roi_click)
        # Preset selection (radio buttons)
        preset_labels = list(WINDOW_PRESETS.keys())
        ax_radio = self.fig.add_axes([0.01, 0.14, 0.09, 0.40])
        self.radio_presets = RadioButtons(ax_radio, preset_labels, active=preset_labels.index(self.preset_name) if self.preset_name in preset_labels else 0)
        def on_preset(label):
            self.apply_new_window(preset=label)
        self.radio_presets.on_clicked(on_preset)
        # Text boxes for WL/WW and apply
        ax_wl = self.fig.add_axes([0.15, 0.02, 0.08, 0.05])
        ax_ww = self.fig.add_axes([0.25, 0.02, 0.08, 0.05])
        self.txt_wl = TextBox(ax_wl, 'WL', initial=str(WINDOW_PRESETS.get(self.preset_name, WINDOW_PRESETS['default'])['center']))
        self.txt_ww = TextBox(ax_ww, 'WW', initial=str(WINDOW_PRESETS.get(self.preset_name, WINDOW_PRESETS['default'])['width']))
        ax_apply = self.fig.add_axes([0.35, 0.02, 0.10, 0.05])
        self.btn_apply = Button(ax_apply, 'Apply WW')
        def apply_manual(event):
            try:
                wl = float(self.txt_wl.text)
                ww = float(self.txt_ww.text)
                self.apply_new_window(center=wl, width=ww)
            except Exception as e:
                print(f"[Apply WW] Invalid WL/WW input: {e}")
        self.btn_apply.on_clicked(apply_manual)
        print('[Hint] Navigation: Mouse wheel or arrow keys; Home/End for first/last loaded slice; Drag on HR to select ROI; press r to reset ROI; change presets or set WL/WW and click Apply')
        # sync to toolbar zoom/pan on all axes
        for ax in [self.ax_hr, self.ax_sr, self.ax_lin, self.ax_bic, self.ax_lr]:
            ax.callbacks.connect('xlim_changed', self.on_axes_limits_change)
            ax.callbacks.connect('ylim_changed', self.on_axes_limits_change)
        self.update()

    def update(self):
        D_hr, _, _, _ = self.hr.shape
        D_lr, _, _, _ = self.lr.shape
        D_sr, _, _, _ = self.sr.shape
        D_lin = self.lin.shape[0] if self.lin is not None else D_sr
        D_bic = self.bic.shape[0] if self.bic is not None else D_sr

        clamped_idx = int(np.clip(self.index, 0, min(D_hr, D_lr, D_sr, D_lin, D_bic) - 1))
        print(f"[Viewer.update] index={clamped_idx} / D={min(D_hr, D_lr, D_sr, D_lin, D_bic)} (0 = last loaded slice, {min(D_hr, D_lr, D_sr, D_lin, D_bic)-1} = first loaded slice)")

        hr_plane, axis_len, _ = extract_slice(self.hr, clamped_idx)
        lr_plane, _, _ = extract_slice(self.lr, clamped_idx)
        sr_plane, _, _ = extract_slice(self.sr, clamped_idx)
        lin_plane = None
        bic_plane = None
        if self.lin is not None:
            lin_plane, _, _ = extract_slice(self.lin, clamped_idx)
        if self.bic is not None:
            bic_plane, _, _ = extract_slice(self.bic, clamped_idx)
        print(f"[Viewer.update] Slice {clamped_idx}/{axis_len-1} | shapes HR={tuple(hr_plane.shape)} SR={tuple(sr_plane.shape)} LR={tuple(lr_plane.shape)} LIN={None if lin_plane is None else tuple(lin_plane.shape)} BIC={None if bic_plane is None else tuple(bic_plane.shape)}")

        # If ROI is set (in HR coords), synchronize axes limits across views
        if self.roi:
            x0, y0, x1, y1 = self.roi
            self.apply_axes_limits(x0, y0, x1, y1)

        img_lr = to_display(lr_plane)
        img_sr = to_display(sr_plane)
        img_hr = to_display(hr_plane)
        img_lin = to_display(lin_plane) if lin_plane is not None else np.zeros_like(img_hr)
        img_bic = to_display(bic_plane) if bic_plane is not None else np.zeros_like(img_hr)
        print(f"[Viewer.update] ranges LR=({img_lr.min():.3f},{img_lr.max():.3f}) SR=({img_sr.min():.3f},{img_sr.max():.3f}) HR=({img_hr.min():.3f},{img_hr.max():.3f})")

        if self.im_lr is None:
            self.im_lr = self.ax_lr.imshow(img_lr, cmap='gray', vmin=0, vmax=1, origin='lower')
        else:
            self.im_lr.set_data(img_lr)
        if self.im_lin is None:
            self.im_lin = self.ax_lin.imshow(img_lin, cmap='gray', vmin=0, vmax=1, origin='lower')
        else:
            self.im_lin.set_data(img_lin)
        if self.im_bic is None:
            self.im_bic = self.ax_bic.imshow(img_bic, cmap='gray', vmin=0, vmax=1, origin='lower')
        else:
            self.im_bic.set_data(img_bic)
        if self.im_sr is None:
            self.im_sr = self.ax_sr.imshow(img_sr, cmap='gray', vmin=0, vmax=1, origin='lower')
        else:
            self.im_sr.set_data(img_sr)
        if self.im_hr is None:
            self.im_hr = self.ax_hr.imshow(img_hr, cmap='gray', vmin=0, vmax=1, origin='lower')
        else:
            self.im_hr.set_data(img_hr)

        # Info panel (always visible)
        info_parts = []
        if self.patient_id:
            info_parts.append(f"PatientID: {self.patient_id}")
        if self.slice_thickness_mm is not None:
            info_parts.append(f"Slice Thickness: {self.slice_thickness_mm:.2f} mm")
        if self.ps_row_mm and self.ps_col_mm:
            info_parts.append(f"PixelSpacing: {self.ps_row_mm:.3f} x {self.ps_col_mm:.3f} mm")
        self.text_info.set_text(' | '.join(info_parts))
        self.text.set_text(f'Index: {clamped_idx}/{axis_len-1}')

        # If no ROI is active, ensure axes show full images explicitly
        if not self.roi:
            h_lr, w_lr = img_lr.shape
            h_hr, w_hr = img_hr.shape
            self._is_syncing = True
            try:
                self.ax_lr.set_xlim(-0.5, w_lr-0.5); self.ax_lr.set_ylim(h_lr-0.5, -0.5)
                for ax in [self.ax_lin, self.ax_bic, self.ax_sr, self.ax_hr]:
                    ax.set_xlim(-0.5, w_hr-0.5); ax.set_ylim(h_hr-0.5, -0.5)
            finally:
                self._is_syncing = False

        # Determine ROI arrays for metrics (crop if ROI exists) using non-displayed tensors to avoid previous crops
        x0i = y0i = x1i = y1i = None
        H, W = self.hr.shape[-2:]
        # Always show ROI status line; if no ROI, treat as full image
        if self.roi:
            x0, y0, x1, y1 = self.roi
        else:
            x0, y0, x1, y1 = 0, 0, W, H
        x0i, x1i = int(round(max(0, min(x0, W-1)))), int(round(max(0, min(x1, W))))
        y0i, y1i = int(round(max(0, min(y0, H-1)))), int(round(max(0, min(y1, H))))
        # ROI overlay text with px and mm + ROI FOV; also include LR resolution
        w_px = max(0, x1i - x0i)
        h_px = max(0, y1i - y0i)
        if self.ps_row_mm and self.ps_col_mm:
            w_mm = w_px * self.ps_col_mm
            h_mm = h_px * self.ps_row_mm
            fov_w_mm = W * self.ps_col_mm
            fov_h_mm = H * self.ps_row_mm
            roi_text = f"ROI HR: ({x0i},{y0i})-({x1i},{y1i}) px | {w_px}x{h_px} px | ROI FOV {w_mm:.1f}x{h_mm:.1f} mm | Global FOV {fov_w_mm:.1f}x{fov_h_mm:.1f} mm"
        else:
            roi_text = f"ROI HR: ({x0i},{y0i})-({x1i},{y1i}) px | {w_px}x{h_px} px"
        # map to LR coords (may be fractional due to scale; display as-is)
        x0_lr = x0i / float(self.scale)
        y0_lr = y0i / float(self.scale)
        x1_lr = x1i / float(self.scale)
        y1_lr = y1i / float(self.scale)
        lr_h, lr_w = self.lr.shape[-2:]
        roi_text += f" | mapped LR: ({x0_lr:.2f},{y0_lr:.2f})-({x1_lr:.2f},{y1_lr:.2f}) | LR res: {lr_w}x{lr_h} px"
        self.text_roi.set_text(roi_text)

        def crop_roi_full(vol):
            t = vol[clamped_idx, 0]
            if self.roi and x1i is not None and x1i > x0i and y1i > y0i:
                return t[y0i:y1i, x0i:x1i]
            return t

        hr_for_metrics = crop_roi_full(self.hr)
        sr_for_metrics = crop_roi_full(self.sr)
        lin_for_metrics = crop_roi_full(self.lin) if self.lin is not None else hr_for_metrics
        bic_for_metrics = crop_roi_full(self.bic) if self.bic is not None else hr_for_metrics

        metrics = {}
        metrics['SR'] = self.compute_metrics(sr_for_metrics, hr_for_metrics)
        if lin_for_metrics is not None:
            metrics['Linear'] = self.compute_metrics(lin_for_metrics, hr_for_metrics)
        if bic_for_metrics is not None:
            metrics['Bicubic'] = self.compute_metrics(bic_for_metrics, hr_for_metrics)

        # Clear previous metric texts
        for t in self.metric_texts:
            try:
                t.remove()
            except Exception:
                pass
        self.metric_texts = []

       # Find best per metric (MSE/RMSE/MAE min, PSNR/SSIM max) across all methods present
        names = list(metrics.keys())
        by_metric = {'MSE': {}, 'RMSE': {}, 'MAE': {}, 'PSNR': {}, 'SSIM': {}, 'LPIPS': {}, 'PI': {}}
        for name in names:
            mse_val, rmse_val, mae_val, psnr_val, ssim_val, lpips_val, pi_val = metrics[name]
            by_metric['MSE'][name] = mse_val
            by_metric['RMSE'][name] = rmse_val
            by_metric['MAE'][name] = mae_val  # MAE hinzugefügt
            by_metric['PSNR'][name] = psnr_val
            by_metric['SSIM'][name] = ssim_val
            by_metric['LPIPS'][name] = lpips_val
            by_metric['PI'][name] = pi_val
        print(f"[Viewer.update] metrics: {metrics}")
        best = {
            'MSE': min(by_metric['MSE'], key=by_metric['MSE'].get),
            'RMSE': min(by_metric['RMSE'], key=by_metric['RMSE'].get),
            'MAE': min(by_metric['MAE'], key=by_metric['MAE'].get),
            'PSNR': max(by_metric['PSNR'], key=by_metric['PSNR'].get),
            'SSIM': max(by_metric['SSIM'], key=by_metric['SSIM'].get),
            'LPIPS': min(by_metric['LPIPS'], key=by_metric['LPIPS'].get),
            'PI': min(by_metric['PI'], key=by_metric['PI'].get),
        }

        # Layout metric labels and values in one row; best ones green
        # Headers with left column label
        header_y = 0.965
        def put_header(x, txt):
            self.metric_texts.append(self.fig.text(x, header_y, txt, ha='left', va='top', family='monospace', color='black'))
        put_header(0.02, "Metrics")
        put_header(0.06, "MSE ↓")
        put_header(0.16, "RMSE ↓")
        put_header(0.28, "MAE ↓")
        put_header(0.38, "PSNR ↑")
        put_header(0.50, "SSIM ↑")
        put_header(0.62, "LPIPS ↓")
        put_header(0.74, "PI ↓")

        y0_text = 0.93
        dy = 0.047
        for i, name in enumerate(['SR', 'Linear', 'Bicubic']):
            if name not in metrics:
                continue
            mse_val, rmse_val, mae_val, psnr_val, ssim_val, lpips_val, pi_val = metrics[name]
            # single row per method with columns in the specified order
            y = y0_text - i*dy
            def put_val(x, value, is_best, fmt):
                color = 'green' if is_best else 'black'
                self.metric_texts.append(self.fig.text(x, y, fmt.format(value), ha='left', va='top', color=color, family='monospace'))
            self.metric_texts.append(self.fig.text(0.02, y, f"{name}", ha='left', va='top', family='monospace', color='black'))
            put_val(0.06, mse_val, name == best['MSE'], "{:.6f}")
            put_val(0.16, rmse_val, name == best['RMSE'], "{:.6f}")
            put_val(0.28, mae_val, name == best['MAE'], "{:.6f}")
            put_val(0.38, psnr_val, name == best['PSNR'], "{:.2f}")
            put_val(0.50, ssim_val, name == best['SSIM'], "{:.4f}")
            put_val(0.62, lpips_val, name == best['LPIPS'], "{:.4f}")
            put_val(0.74, pi_val, name == best['PI'], "{:.3f}")
        self._images_ready = True
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        step = 1 if getattr(event, 'step', 0) >= 0 else -1
        if event.button == 'up':
            step = 1
        elif event.button == 'down':
            step = -1
        old_index = self.index
        D, _, _, _ = self.hr.shape
        # clamp index strictly within [0, D-1]
        self.index = int(np.clip(self.index + step, 0, D - 1))
        print(f"[Scroll] Slice {old_index} -> {self.index} (0 = last loaded slice, {D-1} = first loaded slice)")
        self.update()

    def on_key(self, event):
        if event.key in ['r', 'R']:
            # reset ROI and re-enable selector
            self.roi = None
            self.hide_roi_overlay()
            self.update()
        elif event.key == 'home':
            # Go to first loaded slice (höchster Index)
            D, _, _, _ = self.hr.shape
            self.index = D - 1
            print(f"[Key] Home -> Slice {D-1} (first loaded slice)")
            self.update()
        elif event.key == 'end':
            # Go to last loaded slice (Index 0)
            self.index = 0
            print(f"[Key] End -> Slice 0 (last loaded slice)")
            self.update()
        elif event.key in ['left', 'up']:
            # Previous slice (höherer Index)
            D, _, _, _ = self.hr.shape
            if self.index < D - 1:
                self.index = min(self.index + 1, D - 1)
                print(f"[Key] Previous -> Slice {self.index}")
                self.update()
        elif event.key in ['right', 'down']:
            # Next slice (niedrigerer Index)
            if self.index > 0:
                self.index = max(self.index - 1, 0)
                print(f"[Key] Next -> Slice {self.index}")
                self.update()

    def enable_selector(self):
        from matplotlib.widgets import RectangleSelector
        # Create selector on HR axes to define ROI in HR coordinates
        if self.selector is not None:
            try:
                self.selector.disconnect_events()
            except Exception:
                pass
            self.selector = None

        def onselect(eclick, erelease):
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = erelease.xdata, erelease.ydata
            if x0 is None or y0 is None or x1 is None or y1 is None:
                return
            # Normalize and clip to integer pixel indices of HR
            H, W = self.hr.shape[-2:]
            x0n, x1n = (x0, x1) if x0 <= x1 else (x1, x0)
            y0n, y1n = (y0, y1) if y0 <= y1 else (y1, y0)
            x0n = max(0, min(int(round(x0n)), W-1))
            x1n = max(0, min(int(round(x1n)), W))
            y0n = max(0, min(int(round(y0n)), H-1))
            y1n = max(0, min(int(round(y1n)), H))
            # enforce minimum size to avoid empty ROI
            if x1n <= x0n + 1 or y1n <= y0n + 1:
                return
            self.roi = (x0n, y0n, x1n, y1n)
            print(f"[RectangleSelector] ROI HR: ({x0n}, {y0n}, {x1n}, {y1n}) | mapped LR: ({x0n/self.scale:.2f}, {y0n/self.scale:.2f}, {x1n/self.scale:.2f}, {y1n/self.scale:.2f})")
            # Sync axes to ROI immediately
            self.hide_roi_overlay()
            self.apply_axes_limits(x0n, y0n, x1n, y1n)
            self.update()

        self.selector = RectangleSelector(
            self.ax_hr, onselect,
            useblit=True, button=[1],  # left mouse drag
            minspanx=5, minspany=5, interactive=True,
            spancoords='pixels'
        )
        self.selector.set_active(True)

    def hide_roi_overlay(self):
        # Remove any red rectangle artifacts from previous selector draws
        try:
            for attr in ['to_draw', 'artists']:
                if hasattr(self.selector, attr):
                    for artist in getattr(self.selector, attr):
                        artist.set_visible(False)
        except Exception:
            pass

    def on_axes_limits_change(self, ax):
        # Sync other axes when any view is zoomed/panned; compute ROI in HR coords
        if self._is_syncing or not self._images_ready:
            return
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        # ignore spurious callbacks with tiny height/width (e.g., during init)
        if (x1 - x0) < 5 or (y1 - y0) < 5:
            return
        # map LR axes to HR coordinates via scale
        if ax is self.ax_lr:
            fx = float(self.scale); fy = float(self.scale)
            xr0, xr1 = x0*fx, x1*fx
            yr0, yr1 = y0*fy, y1*fy
        else:
            xr0, xr1 = x0, x1
            yr0, yr1 = y0, y1
        # store ROI in HR coordinates; normalize to ascending
        xr0, xr1 = (xr0, xr1) if xr0 < xr1 else (xr1, xr0)
        yr0, yr1 = (yr0, yr1) if yr0 < yr1 else (yr1, yr0)
        self.roi = (xr0, yr0, xr1, yr1)
        print(f"[AxesChanged] src={ax.get_title()} xlim=({x0:.1f},{x1:.1f}) ylim=({y0:.1f},{y1:.1f}) -> HR ROI={self.roi}")
        self.apply_axes_limits(xr0, yr0, xr1, yr1)
        self.update()

    def apply_axes_limits(self, x0, y0, x1, y1):
        # Apply HR ROI to SR/LIN/BIC axes; map to LR via scale
        if x1 <= x0 or y1 <= y0:
            return
        try:
            self._is_syncing = True
            # Set HR/SR/LIN/BIC limits directly in HR coords
            for ax in [self.ax_hr, self.ax_sr, self.ax_lin, self.ax_bic]:
                if ax is not None:
                    ax.set_xlim(x0-0.5, x1-0.5)
                    ax.set_ylim(y1-0.5, y0-0.5)
            # LR mapping with scale
            fx = float(self.scale)
            fy = float(self.scale)
            self.ax_lr.set_xlim(x0/fx-0.5, x1/fx-0.5)
            self.ax_lr.set_ylim(y1/fy-0.5, y0/fy-0.5)
            print(f"[ApplyLimits] HR=({x0},{y0})-({x1},{y1}) | LR=({x0/fx:.2f},{y0/fy:.2f})-({x1/fx:.2f},{y1/fy:.2f})")
        finally:
            self._is_syncing = False

    def apply_new_window(self, preset=None, center=None, width=None):
        # Reload volumes with new window and rebuild derived volumes
        if self.dicom_folder is None or self.model is None:
            print('[Window] Missing dicom_folder or model; cannot rewindow')
            return
        if preset is not None:
            self.preset_name = preset
            override = None
        else:
            override = (center, width)
        print(f"[Window] Rebuilding with preset={self.preset_name} override={override}")
        new_hr = load_ct_volume(self.dicom_folder, preset=self.preset_name, override_window=override)
        new_hr = center_crop_to_multiple(new_hr, self.scale)
        new_lr = build_lr_volume_from_hr(new_hr, scale=self.scale)
        # model inference per slice
        # batched model inference for speed
        new_sr = build_sr_volume_from_lr(new_lr, self.model, batch_size=8)
        # linear/bicubic upscales for comparison
        new_lin = F.interpolate(new_lr, scale_factor=(self.scale, self.scale), mode='bilinear', align_corners=False)
        new_bic = F.interpolate(new_lr, scale_factor=(self.scale, self.scale), mode='bicubic', align_corners=False)
        # swap and reset ROI
        self.hr, self.lr, self.sr, self.lin, self.bic = new_hr, new_lr, new_sr, new_lin, new_bic
        self.roi = None
        self.hide_roi_overlay()
        # keep index within bounds
        D = self.hr.shape[0]
        self.index = int(np.clip(self.index, 0, D-1))
        print(f"[Window] Reset index to {self.index} (0 = last loaded slice, {D-1} = first loaded slice)")
        # refresh WL/WW textbox to reflect current settings if coming from preset
        if preset is not None:
            wl = WINDOW_PRESETS.get(self.preset_name, WINDOW_PRESETS['default'])['center']
            ww = WINDOW_PRESETS.get(self.preset_name, WINDOW_PRESETS['default'])['width']
            try:
                self.txt_wl.set_val(str(wl))
                self.txt_ww.set_val(str(ww))
            except Exception:
                pass
        self.update()

    def compute_metrics(self, sr_plane_t, hr_plane_t):
        # Inputs are torch tensors in [-1,1]
        sr_np = sr_plane_t.detach().cpu().numpy()
        hr_np = hr_plane_t.detach().cpu().numpy()
        sr_np = ((sr_np + 1.0) / 2.0).clip(0.0, 1.0)
        hr_np = ((hr_np + 1.0) / 2.0).clip(0.0, 1.0)
        diff = (hr_np.astype(np.float64) - sr_np.astype(np.float64))
        mse_val = float(np.mean(diff * diff))
        rmse_val = float(math.sqrt(mse_val))
        mae_val = float(np.mean(np.abs(diff)))  # MAE hinzugefügt
        if mse_val <= 0.0:
            psnr_val = float('inf')
        else:
            psnr_val = float(10.0 * math.log10(1.0 / mse_val))
        try:
            ssim_val = float(ssim(hr_np, sr_np, data_range=1.0))
        except Exception:
            ssim_val = float('nan')
        # LPIPS
        try:
            # Prefer pyiqa LPIPS if available
            lpips_val = float('nan')
            try:
                import pyiqa as _pyiqa
                if not hasattr(self, '_pyiqa_lpips'):
                    self._pyiqa_lpips = _pyiqa.create_metric('lpips')
                    self._pyiqa_lpips.eval()
                x = torch.tensor(sr_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                x3 = x.repeat(1,3,1,1)
                y = torch.tensor(hr_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                y3 = y.repeat(1,3,1,1)
                with torch.no_grad():
                    lpips_val = float(self._pyiqa_lpips(x3, y3).item())
            except Exception:
                import lpips as _lp
                lpips_model = getattr(self, '_lpips_model', None)
                if lpips_model is None:
                    lpips_model = _lp.LPIPS(net='alex')
                    lpips_model.eval()
                    self._lpips_model = lpips_model
                def to3_m11(x01):
                    x = torch.tensor(x01*2-1.0, dtype=torch.float32).unsqueeze(0)
                    return x.repeat(1,3,1,1)
                lpips_val = float(lpips_model(to3_m11(sr_np), to3_m11(hr_np)).item())
        except Exception:
            lpips_val = float('nan')
        # PI
        pi_val = float('nan')
        dbg_ma = float('nan')
        dbg_niqe = float('nan')
        try:
            import pyiqa as _pyiqa
            if not hasattr(self, '_piqa_niqe'):
                self._piqa_niqe = _pyiqa.create_metric('niqe')
                try:
                    self._piqa_niqe.to('cpu')
                except Exception:
                    pass
            # Initialize Ma metric once (nrqm in pyiqa)
            if not hasattr(self, '_ma_init_tried'):
                self._ma_init_tried = True
                self._ma_available = True
                try:
                    self._piqa_ma = _pyiqa.create_metric('nrqm')
                    try:
                        self._piqa_ma.to('cpu')
                    except Exception:
                        pass
                except Exception as e:
                    self._ma_available = False
                    print(f"[Ma-Setup] pyiqa 'nrqm' not available in this environment -> {e}")
                    print("[Ma-Setup] To enable PI, ensure pyiqa provides 'nrqm' metric and weights.")
            x = torch.tensor(sr_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            # Geräteausrichtung
            try:
                dev = next(self._piqa_niqe.parameters()).device
            except Exception:
                dev = torch.device('cpu')
            # NIQE auf nativer Graustufen-Auflösung
            x_niqe = x.to(dev)
            # Ma erfordert 3 Kanäle und konstante Eingangsgröße
            x3 = x.repeat(1,3,1,1)
            x3r = F.interpolate(x3, size=(224, 224), mode='bilinear', align_corners=False).to(dev)
            print(f"[Ma-Debug-Input] device={dev} shape={tuple(x3r.shape)} minmax=({float(x3r.min()):.4f},{float(x3r.max()):.4f}) mean={float(x3r.mean()):.4f}")
            with torch.no_grad():
                dbg_niqe = float(self._piqa_niqe(x_niqe).item())
                if getattr(self, '_ma_available', False):
                    dbg_ma = float(self._piqa_ma(x3r).item())
                else:
                    dbg_ma = float('nan')
            pi_val = float(0.5 * ((10.0 - dbg_ma) + dbg_niqe))
        except Exception as e:
            import traceback
            print(f"[Ma-Debug-Error] {e}")
            traceback.print_exc()
            try:
                from skimage.metrics import niqe as _sk_niqe
                dbg_niqe = float(_sk_niqe(sr_np.astype(np.float64)))
                pi_val = dbg_niqe
            except Exception:
                pass
        print(f"[PI-Debug] NIQE={dbg_niqe:.4f} MA={dbg_ma:.4f} -> PI={pi_val:.4f}")
        return mse_val, rmse_val, mae_val, psnr_val, ssim_val, lpips_val, pi_val


def main():
    parser = argparse.ArgumentParser(description='Visualize LR vs SR vs HR CT slices with mouse-wheel scrolling')
    parser.add_argument('--dicom_folder', type=str, required=True, help='Root folder containing DICOM series')
    parser.add_argument('--preset', type=str, default='soft_tissue', help='Window preset')
    parser.add_argument('--model_path', type=str, default='rrdb_ct_best.pth', help='Path to trained model weights')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    parser.add_argument('--scale', type=int, default=2, help='Upsampling scale (must match model)')
    parser.add_argument('--sr_batch', type=int, default=20, help='Batch size for batched SR inference (speed up GUI)')
    # Degradation flags (default blurnoise)
    parser.add_argument('--degradation', type=str, default='blurnoise', choices=['clean', 'blur', 'blurnoise'], help='Degradation pipeline for LR generation (default: blurnoise)')
    parser.add_argument('--blur_sigma_range', type=float, nargs=2, default=None, help='Range [lo hi] of Gaussian blur sigma; if None, defaults by scale')
    parser.add_argument('--blur_kernel', type=int, default=None, help='Explicit odd kernel size; if None, derived from sigma')
    parser.add_argument('--noise_sigma_range_norm', type=float, nargs=2, default=[0.001, 0.003], help='Gaussian noise sigma range on normalized [-1,1] image')
    parser.add_argument('--dose_factor_range', type=float, nargs=2, default=[0.25, 0.5], help='Dose factor range; noise scales ~ 1/sqrt(dose)')
    parser.add_argument('--antialias_clean', action='store_true', help='Use antialias in clean downsample')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    model = RRDBNet_CT(scale=args.scale).to(device)
    state = torch.load(args.model_path, map_location=device)
    if isinstance(state, dict) and 'model' in state and all(k in state for k in ['epoch', 'model']):
        print("[Vis] Detected checkpoint dict; loading weights from 'model' key")
        state = state['model']
    model.load_state_dict(state)
    model.eval()

    hr_vol = load_ct_volume(args.dicom_folder, preset=args.preset)
    # read DICOM metadata for pixel spacing, thickness, patient id
    (row_mm, col_mm), slice_thickness, patient_id = read_series_metadata(args.dicom_folder)
    hr_vol = center_crop_to_multiple(hr_vol, args.scale)
    print(f"[Vis] Degradation='{args.degradation}' | blur_sigma_range={args.blur_sigma_range} | blur_kernel={args.blur_kernel} | noise_sigma_range_norm={args.noise_sigma_range_norm} | dose_factor_range={args.dose_factor_range}")
    lr_vol = build_lr_volume_from_hr(
        hr_vol, scale=args.scale,
        degradation=args.degradation,
        blur_sigma_range=args.blur_sigma_range,
        blur_kernel=args.blur_kernel,
        noise_sigma_range_norm=args.noise_sigma_range_norm,
        dose_factor_range=args.dose_factor_range,
        antialias_clean=args.antialias_clean
    )
    sr_vol = build_sr_volume_from_lr(lr_vol, model, batch_size=args.sr_batch)
    # Also build linear and bicubic upscales of LR to HR size for side-by-side comparison
    # Treat D as batch, use bilinear/bicubic per slice
    lin_vol = F.interpolate(lr_vol, scale_factor=(args.scale, args.scale), mode='bilinear', align_corners=False)
    bic_vol = F.interpolate(lr_vol, scale_factor=(args.scale, args.scale), mode='bicubic', align_corners=False)

    viewer = ViewerLRSRHR(
        lr_vol, sr_vol, hr_vol,
        scale=args.scale,
        lin_volume=lin_vol,
        bic_volume=bic_vol,
        dicom_folder=args.dicom_folder,
        preset_name=args.preset,
        model=model,
        device=device,
        pixel_spacing_mm=(row_mm, col_mm) if (row_mm is not None and col_mm is not None) else None,
        slice_thickness_mm=slice_thickness,
        patient_id=patient_id,
    )
    print('Navigation: Mouse wheel or arrow keys to navigate slices')
    print('Keyboard shortcuts: Home (first loaded slice), End (last loaded slice), Arrow keys (previous/next)')
    print('Note: Slices are now reverse-indexed (0 = last loaded slice, highest index = first loaded slice, like in Slicer 3D)')
    plt.show()


if __name__ == '__main__':
    main() 