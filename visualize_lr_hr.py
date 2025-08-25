import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pydicom

from window_presets import WINDOW_PRESETS
from ct_dataset_loader import is_ct_image_dicom


def apply_window_np(img, center, width):
    min_val = center - width / 2.0
    max_val = center + width / 2.0
    img = np.clip(img.astype(np.float32), min_val, max_val)
    img = (img - min_val) / (max_val - min_val)
    img = img * 2.0 - 1.0
    return img.astype(np.float32)


def load_ct_volume(folder_path, preset="soft_tissue"):
    """Load only CT image DICOMs; expand multi-frame to per-slice tensors. Returns [D,1,H,W] in [-1,1]."""
    window = WINDOW_PRESETS.get(preset, WINDOW_PRESETS["default"])
    wl, ww = window["center"], window["width"]

    slice_list = []
    for root, _, files in os.walk(folder_path):
        for f in sorted(files):
            if not f.lower().endswith('.dcm'):
                continue
            path = os.path.join(root, f)
            if not is_ct_image_dicom(path):
                continue
            try:
                ds = pydicom.dcmread(path, force=True)
                arr = ds.pixel_array
                if arr.ndim == 2:
                    img = apply_window_np(arr, wl, ww)
                    slice_list.append(torch.tensor(img).unsqueeze(0))  # [1,H,W]
                elif arr.ndim == 3:
                    # Multi-frame: expand each frame
                    for k in range(arr.shape[0]):
                        img = apply_window_np(arr[k], wl, ww)
                        slice_list.append(torch.tensor(img).unsqueeze(0))
            except Exception:
                continue

    if len(slice_list) == 0:
        raise RuntimeError(f"No CT image DICOM files found under {folder_path}")
    # Ensure consistent H,W
    H, W = slice_list[0].shape[-2:] 
    slice_list = [s for s in slice_list if s.shape[-2:] == (H, W)]
    vol = torch.stack(slice_list, dim=0)  # [D,1,H,W]
    return vol


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
    return int(np.clip(hr_index, 0, min(D_hr, D_lr) - 1))


def build_lr_volume_from_hr(hr_volume, scale=2):
	# Per-slice bilinear interpolation: treat D as batch dimension
	return F.interpolate(hr_volume, scale_factor=(1.0/scale, 1.0/scale), mode='bilinear', align_corners=False)


class ViewerLRHR:
    def __init__(self, lr_volume, hr_volume):
        self.lr = lr_volume   # [D,1,h,w]
        self.hr = hr_volume   # [D,1,H,W]
        D, _, _, _ = self.hr.shape
        self.index = D // 2

        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 6))
        self.ax_lr, self.ax_hr = self.axes
        self.ax_lr.set_title('LR')
        self.ax_hr.set_title('HR (original)')
        for ax in self.axes:
            ax.axis('off')

        self.im_lr = None
        self.im_hr = None
        self.text = self.fig.text(0.5, 0.02, '', ha='center', va='bottom')

        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.update()

    def update(self):
        D_hr, _, _, _ = self.hr.shape
        D_lr, _, _, _ = self.lr.shape
        clamped_idx = int(np.clip(self.index, 0, min(D_hr, D_lr) - 1))

        lr_plane, _, _ = extract_slice(self.lr, clamped_idx)
        hr_plane, axis_len, _ = extract_slice(self.hr, clamped_idx)

        img_lr = to_display(lr_plane)
        img_hr = to_display(hr_plane)

        if self.im_lr is None:
            self.im_lr = self.ax_lr.imshow(img_lr, cmap='gray', vmin=0, vmax=1)
        else:
            self.im_lr.set_data(img_lr)
        if self.im_hr is None:
            self.im_hr = self.ax_hr.imshow(img_hr, cmap='gray', vmin=0, vmax=1)
        else:
            self.im_hr.set_data(img_hr)

        self.text.set_text(f'Index: {clamped_idx+1}/{axis_len}')
        self.fig.canvas.draw_idle()

    def on_scroll(self, event):
        step = 1 if getattr(event, 'step', 0) >= 0 else -1
        if event.button == 'up':
            step = 1
        elif event.button == 'down':
            step = -1
        self.index += step
        self.update()


def main():
    parser = argparse.ArgumentParser(description='Visualize LR vs HR CT slices with mouse-wheel scrolling')
    parser.add_argument('--dicom_folder', type=str, required=True, help='Root folder containing DICOM series')
    parser.add_argument('--preset', type=str, default='soft_tissue', help='Window preset')
    parser.add_argument('--scale', type=int, default=2, help='Upsampling scale (must match model)')
    args = parser.parse_args()

    hr_vol = load_ct_volume(args.dicom_folder, preset=args.preset)          # [D,1,H,W]
    lr_vol = build_lr_volume_from_hr(hr_vol, scale=args.scale)              # [D,1,H/2,W/2]

    viewer = ViewerLRHR(lr_vol, hr_vol)
    print('Scroll with mouse wheel to navigate slices (LR left, HR right)')
    plt.show()


if __name__ == '__main__':
    main() 