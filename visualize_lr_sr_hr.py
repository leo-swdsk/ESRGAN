import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from load_ct_tensor import load_ct_as_tensor
from rrdb_ct_model import RRDBNet_CT


def find_dicom_files_recursively(base_folder):
    dicom_files = []
    for root, _, files in os.walk(base_folder):
        for f in files:
            if f.lower().endswith('.dcm'):
                dicom_files.append(os.path.join(root, f))
    return sorted(dicom_files)


def load_volume(folder_path, preset="soft_tissue"):
    paths = find_dicom_files_recursively(folder_path)
    tensors = []
    for p in paths:
        t = load_ct_as_tensor(p, preset=preset)  # [1,H,W], values in [-1,1]
        tensors.append(t)
    if len(tensors) == 0:
        raise RuntimeError(f"No DICOM files found under {folder_path}")
    vol = torch.stack(tensors, dim=0)  # [D,1,H,W]
    return vol


def to_display(img_tensor):
    img = img_tensor.detach().cpu().numpy()
    img = ((img + 1.0) / 2.0).clip(0.0, 1.0)
    return img


def extract_plane(volume, orientation, index):
    D, _, H, W = volume.shape
    if orientation == 'axial':
        index = int(np.clip(index, 0, D - 1))
        plane = volume[index, 0, :, :]
        axis_len = D
    elif orientation == 'coronal':
        index = int(np.clip(index, 0, H - 1))
        plane = volume[:, 0, index, :]
        axis_len = H
    elif orientation == 'sagittal':
        index = int(np.clip(index, 0, W - 1))
        plane = volume[:, 0, :, index]
        axis_len = W
    else:
        raise ValueError('orientation must be one of axial|coronal|sagittal')
    return plane, axis_len, index


def map_index_between_hr_lr(hr_index, orientation, hr_shape, lr_shape):
    D_hr, H_hr, W_hr = hr_shape
    D_lr, H_lr, W_lr = lr_shape
    if orientation == 'axial':
        return int(np.clip(hr_index, 0, D_lr - 1))
    elif orientation == 'coronal':
        if H_hr <= 1:
            return 0
        return int(round(hr_index * (H_lr - 1) / (H_hr - 1)))
    elif orientation == 'sagittal':
        if W_hr <= 1:
            return 0
        return int(round(hr_index * (W_lr - 1) / (W_hr - 1)))
    else:
        return hr_index


def build_lr_volume_from_hr(hr_volume, scale=2):
	D, C, H, W = hr_volume.shape
	vol5 = hr_volume.permute(1, 0, 2, 3).unsqueeze(0)  # [1,1,D,H,W]
	lr5 = F.interpolate(vol5, scale_factor=(1.0, 1.0/scale, 1.0/scale), mode='trilinear', align_corners=False)
	lr = lr5.squeeze(0).permute(1, 0, 2, 3)  # [D,1,H/scale,W/scale]
	return lr


def build_sr_volume_from_lr(lr_volume, model):
	device = next(model.parameters()).device
	model.eval()
	srs = []
	with torch.no_grad():
		for z in range(lr_volume.shape[0]):
			sl = lr_volume[z:z+1]  # [1,1,h,w] (D dimension retained)
			sr = model(sl.to(device)).cpu()  # [1,1,H,W]
			srs.append(sr)
	return torch.cat(srs, dim=0)  # [D,1,H,W]


class ViewerLRSRHR:
    def __init__(self, lr_volume, sr_volume, hr_volume):
        self.lr = lr_volume
        self.sr = sr_volume
        self.hr = hr_volume
        D, _, H, W = self.hr.shape
        self.orientation = 'axial'
        self.index = D // 2

        self.fig, self.axes = plt.subplots(1, 3, figsize=(16, 6))
        self.ax_lr, self.ax_sr, self.ax_hr = self.axes
        self.ax_lr.set_title('LR')
        self.ax_sr.set_title('SR (model)')
        self.ax_hr.set_title('HR')
        for ax in self.axes:
            ax.axis('off')

        self.im_lr = None
        self.im_sr = None
        self.im_hr = None
        self.text = self.fig.text(0.5, 0.02, '', ha='center', va='bottom')

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.update()

    def update(self):
        D_hr, _, H_hr, W_hr = self.hr.shape
        D_lr, _, H_lr, W_lr = self.lr.shape

        hr_plane, axis_len, clamped_idx = extract_plane(self.hr, self.orientation, self.index)
        lr_index = map_index_between_hr_lr(clamped_idx, self.orientation, (D_hr, H_hr, W_hr), (D_lr, H_lr, W_lr))
        lr_plane, _, _ = extract_plane(self.lr, self.orientation, lr_index)
        sr_plane, _, _ = extract_plane(self.sr, self.orientation, clamped_idx)  # SR matches HR resolution

        img_lr = to_display(lr_plane)
        img_sr = to_display(sr_plane)
        img_hr = to_display(hr_plane)

        if self.im_lr is None:
            self.im_lr = self.ax_lr.imshow(img_lr, cmap='gray', vmin=0, vmax=1)
        else:
            self.im_lr.set_data(img_lr)
        if self.im_sr is None:
            self.im_sr = self.ax_sr.imshow(img_sr, cmap='gray', vmin=0, vmax=1)
        else:
            self.im_sr.set_data(img_sr)
        if self.im_hr is None:
            self.im_hr = self.ax_hr.imshow(img_hr, cmap='gray', vmin=0, vmax=1)
        else:
            self.im_hr.set_data(img_hr)

        self.text.set_text(f'Orientation: {self.orientation} | Index: {clamped_idx+1}/{axis_len}')
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        step = 1
        if event.key in ['right', 'down', 'j']:
            self.index += step
        elif event.key in ['left', 'up', 'k']:
            self.index -= step
        elif event.key in ['a']:
            self.orientation = 'axial'
            D, _, _, _ = self.hr.shape
            self.index = np.clip(self.index, 0, D - 1)
        elif event.key in ['c']:
            self.orientation = 'coronal'
            _, _, H, _ = self.hr.shape
            self.index = np.clip(self.index, 0, H - 1)
        elif event.key in ['s']:
            self.orientation = 'sagittal'
            _, _, _, W = self.hr.shape
            self.index = np.clip(self.index, 0, W - 1)
        self.update()

    def on_click(self, event):
        if event.inaxes not in [self.ax_lr, self.ax_sr, self.ax_hr]:
            return
        if event.inaxes is self.ax_lr:
            self.show_triplanar(self.lr, 'LR')
        elif event.inaxes is self.ax_sr:
            self.show_triplanar(self.sr, 'SR')
        else:
            self.show_triplanar(self.hr, 'HR')

    def show_triplanar(self, volume, title_prefix):
        D, _, H, W = volume.shape
        if self.orientation == 'axial':
            z = int(np.clip(self.index, 0, D - 1))
            y = H // 2
            x = W // 2
        elif self.orientation == 'coronal':
            z = D // 2
            y = int(np.clip(self.index, 0, H - 1))
            x = W // 2
        else:
            z = D // 2
            y = H // 2
            x = int(np.clip(self.index, 0, W - 1))

        axial = to_display(volume[z, 0])
        coronal = to_display(volume[:, 0, y, :])
        sagittal = to_display(volume[:, 0, :, x])

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(axial, cmap='gray', vmin=0, vmax=1); axs[0].set_title(f'{title_prefix} Axial (z={z})'); axs[0].axis('off')
        axs[1].imshow(coronal, cmap='gray', vmin=0, vmax=1); axs[1].set_title(f'{title_prefix} Coronal (y={y})'); axs[1].axis('off')
        axs[2].imshow(sagittal, cmap='gray', vmin=0, vmax=1); axs[2].set_title(f'{title_prefix} Sagittal (x={x})'); axs[2].axis('off')
        fig.suptitle(f'Tri-planar views ({title_prefix})')
        fig.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualize LR vs SR vs HR CT slices with orientation control')
    parser.add_argument('--dicom_folder', type=str, required=True, help='Root folder containing DICOM series')
    parser.add_argument('--preset', type=str, default='soft_tissue', help='Window preset (soft_tissue, lung, bone, brain, liver, abdomen, default)')
    parser.add_argument('--model_path', type=str, default='rrdb_ct_trained.pth', help='Path to trained model weights')
    parser.add_argument('--device', type=str, default='cuda', help='cuda or cpu')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    model = RRDBNet_CT(scale=2).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    hr_vol = load_volume(args.dicom_folder, preset=args.preset)   # [D,1,H,W]
    lr_vol = build_lr_volume_from_hr(hr_vol, scale=2)             # [D,1,H/2,W/2]
    sr_vol = build_sr_volume_from_lr(lr_vol, model)               # [D,1,H,W]

    viewer = ViewerLRSRHR(lr_vol, sr_vol, hr_vol)
    print('Controls: a=axial, c=coronal, s=sagittal | left/right/up/down or j/k to change slice | click image for tri-planar view')
    plt.show()


if __name__ == '__main__':
    main() 