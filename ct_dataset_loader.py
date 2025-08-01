import os
import torch
import pydicom
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

def apply_window(img, center, width):
    min_val = center - width / 2
    max_val = center + width / 2
    img = np.clip(img, min_val, max_val)
    img = (img - min_val) / (max_val - min_val)  # [0,1]
    img = img * 2 - 1  # [-1,1]
    return img.astype(np.float32)

def find_dicom_files_recursively(base_folder):
    dicom_files = []
    for root, _, files in os.walk(base_folder):
        for f in files:
            if f.lower().endswith(".dcm"):
                dicom_files.append(os.path.join(root, f))
    return sorted(dicom_files)

def load_dicom_as_tensor(path, window_center=40, window_width=400):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img = apply_window(img, window_center, window_width)
    tensor = torch.tensor(img).unsqueeze(0)  # [1, H, W]
    return tensor

def downsample_tensor(tensor, scale_factor=4):
    tensor = tensor.unsqueeze(0)  # [1, 1, H, W]
    ds = F.interpolate(tensor, scale_factor=1/scale_factor, mode='bilinear', align_corners=False)
    return ds.squeeze(0)  # [1, H/s, W/s]

class CT_Dataset_SR(Dataset):
    def __init__(self, dicom_folder, window_center=40, window_width=400, scale_factor=4, max_slices=None):
        self.paths = find_dicom_files_recursively(dicom_folder)
        if max_slices:
            self.paths = self.paths[:max_slices]
        self.wc = window_center
        self.ww = window_width
        self.scale = scale_factor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        hr = load_dicom_as_tensor(self.paths[idx], self.wc, self.ww)
        lr = downsample_tensor(hr, self.scale)
        return lr, hr

# Beispiel:
if __name__ == "__main__":
    dataset = CT_Dataset_SR(
        r"C:\AA_Leonard\A_Studium\Bachelorarbeit Superresolution\CNNCoding\data\train\manifest-1724965242274\Spine-Mets-CT-SEG\10352\12-03-2011-NA-SpineSPINEBONESBRT Adult-55418\4.000000-SKINTOSKINSIM0.5MM10352a iMAR-32611",
        max_slices=20  # wir nehmen erstmal nur 20 Slices
    )
    print(f"Anzahl Bilder: {len(dataset)}")
    lr, hr = dataset[0]
    print(f"LR-Shape: {lr.shape}, HR-Shape: {hr.shape}")

