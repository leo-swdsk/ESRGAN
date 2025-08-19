import os
import torch
import pydicom
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import os, random
from torch.utils.data import DataLoader, ConcatDataset
from ct_dataset_loader import CT_Dataset_SR

random.seed(42)

# 1) Patienten-Ordner (Top-Level) finden
root = r"C:\AA_Leonard\A_Studium\Bachelorarbeit Superresolution\ESRGAN-Med\data\manifest-1724965242274"  # Root folder
patient_dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
random.shuffle(patient_dirs)

# 2) Split patientenweise
n = len(patient_dirs)
train_dirs = patient_dirs[: int(0.8*n)]
val_dirs   = patient_dirs[int(0.8*n): int(0.9*n)]
test_dirs  = patient_dirs[int(0.9*n):]

# 3) Je Split: ConcatDataset aus Patienten-Datasets
train_ds = ConcatDataset([CT_Dataset_SR(d) for d in train_dirs])
val_ds   = ConcatDataset([CT_Dataset_SR(d) for d in val_dirs])
test_ds  = ConcatDataset([CT_Dataset_SR(d) for d in test_dirs])

# 4) Dataloader – mischen für Training
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False)
test_loader  = DataLoader(test_ds, batch_size=2, shuffle=False)


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

def load_dicom_as_tensor(path, window_center=40, window_width=400):
    ds = pydicom.dcmread(path)
    img = ds.pixel_array.astype(np.float32)
    img = apply_window(img, window_center, window_width)
    tensor = torch.tensor(img).unsqueeze(0)  # [1, H, W]
    return tensor

def downsample_tensor(tensor, scale_factor=2):
    tensor = tensor.unsqueeze(0)  # [1, 1, H, W]
    ds = F.interpolate(tensor, scale_factor=1/scale_factor, mode='bilinear', align_corners=False)
    return ds.squeeze(0)  # [1, H/s, W/s]

class CT_Dataset_SR(Dataset):
    def __init__(self, dicom_folder, window_center=40, window_width=400, scale_factor=2, max_slices=None):
        self.paths = find_dicom_files_recursively(dicom_folder)
        if max_slices:
            self.paths = self.paths[:max_slices]
        self.wc = window_center
        self.ww = window_width
        self.scale = scale_factor
        print(f"[CT-Loader] Dataset ready: {len(self.paths)} slices | scale={self.scale} | window=({self.wc},{self.ww})")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        hr = load_dicom_as_tensor(self.paths[idx], self.wc, self.ww)
        lr = downsample_tensor(hr, self.scale)
        return lr, hr

# Selbst-test-Beispiel:
if __name__ == "__main__":
    dataset = CT_Dataset_SR(
        r"C:\AA_Leonard\A_Studium\Bachelorarbeit Superresolution\ESRGAN-Med\data\manifest-1724965242274\Spine-Mets-CT-SEG\10352\12-03-2011-NA-SpineSPINEBONESBRT Adult-55418\4.000000-SKINTOSKINSIM0.5MM10352a iMAR-32611",
        max_slices=20  # wir nehmen erstmal nur 20 Slices
    )
    print(f"Anzahl Bilder: {len(dataset)}")
    lr, hr = dataset[0]
    print(f"LR-Shape: {lr.shape}, HR-Shape: {hr.shape}")

