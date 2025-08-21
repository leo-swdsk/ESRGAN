import os
import torch
import pydicom
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import random
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
        hr = load_dicom_as_tensor(self.paths[idx], self.wc, self.ww)   # [1, H, W]
        lr = downsample_tensor(hr, self.scale)                         # [1, H/s, W/s]

        # zufälliger, ausgerichteter Crop (z. B. 128 HR-Pixel)
        lr, hr = random_aligned_crop(hr, lr, hr_patch=128, scale=self.scale)
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

