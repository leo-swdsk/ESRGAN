import pydicom
import numpy as np
from pydicom.dataset import Dataset, FileDataset
import datetime

def tensor_to_dicom(tensor, ref_dicom_path, output_path, window_center=40, window_width=400):
    """
    tensor: 2D-Tensor mit Shape [1, H, W] (float in [-1, 1])
    ref_dicom_path: Original-DICOM, das als Vorlage dient
    output_path: Speicherort für neue DICOM
    """

    # 1. Denormalisieren auf HU-Fenster
    min_hu = window_center - window_width / 2
    max_hu = window_center + window_width / 2
    np_img = tensor.squeeze(0).cpu().numpy()  # [H, W]
    np_img = (np_img + 1) / 2  # [-1,1] → [0,1]
    np_img = np_img * (max_hu - min_hu) + min_hu  # → HU

    # 2. Optional: cast to int16 (DICOM-typisch)
    np_img = np.clip(np_img, -1024, 3071).astype(np.int16)

    # 3. DICOM-Vorlage laden und kopieren
    ref = pydicom.dcmread(ref_dicom_path)
    new_ds = ref
    new_ds.PixelData = np_img.tobytes()
    new_ds.Rows, new_ds.Columns = np_img.shape
    new_ds.BitsStored = 16
    new_ds.BitsAllocated = 16
    new_ds.SamplesPerPixel = 1
    new_ds.HighBit = 15
    new_ds.PixelRepresentation = 1  # signed int
    new_ds.InstanceCreationDate = datetime.datetime.now().strftime('%Y%m%d')
    new_ds.InstanceCreationTime = datetime.datetime.now().strftime('%H%M%S')
    new_ds.save_as(output_path)


#Beispielverwendung
# Annahme: prediction = [1, 512, 512]
#tensor = prediction.detach().cpu()
#tensor_to_dicom(tensor, ref_dicom_path="original.dcm", output_path="sr_result.dcm")
