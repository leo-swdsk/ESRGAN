# load_ct_volume.py

import os
from load_ct_tensor import load_ct_as_tensor

def load_folder_as_tensor_stack(folder_path, preset="soft_tissue"):
    tensor_stack = []

    for fname in sorted(os.listdir(folder_path)):
        if fname.endswith(".dcm"):
            path = os.path.join(folder_path, fname)
            tensor = load_ct_as_tensor(path, preset)
            tensor_stack.append(tensor)

    # Ausgabe: Tensor [Num_Slices, 1, H, W]
    volume = torch.stack(tensor_stack)
    return volume


# So sieht zum Beispiel der Aufruf aus:
# tensor = load_ct_as_tensor("path/to/file.dcm", preset="bone")
# volume = load_folder_as_tensor_stack("path/to/folder", preset="lung")

def find_dicom_files_recursively(base_folder):
    dicom_files = []
    for root, _, files in os.walk(base_folder):
        for f in files:
            if f.lower().endswith('.dcm'):
                dicom_files.append(os.path.join(root, f))
    return sorted(dicom_files)
