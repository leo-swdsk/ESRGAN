import os
import numpy as np
import torch
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from .types import SliceMeta, SeriesInfo
from ..ct_dataset_loader import is_ct_image_dicom


def _slice_order_key(path: str):
    try:
        ds_hdr = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        ipp = getattr(ds_hdr, 'ImagePositionPatient', None)
        iop = getattr(ds_hdr, 'ImageOrientationPatient', None)
        inst = getattr(ds_hdr, 'InstanceNumber', None)
        sloc = getattr(ds_hdr, 'SliceLocation', None)
        if iop is not None and len(iop) >= 6 and ipp is not None and len(ipp) >= 3:
            r = np.array([float(iop[0]), float(iop[1]), float(iop[2])], dtype=np.float64)
            c = np.array([float(iop[3]), float(iop[4]), float(iop[5])], dtype=np.float64)
            n = np.cross(r, c)
            p = np.array([float(ipp[0]), float(ipp[1]), float(ipp[2])], dtype=np.float64)
            zproj = float(np.dot(p, n))
            return (0, zproj, 0 if inst is None else int(inst))
        if ipp is not None and len(ipp) >= 3:
            return (1, float(ipp[2]), 0 if inst is None else int(inst))
        if sloc is not None:
            return (2, float(sloc), 0 if inst is None else int(inst))
        if inst is not None:
            return (3, float(int(inst)), 0)
    except Exception:
        pass
    return (4, 0.0, 0.0)


def load_ct_volume_hu(folder_path: str):
    # gather candidates
    cand_paths = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if not f.lower().endswith('.dcm'):
                continue
            path = os.path.join(root, f)
            if is_ct_image_dicom(path):
                cand_paths.append(path)
    if len(cand_paths) == 0:
        raise RuntimeError(f"No CT image DICOM files found under {folder_path}")

    slice_paths = sorted(cand_paths, key=_slice_order_key)

    slice_list = []
    meta_list = []
    for path in slice_paths:
        try:
            ds = pydicom.dcmread(path, force=True)
            arr = ds.pixel_array
            hu = apply_modality_lut(arr, ds).astype(np.float32)
            if hu.ndim == 2:
                slice_list.append(torch.tensor(hu).unsqueeze(0))
                meta_list.append(SliceMeta(
                    path=path,
                    instance_number=getattr(ds, 'InstanceNumber', None),
                    sop_instance_uid=str(getattr(ds, 'SOPInstanceUID', ''))
                ))
            elif hu.ndim == 3:
                for k in range(hu.shape[0]):
                    slice_list.append(torch.tensor(hu[k]).unsqueeze(0))
                    meta_list.append(SliceMeta(
                        path=path,
                        instance_number=getattr(ds, 'InstanceNumber', None),
                        sop_instance_uid=str(getattr(ds, 'SOPInstanceUID', '')),
                        subindex=k
                    ))
        except Exception:
            continue
    if len(slice_list) == 0:
        raise RuntimeError(f"No CT image DICOM files found under {folder_path}")
    H, W = slice_list[0].shape[-2:]
    filtered = [(s, m) for s, m in zip(slice_list, meta_list) if s.shape[-2:] == (H, W)]
    if len(filtered) == 0:
        raise RuntimeError("No slices with consistent dimensions found")
    slice_list, meta_list = zip(*filtered)
    vol = torch.stack(list(slice_list), dim=0)
    return vol, list(meta_list)


def read_series_metadata(folder_path: str) -> SeriesInfo:
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
                break
            except Exception:
                continue
    return SeriesInfo(pixel_spacing=(row_mm, col_mm), slice_thickness=slice_thickness, patient_id=patient_id)


