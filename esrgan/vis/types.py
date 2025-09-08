from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SliceMeta:
    path: str
    instance_number: Optional[int]
    sop_instance_uid: str
    subindex: Optional[int] = None


@dataclass
class SeriesInfo:
    pixel_spacing: Tuple[Optional[float], Optional[float]]
    slice_thickness: Optional[float]
    patient_id: str


@dataclass
class WindowCfg:
    center: float
    width: float


@dataclass
class DegUsed:
    blur_sigma: Optional[float]
    blur_kernel_k: Optional[int]
    noise_sigma: Optional[float]
    dose: Optional[float]


