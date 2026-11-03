"""Dataset discovery, integrity checks, duplicate detection and leakage-free splits (M3-M4)."""

from .aptos import AptosIndex, find_aptos
from .drive import DriveSample, find_drive_samples, load_binary, load_rgb
from .idrid import IdridIndex, disc_diameter_px, find_idrid
from .records import DR_GRADE_NAMES, REFERABLE_MIN_GRADE, ImageRecord
from .rite import RiteIndex, decode_av, find_rite

__all__ = [
    "AptosIndex", "DR_GRADE_NAMES", "DriveSample", "IdridIndex", "ImageRecord", "REFERABLE_MIN_GRADE",
    "RiteIndex", "decode_av", "disc_diameter_px", "find_aptos", "find_drive_samples", "find_idrid",
    "find_rite", "load_binary", "load_rgb",
]
