"""
MRI neuroimaging processor for NeuroSense AI (OASIS-2 dataset).
Loads NIfTI volumes (.hdr/.img pairs), extracts slice projections,
and generates structural brain atrophy heatmaps.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image


class MRIProcessor:
    def __init__(self, mri_base_dir: str | Path = "data/raw/oasis2_mri"):
        self.mri_base_dir = Path(mri_base_dir)

    def find_scan(self, mri_id: str) -> Path | None:
        """Locates the first available .nifti.hdr file for a given MRI ID."""
        session_dir = self.mri_base_dir / mri_id / "RAW"
        if not session_dir.exists():
            return None
        hdr_files = sorted(session_dir.glob("*.nifti.hdr"))
        return hdr_files[0] if hdr_files else None

    def load_volume(self, hdr_path: Path | str) -> np.ndarray:
        """Loads 3D/4D MRI volume and returns normalized 3D float array."""
        img = nib.load(str(hdr_path))
        data = img.get_fdata()
        if data.ndim == 4:
            data = data[:, :, :, 0]
        # Min-max intensity normalization (clipping top 1% outliers)
        p99 = np.percentile(data, 99.5)
        p1 = np.percentile(data, 1.0)
        data = np.clip(data, p1, p99)
        data = (data - p1) / (p99 - p1 + 1e-6)
        return data.astype(np.float32)

    def extract_axial_slice(self, volume: np.ndarray, slice_idx: int | None = None) -> np.ndarray:
        """Extracts the middle axial slice showing ventricles & hippocampal area."""
        if slice_idx is None:
            slice_idx = volume.shape[2] // 2
        slice_2d = volume[:, :, slice_idx]
        # Rotate 90 degrees for standard radiological orientation
        return np.rot90(slice_2d)

    def generate_atrophy_overlay(self, axial_slice: np.ndarray) -> np.ndarray:
        """
        Generates an Explainable AI visual heatmap highlighting regions of
        cerebrospinal fluid (CSF) expansion and cortical/ventricular atrophy.
        """
        # Threshold for CSF / enlarged ventricular spaces (typically darker voxels inside brain mask)
        brain_mask = axial_slice > 0.05
        # Inverted intensity inside the brain indicates ventricle/sulci enlargement
        atrophy_intensity = np.where(brain_mask, (1.0 - axial_slice) * brain_mask, 0.0)
        # Suppress noise
        atrophy_intensity = np.clip((atrophy_intensity - 0.45) / 0.55, 0.0, 1.0)
        return atrophy_intensity
