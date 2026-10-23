"""
Biomechanical gait & kinematic processor for NeuroSense AI (Parkinson's Disease module).
Extracts vertical ground reaction force (VGRF) features, stride variability,
bilateral asymmetry, stepping cadence, and oscillatory tremor band power.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import signal


GAIT_FEATURE_NAMES = [
    "mean_force",
    "force_std",
    "stride_cv",
    "gait_asymmetry",
    "cadence_hz",
    "tremor_band_power",
    "left_right_ratio",
    "force_peak_to_mean"
]


class GaitProcessor:
    def __init__(self, sampling_rate: float = 100.0):
        self.sampling_rate = sampling_rate

    def extract_features_from_array(self, vgrf_data: np.ndarray) -> dict[str, float]:
        """
        Extracts gait biomarkers from a 2D array of sensor measurements over time.
        Expected shape: (N_timesteps, 18) or (N_timesteps, 16+).
        """
        if vgrf_data.ndim == 1:
            total_force = vgrf_data
            left_total = vgrf_data * 0.5
            right_total = vgrf_data * 0.5
        elif vgrf_data.shape[1] >= 18:
            left_total = vgrf_data[:, 16]
            right_total = vgrf_data[:, 17]
            total_force = left_total + right_total
        elif vgrf_data.shape[1] >= 16:
            left_total = vgrf_data[:, :8].sum(axis=1)
            right_total = vgrf_data[:, 8:16].sum(axis=1)
            total_force = left_total + right_total
        else:
            total_force = vgrf_data.sum(axis=1)
            left_total = total_force * 0.5
            right_total = total_force * 0.5

        # 1. Total force statistics
        mean_force = float(np.mean(total_force))
        force_std = float(np.std(total_force))
        # Stride Coefficient of Variation (CV) - key PD instability marker
        stride_cv = float(force_std / (mean_force + 1e-6))
        peak_to_mean = float(np.max(total_force) / (mean_force + 1e-6))

        # 2. Bilateral Gait Asymmetry Index (|Left - Right| / (Left + Right))
        asym = float(np.mean(np.abs(left_total - right_total) / (left_total + right_total + 1e-6)))
        left_right_ratio = float(np.mean(left_total) / (np.mean(right_total) + 1e-6))

        # 3. Spectral analysis (Cadence / Stride Frequency)
        nperseg = min(256, len(total_force))
        freqs, psd = signal.welch(total_force, fs=self.sampling_rate, nperseg=nperseg)
        # Exclude DC component (freq=0)
        valid_freq_mask = freqs > 0.5
        if np.any(valid_freq_mask):
            dominant_idx = np.argmax(psd[valid_freq_mask])
            cadence_hz = float(freqs[valid_freq_mask][dominant_idx])
        else:
            cadence_hz = 1.6

        # 4. Parkinsonian Tremor & Freezing of Gait (FoG) band energy (3.0 Hz - 8.0 Hz)
        tremor_mask = (freqs >= 3.0) & (freqs <= 8.0)
        tremor_power = float(np.sum(psd[tremor_mask]) / (np.sum(psd) + 1e-6))

        return {
            "mean_force": round(mean_force, 4),
            "force_std": round(force_std, 4),
            "stride_cv": round(stride_cv, 4),
            "gait_asymmetry": round(asym, 4),
            "cadence_hz": round(cadence_hz, 4),
            "tremor_band_power": round(tremor_power, 4),
            "left_right_ratio": round(left_right_ratio, 4),
            "force_peak_to_mean": round(peak_to_mean, 4),
        }

    def process_file(self, file_path: str | Path) -> dict[str, Any]:
        """Loads a single .txt force file and extracts all biomarkers."""
        path = Path(file_path)
        data = np.loadtxt(path, delimiter=",")
        features = self.extract_features_from_array(data)
        
        # Subject and target metadata from filename (e.g. GaCo01_01.txt, JuPt04_02.txt)
        stem = path.stem
        subject_id = stem.split("_")[0] if "_" in stem else stem
        is_pd = 1 if "Pt" in subject_id else 0

        return {
            "filename": path.name,
            "subject_id": subject_id,
            "target": is_pd,
            **features,
        }

    def process_directory(self, dir_path: str | Path) -> pd.DataFrame:
        """Batch processes an entire directory of force .txt files into a DataFrame."""
        path = Path(dir_path)
        files = sorted(list(path.glob("*.txt")))
        records = []
        for f in files:
            try:
                records.append(self.process_file(f))
            except Exception:
                continue
        return pd.DataFrame(records)
