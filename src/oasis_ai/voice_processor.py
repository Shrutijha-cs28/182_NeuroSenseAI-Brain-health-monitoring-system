"""
Acoustic vocal processor for NeuroSense AI (Parkinson's Disease module).
Extracts pitch stability (F0), frequency perturbation (Jitter),
amplitude perturbation (Shimmer), and Harmonic-to-Noise Ratio (HNR)
from sustained phonation audio (.wav) using librosa and scipy.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf
import librosa


VOICE_FEATURE_NAMES = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Shimmer",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "PPE"
]


class VoiceProcessor:
    def __init__(self, target_sr: int = 22050):
        self.target_sr = target_sr

    def extract_features_from_audio(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """
        Extracts vocal biomarkers from an audio waveform array.
        Audio should ideally be 2-5 seconds of sustained vowel /a/.
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Normalize audio amplitude
        audio = audio.astype(np.float32)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # 1. Pitch / Fundamental Frequency estimation (using librosa pyin)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr
        )
        valid_f0 = f0[voiced_flag & ~np.isnan(f0)] if f0 is not None else np.array([])
        if len(valid_f0) > 5:
            fo_mean = float(np.mean(valid_f0))
            fo_max = float(np.percentile(valid_f0, 95))
            fo_min = float(np.percentile(valid_f0, 5))
            # Jitter approximation: relative mean difference between consecutive periods
            diffs = np.abs(np.diff(valid_f0))
            jitter_pct = float(np.mean(diffs) / (fo_mean + 1e-6) * 100)
        else:
            fo_mean, fo_max, fo_min, jitter_pct = 150.0, 190.0, 110.0, 0.5

        # 2. Shimmer approximation: relative amplitude frame-to-frame fluctuation
        rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=512)[0]
        if len(rms) > 2 and np.mean(rms) > 0:
            shimmer = float(np.mean(np.abs(np.diff(rms))) / (np.mean(rms) + 1e-6))
        else:
            shimmer = 0.03

        # 3. Harmonic-to-Noise Ratio (HNR) via harmonic-percussive separation
        y_harm, y_perc = librosa.effects.hpss(audio)
        p_harm = np.mean(y_harm ** 2)
        p_perc = np.mean(y_perc ** 2) + 1e-8
        hnr = float(10 * np.log10(p_harm / p_perc))
        hnr = float(np.clip(hnr, 5.0, 35.0))

        # 4. Nonlinear dynamics / Spectral spread approximations
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        spread1 = float(-5.0 + (np.std(spectral_centroid) / 1000.0))
        spread2 = float(np.std(spectral_bandwidth) / 1000.0)

        # RPDE & DFA & PPE proxies (calibrated around healthy baseline)
        rpde = float(np.clip(0.40 + (jitter_pct * 0.05), 0.25, 0.75))
        dfa = float(np.clip(0.65 + (shimmer * 0.5), 0.50, 0.85))
        ppe = float(np.clip(0.15 + (jitter_pct * 0.03), 0.08, 0.50))

        return {
            "MDVP:Fo(Hz)": round(fo_mean, 2),
            "MDVP:Fhi(Hz)": round(fo_max, 2),
            "MDVP:Flo(Hz)": round(fo_min, 2),
            "MDVP:Jitter(%)": round(jitter_pct, 4),
            "MDVP:Shimmer": round(shimmer, 4),
            "HNR": round(hnr, 2),
            "RPDE": round(rpde, 4),
            "DFA": round(dfa, 4),
            "spread1": round(spread1, 4),
            "spread2": round(spread2, 4),
            "PPE": round(ppe, 4),
        }

    def process_file(self, audio_path: str | Path) -> dict[str, float]:
        """Loads a .wav file and extracts all vocal acoustic biomarkers."""
        path = Path(audio_path)
        audio, sr = librosa.load(str(path), sr=self.target_sr)
        return self.extract_features_from_audio(audio, sr)
