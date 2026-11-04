from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def validate_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not audio.size:
        raise ValueError("Expected non-empty mono audio")
    if sr <= 0 or not np.isfinite(audio).all():
        raise ValueError("Invalid sample rate or non-finite waveform")
    return audio


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float64", always_2d=True)
    return validate_audio(audio.mean(axis=1), sr), int(sr)


def write_audio(path: str | Path, audio: np.ndarray, sr: int, subtype: str = "FLOAT") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, validate_audio(audio, sr), sr, subtype=subtype, format="WAV")


def resample_audio(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    audio = validate_audio(audio, sr)
    if target_sr <= 0:
        raise ValueError("Invalid target sample rate")
    if sr == target_sr:
        return audio.copy()
    divisor = math.gcd(sr, target_sr)
    return resample_poly(audio, target_sr // divisor, sr // divisor)


def prepare_audio(audio: np.ndarray, sr: int, target_sr: int = 16000,
                  peak_limit: float = 0.5) -> tuple[np.ndarray, dict]:
    if not 0 < peak_limit < 1:
        raise ValueError("peak_limit must be in (0,1)")
    audio = resample_audio(audio, sr, target_sr)
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-10:
        raise ValueError("Silent or near-silent source audio")
    gain = min(1.0, peak_limit / peak)
    return audio * gain, {"input_peak": peak, "gain": gain, "sample_rate": target_sr}

