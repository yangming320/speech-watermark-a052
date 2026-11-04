from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import get_window, istft, stft

from .audio import validate_audio


@dataclass(frozen=True)
class SpectralConfig:
    frame_ms: float = 20.0
    hop_ms: float = 10.0
    low_hz: float = 500.0
    high_hz_exclusive: float = 3650.0
    expected_bins: int = 63
    cepstral_cutoff_ms: float = 0.5

    def lengths(self, sr: int) -> tuple[int, int]:
        window = int(round(sr * self.frame_ms / 1000))
        hop = int(round(sr * self.hop_ms / 1000))
        if window < 2 or window % 2 or not 0 < hop < window:
            raise ValueError("Require an even analysis window and 0 < hop < window")
        if not np.isclose(window * 1000 / sr, self.frame_ms):
            raise ValueError("Frame duration must be an integer number of samples")
        if not np.isclose(hop * 1000 / sr, self.hop_ms):
            raise ValueError("Hop duration must be an integer number of samples")
        if not 0 <= self.low_hz < self.high_hz_exclusive <= sr / 2:
            raise ValueError("Watermark band is outside the Nyquist interval")
        return window, hop

    def indices(self, sr: int) -> tuple[np.ndarray, np.ndarray]:
        window, _ = self.lengths(sr)
        freqs = np.fft.rfftfreq(window, 1 / sr)
        indices = np.flatnonzero((freqs >= self.low_hz) & (freqs < self.high_hz_exclusive))
        if indices.size != self.expected_bins:
            raise ValueError(f"Expected {self.expected_bins} watermark bins, got {indices.size}")
        return indices, freqs[indices]


def analyze(audio: np.ndarray, sr: int, cfg: SpectralConfig) -> tuple[np.ndarray, np.ndarray]:
    audio = validate_audio(audio, sr)
    window, hop = cfg.lengths(sr)
    if audio.size < window:
        raise ValueError("Audio is shorter than one analysis frame")
    _, times, spectrum = stft(
        audio, fs=sr, window=get_window("hann", window, fftbins=True),
        nperseg=window, noverlap=window-hop, nfft=window, detrend=False,
        return_onesided=True, boundary="zeros", padded=True, scaling="spectrum",
    )
    return spectrum, times


def synthesize(spectrum: np.ndarray, sr: int, length: int, cfg: SpectralConfig) -> np.ndarray:
    window, hop = cfg.lengths(sr)
    _, audio = istft(
        spectrum, fs=sr, window=get_window("hann", window, fftbins=True),
        nperseg=window, noverlap=window-hop, nfft=window, input_onesided=True,
        boundary=True, scaling="spectrum",
    )
    if audio.size < length:
        raise ValueError("ISTFT output is shorter than requested")
    return audio[:length]


def log_magnitude(spectrum: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    return 20 * np.log10(np.maximum(np.abs(spectrum), epsilon))


def cepstral_filter(log_spectrum: np.ndarray, sr: int, cfg: SpectralConfig) -> np.ndarray:
    if cfg.cepstral_cutoff_ms < 0:
        raise ValueError("Negative cepstral cutoff")
    if cfg.cepstral_cutoff_ms == 0:
        return log_spectrum.copy()
    window, _ = cfg.lengths(sr)
    if log_spectrum.shape[0] != window // 2 + 1:
        raise ValueError("Cepstral filtering requires the complete one-sided spectrum")
    cepstrum = np.fft.irfft(log_spectrum, n=window, axis=0)
    quefrency_ms = np.minimum(np.arange(window), window-np.arange(window)) * 1000 / sr
    cepstrum[quefrency_ms <= cfg.cepstral_cutoff_ms, ...] = 0
    return np.fft.rfft(cepstrum, n=window, axis=0).real

