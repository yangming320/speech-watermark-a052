from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import validate_audio
from .keys import validate_key
from .spectral import SpectralConfig, analyze, cepstral_filter, log_magnitude


@dataclass
class FeatureResult:
    features: np.ndarray
    times_s: np.ndarray
    valid_mask: np.ndarray
    frequency_hz: np.ndarray
    metadata: dict


def vad_mask(audio: np.ndarray, sr: int, times_s: np.ndarray,
             aggressiveness: int = 2) -> np.ndarray:
    import webrtcvad

    if not 0 <= aggressiveness <= 3:
        raise ValueError("Invalid WebRTC VAD aggressiveness")
    vad = webrtcvad.Vad(aggressiveness)
    frame_size = sr * 20 // 1000
    count = len(audio) // frame_size
    flags = np.zeros(count, dtype=bool)
    pcm = np.rint(np.clip(audio, -1, 32767/32768)*32768).astype("<i2")
    for i in range(count):
        flags[i] = vad.is_speech(pcm[i*frame_size:(i+1)*frame_size].tobytes(), sr)
    # STFT centers occur at multiples of a hop; integer samples avoid float boundary drift.
    center_samples = np.rint(times_s*sr).astype(int)
    intervals = center_samples // frame_size
    mask = (intervals >= 0) & (intervals < count)
    mask[mask] &= flags[intervals[mask]]
    return mask


def extract_features(audio: np.ndarray, sr: int, key: np.ndarray,
                     config: SpectralConfig | None = None,
                     vad: str = "webrtc", vad_aggressiveness: int = 2) -> FeatureResult:
    cfg = config or SpectralConfig()
    audio = validate_audio(audio, sr)
    key = validate_key(key, cfg.expected_bins)
    spectrum, times = analyze(audio, sr, cfg)
    indices, frequencies = cfg.indices(sr)
    filtered = cepstral_filter(log_magnitude(spectrum), sr, cfg)
    features = (filtered[indices] * key[:, None]).T
    window, _ = cfg.lengths(sr)
    centers = np.rint(times*sr).astype(int)
    valid = (centers-window//2 >= 0) & (centers+window//2 <= len(audio))
    if vad == "webrtc":
        valid &= vad_mask(audio, sr, times, vad_aggressiveness)
    elif vad != "none":
        raise ValueError(f"Unknown VAD mode: {vad}")
    return FeatureResult(features, times, valid, frequencies,
                         {"total_frames": len(times), "valid_frames": int(valid.sum()),
                          "vad": vad, "vad_aggressiveness": vad_aggressiveness,
                          "sample_rate": sr})


def score_frames(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[1] != 63 or not np.isfinite(features).all():
        raise ValueError("Expected finite [T,63] features")
    return features.mean(axis=1)


def aggregate(scores: np.ndarray, times_s: np.ndarray, frames: int) -> list[dict]:
    scores = np.asarray(scores)
    if frames < 1 or len(scores) != len(times_s):
        raise ValueError("Invalid aggregation arguments")
    groups = []
    for start in range(0, len(scores)-frames+1, frames):
        stop = start+frames
        groups.append({"group_index": len(groups), "start_frame_in_valid": start,
                       "frames": frames, "start_s": float(times_s[start]),
                       "end_s": float(times_s[stop-1]),
                       "score": float(scores[start:stop].mean())})
    return groups

