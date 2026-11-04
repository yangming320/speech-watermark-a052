from __future__ import annotations

import numpy as np

from .keys import key_hash, validate_key
from .spectral import SpectralConfig, analyze, synthesize


def modify_spectrum(spectrum: np.ndarray, sr: int, key: np.ndarray,
                    delta_db: float, cfg: SpectralConfig,
                    weights: np.ndarray | None = None) -> np.ndarray:
    key = validate_key(key, cfg.expected_bins)
    if not np.isfinite(delta_db) or not 0 <= delta_db <= 20:
        raise ValueError("delta_db must be finite and in [0,20]")
    indices, _ = cfg.indices(sr)
    result = spectrum.copy()
    if weights is None:
        result[indices] *= np.power(10.0, delta_db*key.astype(float)/20)[:, None]
    else:
        weights=np.asarray(weights,dtype=float)
        if weights.shape!=(len(indices),spectrum.shape[1]) or not np.isfinite(weights).all() or np.any((weights<0)|(weights>1)):
            raise ValueError("Weights must be finite [63,T] values in [0,1]")
        result[indices] *= np.power(10.0,delta_db*key[:,None]*weights/20)
    return result


def embed(audio: np.ndarray, sr: int, key: np.ndarray, delta_db: float,
          config: SpectralConfig | None = None) -> tuple[np.ndarray, dict]:
    cfg = config or SpectralConfig()
    spectrum, _ = analyze(audio, sr, cfg)
    modified = modify_spectrum(spectrum, sr, key, delta_db, cfg)
    output = synthesize(modified, sr, len(audio), cfg)
    _, frequencies = cfg.indices(sr)
    return output, {"embedder": "SS", "delta_db": float(delta_db),
                    "sample_rate": sr, "key_sha256": key_hash(key),
                    "frequency_hz": frequencies.tolist(),
                    "peak": float(np.max(np.abs(output)))}


def embed_lpc(audio: np.ndarray, sr: int, key: np.ndarray, delta_db: float,
              config: SpectralConfig | None = None, lpc_config=None) -> tuple[np.ndarray,dict]:
    from .lpc import analyze_lpc

    cfg=config or SpectralConfig()
    profile=analyze_lpc(audio,sr,cfg,lpc_config)
    spectrum,times=analyze(audio,sr,cfg)
    # Compare sample centers; SciPy's boundary time subtraction can differ by <1e-15 s.
    np.testing.assert_array_equal(np.rint(times*sr),np.rint(profile.times_s*sr))
    output=synthesize(modify_spectrum(spectrum,sr,key,delta_db,cfg,profile.weights),sr,len(audio),cfg)
    return output,{"embedder":"SS_LPC","delta_db":float(delta_db),"sample_rate":sr,
                   "key_sha256":key_hash(key),"lpc":profile.metadata,
                   "peak":float(np.max(np.abs(output)))}

