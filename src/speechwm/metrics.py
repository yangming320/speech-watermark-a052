from __future__ import annotations

import numpy as np


def detection_curve(labels: np.ndarray, scores: np.ndarray) -> dict[str, np.ndarray]:
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("Labels and scores must be matching non-empty vectors")
    if not np.isin(labels, [0,1]).all() or np.unique(labels).size != 2:
        raise ValueError("Both binary classes are required")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite score")
    order = np.argsort(-scores, kind="stable")
    y, values = labels[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(values)), len(values)-1]
    tp = np.r_[0, np.cumsum(y)[ends]]
    fp = np.r_[0, np.cumsum(1-y)[ends]]
    # nextafter supplies a finite threshold that predicts no positives.
    thresholds = np.r_[np.nextafter(values[0], np.inf), values[ends]]
    return {"threshold": thresholds, "far": fp/np.sum(labels==0),
            "frr": 1-tp/np.sum(labels==1)}


def equal_error_rate(labels: np.ndarray, scores: np.ndarray) -> dict:
    curve = detection_curve(labels, scores)
    diff = curve["far"]-curve["frr"]
    right = int(np.flatnonzero(diff >= 0)[0])
    if right == 0 or diff[right] == 0:
        weight, left = 0.0, right
    else:
        left = right-1
        weight = float(-diff[left]/(diff[right]-diff[left]))
    eer = float(curve["far"][left]*(1-weight)+curve["far"][right]*weight)
    threshold = float(curve["threshold"][left]*(1-weight)+curve["threshold"][right]*weight)
    return {"eer": eer, "eer_threshold_interpolated": threshold,
            "threshold_role": "descriptive only; not calibrated for deployment",
            "samples": len(scores), "negative_samples": int(np.sum(np.asarray(labels)==0)),
            "positive_samples": int(np.sum(np.asarray(labels)==1))}


def waveform_snr(reference: np.ndarray, observed: np.ndarray) -> float | None:
    signal_power = float(np.mean(np.square(reference)))
    error_power = float(np.mean(np.square(observed-reference)))
    return None if error_power == 0 else float(10*np.log10(signal_power/error_power))


def quality_scores(reference: np.ndarray, observed: np.ndarray, sr: int,
                   compute_pesq: bool = True) -> dict:
    from .audio import resample_audio

    result = {"embedding_snr_db": waveform_snr(reference, observed),
              "pesq_nb": None, "pesq_wb": None, "quality_failures": []}
    if compute_pesq:
        from pesq import pesq
        for mode, target_sr in [("nb",8000),("wb",16000)]:
            try:
                ref = resample_audio(reference, sr, target_sr)
                deg = resample_audio(observed, sr, target_sr)
                value = float(pesq(target_sr, ref, deg, mode))
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid PESQ score: {value}")
                result[f"pesq_{mode}"] = value
            except Exception as exc:
                result["quality_failures"].append({"mode": mode, "error": repr(exc)})
    return result

