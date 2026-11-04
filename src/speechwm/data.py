from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf

from .audio import prepare_audio, read_audio, write_audio
from .utils import sha256_file, write_json, write_jsonl


def download_verified(url: str, destination: Path, expected_sha256: str) -> None:
    if destination.exists():
        if sha256_file(destination) != expected_sha256:
            raise ValueError(f"Existing source hash mismatch: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix+".partial")
    try:
        request = Request(url, headers={"User-Agent": "speechwm-reproduction/0.1"})
        with urlopen(request, timeout=40) as response, temporary.open("wb") as output:
            total = 0
            while chunk := response.read(65536):
                total += len(chunk)
                if total > 10*1024*1024:
                    raise ValueError("Source exceeds the 10 MB per-file download bound")
                output.write(chunk)
        if sha256_file(temporary) != expected_sha256:
            raise ValueError(f"Downloaded source hash mismatch: {url}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_public(root: Path, sources_path: Path, segments: int = 20,
                   target_sr: int = 16000, peak_limit: float = 0.5) -> dict:
    start = time.perf_counter()
    sources = json.loads(sources_path.read_text())
    if segments < len(sources["sources"]):
        raise ValueError("Need at least one segment per source")
    rows, parents = [], []
    for source_i, record in enumerate(sources["sources"]):
        uid = record["utterance_id"]
        source_path = root/"data/raw/public"/(uid+".hq.ogg")
        info_path = root/"data/raw/public"/(uid+".txt")
        download_verified(record["audio_url"], source_path, record["audio_sha256"])
        download_verified(record["info_url"], info_path, record["info_sha256"])
        original, sr = read_audio(source_path)
        audio, prep = prepare_audio(original, sr, target_sr, peak_limit)
        source_info = sf.info(source_path)
        count = segments//len(sources["sources"])+(source_i < segments%len(sources["sources"]))
        edges = np.linspace(0, len(audio), count+1, dtype=int)
        for i, (left, right) in enumerate(zip(edges[:-1],edges[1:])):
            segment_id = f"{uid}_seg{i:02d}"
            dest = root/"data/processed/smoke"/(segment_id+".wav")
            write_audio(dest, audio[left:right], target_sr)
            rows.append({
                "utterance_id": segment_id, "parent_utterance_id": uid,
                "speaker_id": record["speaker_id"], "split": "smoke",
                "source_path": str(source_path.relative_to(root)),
                "source_sha256": record["audio_sha256"],
                "waveform_path": str(dest.relative_to(root)), "waveform_sha256": sha256_file(dest),
                "sample_rate": target_sr, "duration_s": (right-left)/target_sr,
                "channels": 1, "source_sample_rate": sr, "source_format": source_info.format,
                "start_s_in_parent": left/target_sr, "end_s_in_parent": right/target_sr,
                "source_url": record["audio_url"], "dataset_id": sources["dataset_id"],
                "scope": "development_smoke_only", "preparation": prep,
            })
        parents.append({**record, "source_path": str(source_path.relative_to(root)),
                        "source_sample_rate": sr, "duration_s": len(original)/sr,
                        "source_format": source_info.format, "segments": int(count),
                        "source_info_path": str(info_path.relative_to(root))})
    write_jsonl(root/"data/manifests/smoke.jsonl", rows)
    summary = {
        "dataset_id": sources["dataset_id"], "scope": "development_smoke_only",
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "segments": len(rows), "parent_utterances": len(parents),
        "speakers": len({r["speaker_id"] for r in rows}),
        "duration_s": sum(r["duration_s"] for r in rows),
        "description": "20 non-overlapping segments from 3 public utterances; not 20 independent utterances",
        "source_compression": "Downloaded HQ Ogg examples; not original lossless LibriSpeech or TIMIT",
        "partitions": ["smoke"], "parents": parents,
        "elapsed_s": time.perf_counter()-start,
    }
    write_json(root/"data/manifests/data_summary.json", summary)
    return summary

