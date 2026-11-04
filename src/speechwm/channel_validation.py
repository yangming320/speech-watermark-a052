"""Audit a completed M2 run from saved files rather than in-memory signals."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from .audio import read_audio
from .channels import inspect_opus
from .utils import canonical_hash, read_jsonl, sha256_file


def validate_run(out: Path) -> dict:
    root = next(p for p in out.resolve().parents if (p/"pyproject.toml").is_file())
    config = yaml.safe_load((out/"config.yaml").read_text())
    metrics = json.loads((out/"metrics.json").read_text())
    assert canonical_hash(config) == metrics["config_hash"], "Config hash mismatch"
    assert sha256_file(out/"input_manifest.jsonl") == metrics["input_manifest_sha256"], "Input manifest hash mismatch"
    channels = read_jsonl(out/"channel_manifest.jsonl")
    artifacts = read_jsonl(out/"audio_manifest.jsonl")
    failures = read_jsonl(out/"failures.jsonl")
    expected = metrics["segments"] * len(config["evaluation"]["deltas_db"]) * 2 * len(config["evaluation"]["channels"])
    assert len(channels) == len(artifacts) == expected, "Incomplete channel matrix"
    by_key = {(r["utterance_id"],r["delta_db"],r["label"],r["channel"]):r for r in channels}
    artifact_index = {(r["utterance_id"],r["delta_db"],r["label"],r["channel"]):r for r in artifacts}
    assert len(by_key) == expected, "Duplicate channel condition"
    identical, opus_count, payload_rates, pre_skips, file_rates = 0, 0, [], [], []
    noise_groups, noise_ranges = defaultdict(list), defaultdict(list)
    for item in artifacts:
        key = (item["utterance_id"],item["delta_db"],item["label"],item["channel"])
        meta = by_key[key]
        path = root/item["waveform_path"]
        assert sha256_file(path) == item["waveform_sha256"] == meta["waveform_sha256"]
        x, sr = read_audio(path)
        assert sr == item["sample_rate"] == meta["output_sample_rate"]
        assert len(x) == meta["output_samples"] and abs(x).max() < 1
        assert len(x) == (meta["input_samples"] if sr == 16000 else (meta["input_samples"]+1)//2)
        if item["delta_db"] == 0 and item["label"] == 1:
            other = artifact_index[(item["utterance_id"],0,0,item["channel"])]
            other_pcm, other_sr = read_audio(root/other["waveform_path"])
            # libsndfile FLOAT WAV PEAK chunks may contain creation timestamps.
            assert other_sr == sr and np.array_equal(other_pcm,x), "Delta-zero PCM label cue"
            identical += 1
        if "nominal_snr_db" in meta:
            assert abs(meta["actual_reference_snr_pre_db"]-meta["nominal_snr_db"]) < 1e-9
            noise_groups[(item["utterance_id"],item["channel"])].append(meta["noise_sha256"])
            noise_ranges[(item["channel"],"pre")].append(meta["actual_reference_snr_pre_db"])
            noise_ranges[(item["channel"],"post")].append(meta["actual_reference_snr_post_db"])
        if item["channel"] == "opus8k_8kbps":
            compressed = path.with_suffix(".opus")
            assert sha256_file(compressed) == meta["compressed_sha256"]
            inspection = inspect_opus(compressed)
            assert inspection == meta["inspection"]
            assert inspection["packet_sizes_bytes"] == [20] and inspection["packet_duration_ms"] == [20]
            assert inspection["bandwidth_counts"].keys() == {"NB"}
            payload_rates.append(inspection["payload_bitrate_bps"])
            file_rates.append(inspection["file_bitrate_per_pcm_second_bps"])
            pre_skips.append(inspection["pre_skip_ms"])
            opus_count += 1
    assert all(len(set(values)) == 1 for values in noise_groups.values()), "Unpaired noise across labels/strengths"
    assert not failures, "Inspect failures.jsonl before accepting milestone"
    null_rows = [r for r in metrics["metrics"] if r["delta_db"] == 0]
    assert len(null_rows) == len(config["evaluation"]["channels"])*len(config["evaluation"]["aggregate_frames"])
    assert all(r["eer"] == .5 for r in null_rows)
    coverage = list(csv.DictReader((out/"coverage.csv").open()))
    assert len(coverage) == expected and all(int(r["valid_frames"]) >= 30 for r in coverage)
    ranges = {f"{channel}_{stage}_db": {"min":min(v),"max":max(v)} for (channel,stage),v in noise_ranges.items()}
    m1 = root/"outputs/E01_public_smoke_v1/baseline-v1/metrics.json"
    if not m1.exists():
        m1 = root/"milestones/M1/evidence/metrics.json"
    regression = {"available":m1.exists(), "scope":"same frozen M1 clean/downsample settings"}
    if m1.exists():
        old = json.loads(m1.read_text())["metrics"]
        index = {(r["delta_db"],r["channel"],r["detector"],r["frames"]):r for r in metrics["metrics"]}
        assert all(index[(r["delta_db"],r["channel"],r["detector"],r["frames"])] == r for r in old)
        regression.update({"passed":True,"metric_rows_identical":len(old)})
    return {"passed":True, "expected_channel_outputs":expected, "received_wav_hashes_verified":len(artifacts),
            "delta_zero_identical_pcm_pairs":identical, "delta_zero_eer_settings":len(null_rows),
            "pair_identity_basis":"decoded PCM values; whole-file hashes include possible WAV PEAK timestamps",
            "noise_groups_shared_across_labels_and_strengths":len(noise_groups),
            "reference_snr_ranges":ranges, "opus_files_verified":opus_count,
            "opus_payload_rate_bps_range":[min(payload_rates),max(payload_rates)],
            "opus_file_rate_per_pcm_second_bps_range":[min(file_rates),max(file_rates)],
            "opus_pre_skip_ms":sorted(set(pre_skips)), "all_lengths_and_rates_verified":True,
            "all_outputs_peak_below_one":True, "config_and_input_manifest_hashes_verified":True,
            "minimum_valid_frames":min(int(r["valid_frames"]) for r in coverage),
            "failures":len(failures), "m1_regression":regression,
            "scope":"controlled public-data development verification only"}
