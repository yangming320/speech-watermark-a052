"""Re-read M3 waveforms, LPC profiles and scores before accepting a comparison."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from .audio import read_audio, resample_audio
from .channels import snr_db
from .comparison import choose_quality_pair, pcm_hash
from .features import aggregate, extract_features, score_frames
from .keys import key_hash, load_key
from .lpc import LPCConfig, analyze_lpc, gaussian_weights
from .metrics import equal_error_rate
from .spectral import SpectralConfig
from .utils import canonical_hash, read_jsonl, sha256_file


def validate_comparison(out: Path) -> dict:
    root=next(p for p in out.resolve().parents if (p/"pyproject.toml").exists())
    config=yaml.safe_load((out/"config.yaml").read_text());cfg=SpectralConfig(**config["spectral"])
    lp=LPCConfig(**config["lpc"]);summary=json.loads((out/"metrics.json").read_text())
    assert canonical_hash(config)==summary["config_hash"]
    assert sha256_file(out/"input_manifest.jsonl")==summary["input_manifest_sha256"]
    key=load_key(out/"key.json");sources=read_jsonl(out/"input_manifest.jsonl")
    assert key_hash(key)==summary["key_sha256"]
    rows=read_jsonl(out/"audio_manifest.jsonl");embedded=read_jsonl(out/"embedding_manifest.jsonl")
    channel_rows=read_jsonl(out/"channel_manifest.jsonl")
    artifact_index={(r["utterance_id"],r["method"],r["delta_db"],r["label"],r["channel"]):r for r in rows}
    embedding_index={(r["utterance_id"],r["method"],r["delta_db"],r["label"]):r for r in embedded}
    gains={r["utterance_id"]:float(r["common_gain"]) for r in csv.DictReader((out/"common_gains.csv").open())}
    references={r["utterance_id"]:read_audio(root/r["waveform_path"])[0]*gains[r["utterance_id"]] for r in sources}
    expected=len(sources)*len(config["evaluation"]["methods"])*len(config["evaluation"]["deltas_db"])*2*len(config["evaluation"]["channels"])
    assert len(rows)==len(channel_rows)==expected and len(embedded)*len(config["evaluation"]["channels"])==expected
    null_pairs,negative_groups,noise_groups=defaultdict(list),defaultdict(list),defaultdict(list)
    setting_ids=set();max_peak=0
    frame_stream=(out/"frame_scores.csv").open();group_stream=(out/"aggregate_scores.csv").open()
    frame_reader=csv.DictReader(frame_stream);group_reader=csv.DictReader(group_stream)
    frame_rows_verified=group_rows_verified=0
    for item in rows:
        setting=(item["utterance_id"],item["method"],item["delta_db"],item["channel"],item["label"])
        assert setting not in setting_ids
        setting_ids.add(setting)
        path=root/item["waveform_path"];x,sr=read_audio(path)
        assert sha256_file(path)==item["waveform_sha256"] and pcm_hash(x)==item["pcm_sha256"]
        assert len(x)==item["samples"] and sr==item["sample_rate"]
        assert sr==(16000 if item["channel"]=="clean" else 8000)
        assert len(x)==(len(references[item["utterance_id"]]) if sr==16000 else (len(references[item["utterance_id"]])+1)//2)
        max_peak=max(max_peak,float(abs(x).max()));assert abs(x).max()<1
        f=extract_features(x,sr,key,cfg,vad=config["evaluation"]["vad"]["mode"],
            vad_aggressiveness=config["evaluation"]["vad"]["aggressiveness"])
        assert pcm_hash(score_frames(f.features))==item["frame_scores_sha256"],"Saved WAV redetection differs"
        assert hashlib.sha256(f.valid_mask.tobytes()).hexdigest()==item["valid_mask_sha256"]
        for i,(score,stamp,valid) in enumerate(zip(score_frames(f.features),f.times_s,f.valid_mask)):
            saved=next(frame_reader)
            assert (saved["utterance_id"],saved["method"],float(saved["delta_db"]),saved["channel"],int(saved["label"]))==setting
            assert int(saved["frame_index"])==i and float(saved["score"])==score
            assert float(saved["time_s"])==stamp and bool(int(saved["valid"]))==valid
            frame_rows_verified+=1
        for count in config["evaluation"]["aggregate_frames"]:
            for group in aggregate(score_frames(f.features)[f.valid_mask],f.times_s[f.valid_mask],count):
                saved=next(group_reader)
                assert (saved["utterance_id"],saved["method"],float(saved["delta_db"]),saved["channel"],int(saved["label"]))==setting
                for field in ["group_index","start_frame_in_valid","frames"]:assert int(saved[field])==group[field]
                for field in ["score","start_s","end_s"]:assert float(saved[field])==group[field]
                group_rows_verified+=1
        if item["delta_db"]==0:null_pairs[(item["utterance_id"],item["method"],item["channel"])].append(item["pcm_sha256"])
        if item["label"]==0:negative_groups[(item["utterance_id"],item["channel"])].append(item["pcm_sha256"])
    assert next(frame_reader,None) is None and next(group_reader,None) is None
    frame_stream.close();group_stream.close()
    assert frame_rows_verified==summary["frame_score_rows"] and group_rows_verified==summary["aggregate_score_rows"]
    assert all(len(v)==2 and len(set(v))==1 for v in null_pairs.values())
    assert all(len(set(v))==1 for v in negative_groups.values()),"Negative class depends on method or strength"
    for item in embedded:
        path=root/item["waveform_path"];audio,_=read_audio(path)
        assert sha256_file(path)==item["waveform_sha256"] and pcm_hash(audio)==item["pcm_sha256"]
    snr_errors=[]
    for r in channel_rows:
        if "noise_sha256" in r:
            assert abs(r["actual_reference_snr_pre_db"]-r["nominal_snr_db"])<1e-9
            noise_groups[(r["utterance_id"],r["channel"])].append(r["noise_sha256"])
            e=embedding_index[(r["utterance_id"],r["method"],r["delta_db"],r["label"])]
            a=artifact_index[(r["utterance_id"],r["method"],r["delta_db"],r["label"],r["channel"])]
            before,_=read_audio(root/e["waveform_path"]);after,_=read_audio(root/a["waveform_path"])
            residual=after-resample_audio(before,16000,8000)
            measured=snr_db(resample_audio(references[r["utterance_id"]],16000,8000),residual)
            error=abs(measured-r["actual_reference_snr_post_db"])
            assert error<1e-3,"Actual saved noise residual has wrong SNR"
            snr_errors.append(error)
    assert all(len(set(v))==1 for v in noise_groups.values()),"Noise changed across methods or strengths"
    lpc_frames,low_energy=0,0
    for row in sources:
        uid=row["utterance_id"]
        assert sha256_file(root/row["waveform_path"])==row["waveform_sha256"]
        x,sr=read_audio(root/row["waveform_path"]);fresh=analyze_lpc(x,sr,cfg,lp)
        with np.load(out/"lpc"/f"{uid}.npz",allow_pickle=False) as p:
            for field in ["weights","envelope_db","centered_band_db","coefficients","times_s","frame_power"]:
                np.testing.assert_array_equal(p[field],getattr(fresh,field))
            np.testing.assert_array_equal(p["status"],fresh.status)
            valid=p["status"]=="ok";weights=p["weights"]
            assert np.isfinite(weights).all() and np.all((weights>=0)&(weights<=1))
            np.testing.assert_allclose(p["centered_band_db"][:,valid].mean(axis=0),0,atol=1e-12)
            np.testing.assert_array_equal(weights[:,valid],gaussian_weights(p["centered_band_db"][:,valid],lp.alpha,lp.gaussian_sigma_db))
            assert not weights[:,~valid].any()
        assert not fresh.metadata["numerical_failure_frames"]
        lpc_frames+=fresh.metadata["frames"];low_energy+=len(fresh.metadata["low_energy_frames"])
    # Recompute EER from the persisted aggregation CSV, independently of runner memory.
    buckets=defaultdict(list)
    with (out/"aggregate_scores.csv").open() as stream:
        for r in csv.DictReader(stream):
            buckets[(r["method"],float(r["delta_db"]),r["channel"],int(r["frames"]))].append((int(r["label"]),float(r["score"])))
    for r in summary["metrics"]:
        pairs=buckets[(r["method"],r["delta_db"],r["channel"],r["frames"])]
        recomputed=equal_error_rate(np.array([v[0] for v in pairs]),np.array([v[1] for v in pairs]))
        assert recomputed["eer"]==r["eer"] and recomputed["samples"]==r["samples"]
    null_metrics=[r for r in summary["metrics"] if r["delta_db"]==0]
    assert all(r["eer"]==.5 for r in null_metrics)
    q=list(csv.DictReader((out/"quality_scores.csv").open()))
    q=[{**r,"delta_db":float(r["delta_db"]),"pesq_nb":float(r["pesq_nb"]) if r["pesq_nb"] else None} for r in q]
    selected=json.loads((out/"quality_selection.json").read_text())
    fresh_selection=choose_quality_pair(q,selected["target_nb_pesq"],selected["tolerance"])
    assert {k:selected[k] for k in fresh_selection}==fresh_selection
    assert selected["quality_csv_sha256"]==sha256_file(out/"quality_scores.csv")
    assert selected==summary["quality_selection"]
    assert not read_jsonl(out/"failures.jsonl")
    coverage=list(csv.DictReader((out/"coverage.csv").open()))
    assert len(coverage)==expected and min(int(r["valid_frames"]) for r in coverage)>=max(config["evaluation"]["aggregate_frames"])
    frozen_verified={}
    for stage in ["M1","M2"]:
        folder=root/"milestones"/stage
        m=json.loads((folder/"snapshot_manifest.json").read_text())
        for r in m["files"]:assert sha256_file(folder/r["path"])==r["sha256"]
        frozen_verified[stage]=len(m["files"])
    old=json.loads((root/"milestones/M2/evidence/metrics.json").read_text())["metrics"]
    index={(r["delta_db"],r["channel"],r["frames"]):r for r in summary["metrics"] if r["method"]=="SS"}
    regression=[]
    for r in old:
        setting=(r["delta_db"],r["channel"],r["frames"])
        if r["detector"]=="correlation" and setting in index:
            regression.append(abs(r["eer"]-index[setting]["eer"]))
    return {"passed":True,"received_wav_hashes_and_redetections_verified":len(rows),
            "frame_csv_rows_verified_against_wavs":frame_rows_verified,"aggregate_csv_rows_verified_against_wavs":group_rows_verified,
            "embedded_wav_hashes_verified":len(embedded),"lpc_frames_recomputed":lpc_frames,
            "lpc_low_energy_frames":low_energy,"lpc_numerical_failures":0,
            "shared_noise_groups":len(noise_groups),"negative_pcm_shared_across_methods_and_strengths":len(negative_groups),
            "noise_residual_measurements":len(snr_errors),"max_saved_noise_snr_error_db":max(snr_errors),
            "delta_zero_identical_pcm_pairs":len(null_pairs),"delta_zero_eer_settings":len(null_metrics),
            "eer_settings_recomputed_from_csv":len(summary["metrics"]),"quality_match_recomputed":True,
            "minimum_valid_frames":min(int(r["valid_frames"]) for r in coverage),"maximum_received_peak":max_peak,
            "frozen_snapshot_files_verified":frozen_verified,"failures":0,
            "historical_ss_comparison":{"overlapping_m2_settings":len(regression),
                "max_absolute_eer_difference":max(regression),"changed_settings":sum(v>0 for v in regression),
                "note":"M3 joint gain includes 8 dB variants; VAD can change vs M2. Fair method comparisons share gain within M3."},
            "scope":"public smoke only; same-environment numerical replay, no formal benchmark"}
