from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from .audio import read_audio, resample_audio, write_audio
from .embedding import embed
from .environment import probe_environment
from .features import aggregate, extract_features, score_frames
from .keys import key_hash, load_key, make_key, save_key
from .metrics import detection_curve, equal_error_rate, quality_scores
from .spectral import SpectralConfig, analyze, log_magnitude, synthesize
from .utils import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def smoke_checks(cfg: SpectralConfig, key: np.ndarray) -> dict:
    checks={}
    for sr in [8000,16000]:
        signal=np.random.default_rng(11).normal(0,.04,sr+13)
        signal[0],signal[-1]=.25,-.25
        spec,_=analyze(signal,sr,cfg)
        reconstructed=synthesize(spec,sr,len(signal),cfg)
        error=float(np.max(np.abs(signal-reconstructed)))
        checks[f"roundtrip_{sr}_max_error"]=error
        if error>1e-6:
            raise AssertionError(f"STFT roundtrip failed at {sr} Hz")
    _,f16=cfg.indices(16000)
    _,f8=cfg.indices(8000)
    np.testing.assert_array_equal(f16,f8)
    np.testing.assert_array_equal(f16,np.arange(500,3650,50))
    checks["frequency_hz"]=f16.tolist()
    checks["key_sha256"]=key_hash(key)
    checks["key_positive_count"]=int(np.sum(key==1))
    checks["key_negative_count"]=int(np.sum(key==-1))
    checks["eer_perfect"]=equal_error_rate(np.array([0,0,1,1]),np.arange(4))["eer"]
    checks["eer_tied"]=equal_error_rate(np.array([0,0,1,1]),np.zeros(4))["eer"]
    assert checks["eer_perfect"]==0 and checks["eer_tied"]==.5
    checks["passed"]=True
    return checks


def _channel(audio: np.ndarray, sr: int, name: str) -> tuple[np.ndarray,int]:
    if name=="clean":
        return audio.copy(),sr
    if name=="downsample_8k":
        return resample_audio(audio,sr,8000),8000
    raise ValueError(f"Unsupported baseline channel: {name}")


def run_baseline(root: Path, config_path: Path, run_id: str | None = None) -> Path:
    started=time.perf_counter()
    config=yaml.safe_load(config_path.read_text())
    controlled = config.get("channel_protocol") == "controlled_v1"
    if controlled:
        from .channels import CHANNELS, codec_runtime
        if not set(config["evaluation"]["channels"]) <= CHANNELS:
            raise ValueError("Unknown controlled channel")
        runtime = codec_runtime()  # Fail before creating a partial experiment if codec is unavailable.
    if config["key"]["bit_generator"]!="PCG64":
        raise ValueError("Only explicitly recorded PCG64 keys are supported")
    cfg=SpectralConfig(**config["spectral"])
    manifest_path=root/config["data"]["manifest"]
    manifest=read_jsonl(manifest_path)
    if not manifest or any(row["split"]!="smoke" for row in manifest):
        raise ValueError("This command is for development smoke data only")
    key_path=root/config["key"]["path"]
    if not key_path.exists():
        save_key(key_path,config["key"]["seed"])
    key=load_key(key_path)
    wrong_key=make_key(config["key"]["seed"]+1)
    run_id=run_id or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    if not run_id or Path(run_id).name!=run_id or run_id in {".",".."}:
        raise ValueError("run_id must be one directory component")
    out=root/config["output"]["root"]/config["experiment_id"]/run_id
    out.mkdir(parents=True,exist_ok=False)
    (out/"audio").mkdir()
    (out/"config.yaml").write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False))
    write_jsonl(out/"input_manifest.jsonl",manifest)
    write_json(out/"environment.json",probe_environment())
    if controlled:
        write_json(out/"codec_runtime.json",runtime)
    write_json(out/"e00_checks.json",smoke_checks(cfg,key))
    write_json(out/"key.json",json.loads(key_path.read_text()))
    frame_rows,group_rows,quality_rows,coverage_rows,failures,artifacts=[],[],[],[],[],[]
    channel_rows,post_quality_rows=[],[]
    config_hash=canonical_hash(config)
    diagnostic_written=False
    deltas=config["evaluation"]["deltas_db"]
    for source_i,row in enumerate(manifest):
        uid=row["utterance_id"]
        path=root/row["waveform_path"]
        if sha256_file(path)!=row["waveform_sha256"]:
            raise ValueError(f"Prepared audio hash mismatch: {path}")
        reference,sr=read_audio(path)
        if sr!=config["audio"]["embed_sample_rate"]:
            raise ValueError(f"Unexpected sample rate in prepared manifest: {sr}")
        negative,_=embed(reference,sr,key,0,cfg)
        embedded={float(d):embed(reference,sr,key,float(d),cfg)[0] for d in deltas}
        peak=max(float(np.max(np.abs(x))) for x in [negative,*embedded.values()])
        common_gain=min(1.0,.99/max(peak,1e-12))
        reference=reference*common_gain
        negative=negative*common_gain
        for delta in deltas:
            delta=float(delta)
            positive=embedded[delta]*common_gain
            tag=f"{delta:g}"
            # Both labels pass through the same reconstruction, encoding, storage and feature path.
            embed_paths={}
            for label,audio in [(0,negative),(1,positive)]:
                embed_path=out/"audio"/f"{uid}_d{tag}_label{label}_embed.wav"
                write_audio(embed_path,audio,sr)
                embed_paths[label]=embed_path
            saved_positive,_=read_audio(embed_paths[1])
            quality=quality_scores(reference,saved_positive,sr,config["evaluation"]["compute_pesq"])
            quality_rows.append({"utterance_id":uid,"parent_utterance_id":row["parent_utterance_id"],
                                 "delta_db":delta,"common_gain":common_gain,
                                 "embedding_snr_db":quality["embedding_snr_db"],
                                 "pesq_nb":quality["pesq_nb"],"pesq_wb":quality["pesq_wb"],
                                 "scope":"pre_channel_embedding_quality"})
            failures.extend({"utterance_id":uid,"delta_db":delta,"stage":"quality",**f}
                            for f in quality["quality_failures"])
            if not diagnostic_written and delta==1:
                ref_spec,times=analyze(reference,sr,cfg)
                actual_spec,_=analyze(saved_positive,sr,cfg)
                indices,frequencies=cfg.indices(sr)
                np.savez_compressed(out/"diagnostic.npz",reference=reference,positive=saved_positive,
                                    sr=sr,times_s=times,ref_db=log_magnitude(ref_spec),
                                    actual_delta_db=log_magnitude(actual_spec)-log_magnitude(ref_spec),
                                    target_band_delta_db=np.broadcast_to(key[:,None],(63,len(times))),
                                    band_indices=indices,band_frequency_hz=frequencies)
                diagnostic_written=True
            for channel in config["evaluation"]["channels"]:
                for label in [0,1]:
                    saved,saved_sr=read_audio(embed_paths[label])
                    received_path=out/"audio"/f"{uid}_d{tag}_label{label}_{channel}.wav"
                    if controlled:
                        from .channels import apply_channel
                        try:
                            received,received_sr,channel_metadata=apply_channel(saved,saved_sr,channel,
                                received_path,reference,uid,config["data"]["seed"])
                        except Exception as exc:
                            failure={"utterance_id":uid,"delta_db":delta,"channel":channel,"label":label,
                                     "stage":"channel","error":repr(exc)}
                            write_jsonl(out/"failures.jsonl",[*failures,failure])
                            write_json(out/"run_status.json",{"status":"failed","fatal":failure})
                            raise
                        channel_rows.append({"utterance_id":uid,"delta_db":delta,"label":label,
                                             **channel_metadata})
                    else:
                        received,received_sr=_channel(saved,saved_sr,channel)
                        write_audio(received_path,received,received_sr)
                    received,received_sr=read_audio(received_path)
                    if controlled and delta in config["evaluation"]["post_quality_deltas"]:
                        post_quality=quality_scores(resample_audio(reference,sr,received_sr),received,
                                                    received_sr,config["evaluation"]["compute_pesq"])
                        post_quality_rows.append({"utterance_id":uid,"delta_db":delta,"channel":channel,
                            "label":label,"pesq_nb":post_quality["pesq_nb"],"pesq_wb":post_quality["pesq_wb"],
                            "total_distortion_snr_db":post_quality["embedding_snr_db"],
                            "scope":"received vs unmarked reference at matched rate; total distortion"})
                        failures.extend({"utterance_id":uid,"delta_db":delta,"channel":channel,
                            "label":label,"stage":"post_channel_quality",**f} for f in post_quality["quality_failures"])
                    feature=extract_features(received,received_sr,key,cfg,
                        vad=config["evaluation"]["vad"]["mode"],
                        vad_aggressiveness=config["evaluation"]["vad"]["aggressiveness"])
                    scores=score_frames(feature.features)
                    valid=feature.valid_mask
                    base={"utterance_id":uid,"parent_utterance_id":row["parent_utterance_id"],
                          "speaker_id":row["speaker_id"],"delta_db":delta,"channel":channel,"label":label}
                    coverage_rows.append({**base,**feature.metadata,
                                          "voiced_fraction":float(valid.mean())})
                    artifacts.append({**base,"sample_rate":received_sr,
                                      "waveform_path":str(received_path.relative_to(root)),
                                      "waveform_sha256":sha256_file(received_path),
                                      "duration_s":len(received)/received_sr,
                                      "max_abs":float(np.max(np.abs(received))),
                                      "config_hash":config_hash})
                    for i,(score,stamp,is_valid) in enumerate(zip(scores,feature.times_s,valid)):
                        frame_rows.append({**base,"detector":"correlation", "frame_index":i,
                                           "time_s":float(stamp),"valid":int(is_valid),"score":float(score)})
                    detectors={"correlation":scores}
                    # A diagnostic, not an assertion that every unrelated sequence yields chance-level scores.
                    if delta==1:
                        wrong_features=feature.features*(wrong_key/key)[None,:]
                        detectors["correlation_wrong_key"]=score_frames(wrong_features)
                    if not valid.any():
                        failures.append({**base,"stage":"detection","error":"No valid voiced frames"})
                    for detector,detector_scores in detectors.items():
                        for frames in config["evaluation"]["aggregate_frames"]:
                            groups=aggregate(detector_scores[valid],feature.times_s[valid],frames)
                            for group in groups:
                                group_rows.append({**base,"detector":detector,**group})
                            if not groups:
                                failures.append({**base,"stage":"aggregation","frames":frames,
                                                 "error":"Insufficient valid voiced frames"})
        print(f"[{source_i+1}/{len(manifest)}] {uid}: saved/reloaded and scored",flush=True)
    write_csv(out/"frame_scores.csv",frame_rows)
    write_csv(out/"aggregate_scores.csv",group_rows)
    write_csv(out/"quality_scores.csv",quality_rows)
    write_csv(out/"coverage.csv",coverage_rows)
    write_jsonl(out/"failures.jsonl",failures)
    write_jsonl(out/"audio_manifest.jsonl",artifacts)
    if controlled:
        write_jsonl(out/"channel_manifest.jsonl",channel_rows)
        write_csv(out/"post_channel_quality.csv",post_quality_rows)
    groups_by_setting=defaultdict(list)
    for row in group_rows:
        groups_by_setting[(row["delta_db"],row["channel"],row["detector"],row["frames"])].append(row)
    metric_rows,curve_rows=[],[]
    for (delta,channel,detector,frames),rows in sorted(groups_by_setting.items()):
        labels=np.array([r["label"] for r in rows])
        scores=np.array([r["score"] for r in rows])
        if np.unique(labels).size!=2:
            failures.append({"stage":"metrics","delta_db":delta,"channel":channel,
                             "detector":detector,"frames":frames,"error":"Missing one class"})
            continue
        stats=equal_error_rate(labels,scores)
        stats.pop("threshold_role")
        base={"delta_db":delta,"channel":channel,"detector":detector,"frames":frames}
        first_rows=[r for r in rows if r["group_index"]==0]
        first_labels=np.array([r["label"] for r in first_rows])
        first_scores=np.array([r["score"] for r in first_rows])
        first_eer=(equal_error_rate(first_labels,first_scores)["eer"]
                   if np.unique(first_labels).size==2 else None)
        paired={r["utterance_id"] for r in rows if r["label"]==0}&{
            r["utterance_id"] for r in rows if r["label"]==1}
        metric_rows.append({**base,**stats,"eer_first_group_per_segment":first_eer,
                            "eligible_segments":len({r["utterance_id"] for r in rows}),
                            "paired_eligible_segments":len(paired),
                            "negative_groups":int(np.sum(labels==0)),
                            "positive_groups":int(np.sum(labels==1)),
                            "negative_score_mean":float(scores[labels==0].mean()),
                            "positive_score_mean":float(scores[labels==1].mean())})
        curve=detection_curve(labels,scores)
        for threshold,far,frr in zip(curve["threshold"],curve["far"],curve["frr"]):
            curve_rows.append({**base,"threshold":float(threshold),"far":float(far),"frr":float(frr)})
    write_csv(out/"metrics.csv",metric_rows)
    write_csv(out/"threshold_curves.csv",curve_rows)
    write_jsonl(out/"failures.jsonl",failures)
    summary={"experiment_id":config["experiment_id"],"run_id":run_id,
             "scope":"development_smoke_only; no held-out performance claims",
             "config_hash":config_hash,"input_manifest_sha256":sha256_file(manifest_path),
             "segments":len(manifest),"parents":len({r["parent_utterance_id"] for r in manifest}),
             "speakers":len({r["speaker_id"] for r in manifest}),
             "audio_artifacts":len(artifacts),"frame_score_rows":len(frame_rows),
             "aggregate_score_rows":len(group_rows),"metric_rows":len(metric_rows),
             "failures":len(failures),"elapsed_s":time.perf_counter()-started,
             "threshold_policy":"EER thresholds are descriptive, no deployment threshold calibrated",
             "spectral":asdict(cfg),"key_sha256":key_hash(key),
             "source_compression":"HQ Ogg public examples; not lossless source corpus",
             "metrics":metric_rows}
    if controlled:
        summary["channel_protocol"]="controlled_v1"
    write_json(out/"metrics.json",summary)
    if controlled:
        from .channel_validation import validate_run
        try:
            write_json(out/"channel_validation.json",validate_run(out))
        except Exception as exc:
            write_json(out/"run_status.json",{"status":"failed_validation","error":repr(exc)})
            raise
        from .channel_reporting import make_channel_report
        make_channel_report(out)
        write_json(out/"run_status.json",{"status":"complete","channel_validation_passed":True})
    else:
        from .reporting import make_report
        make_report(out)
    return out

