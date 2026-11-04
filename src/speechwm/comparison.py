"""M3: paired SS/LPC quality scan followed by frozen-quality robustness evaluation."""
from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from .audio import read_audio, write_audio
from .channels import apply_channel
from .embedding import embed, modify_spectrum
from .environment import probe_environment
from .evaluation import smoke_checks, write_csv
from .features import aggregate, extract_features, score_frames
from .keys import key_hash, load_key
from .lpc import LPCConfig, analyze_lpc
from .metrics import detection_curve, equal_error_rate, quality_scores
from .spectral import SpectralConfig, analyze, log_magnitude, synthesize
from .utils import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl


class CSVStream:
    def __init__(self,path):
        self.file=path.open("w",newline="",encoding="utf-8")
        self.writer=None
        self.count=0

    def add(self,row):
        if self.writer is None:
            self.writer=csv.DictWriter(self.file,fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.count+=1

    def close(self):
        self.file.close()


def pcm_hash(audio):
    return hashlib.sha256(np.asarray(audio,dtype="<f8").tobytes()).hexdigest()


def choose_quality_pair(rows: list[dict], target: float, tolerance: float) -> dict:
    """No EER, label scores or channels are accepted as selection inputs."""
    buckets=defaultdict(list)
    for row in rows:
        if row["delta_db"]>0 and row["pesq_nb"] is not None:
            buckets[(row["method"],row["delta_db"])].append(float(row["pesq_nb"]))
    means={k:float(np.mean(v)) for k,v in buckets.items()}
    candidates=[]
    for (method,ss),ss_quality in means.items():
        if method!="SS":continue
        for (other,lpc),lpc_quality in means.items():
            if other!="SS_LPC":continue
            difference=abs(ss_quality-lpc_quality)
            if difference<=tolerance:
                candidates.append((abs(ss_quality-target)+abs(lpc_quality-target),difference,ss,lpc))
    if not candidates:
        raise ValueError("No measured quality pair within tolerance; extend the quality grid in a new run")
    _,difference,ss,lpc=min(candidates)
    return {"scope":"smoke quality matching only; no held-out calibration or test",
            "target_nb_pesq":target,"tolerance":tolerance,"ss_delta_db":ss,"lpc_delta_db":lpc,
            "ss_mean_nb_pesq":means[("SS",ss)],"lpc_mean_nb_pesq":means[("SS_LPC",lpc)],
            "absolute_mean_difference":difference,"status":"matched_measured_points",
            "selection":"minimize sum of distances to target among measured pairs within tolerance; then mean difference and strengths",
            "selected_before_channel_evaluation":True,"interpolated":False,
            "per_utterance_strength_selection":False}


def run_comparison(root: Path, config_path: Path, run_id: str | None = None) -> Path:
    started=time.perf_counter()
    config=yaml.safe_load(config_path.read_text())
    cfg=SpectralConfig(**config["spectral"]);lp=LPCConfig(**config["lpc"])
    if config.get("experiment_kind")!="lpc_comparison_v1" or config["key"]["bit_generator"]!="PCG64":
        raise ValueError("Expected explicit LPC comparison protocol and PCG64 key")
    manifest=read_jsonl(root/config["data"]["manifest"])
    if not manifest or any(r["split"]!="smoke" for r in manifest):
        raise ValueError("This runner supports development smoke data only")
    methods=config["evaluation"]["methods"];deltas=config["evaluation"]["deltas_db"]
    if methods!=["SS","SS_LPC"] or 0 not in deltas or len(set(deltas))!=len(deltas):
        raise ValueError("Require paired SS/SS_LPC and unique strengths including zero")
    key=load_key(root/config["key"]["path"])
    run_id=run_id or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    if not run_id or Path(run_id).name!=run_id or run_id in {".",".."}:raise ValueError("Invalid run ID")
    out=root/config["output"]["root"]/config["experiment_id"]/run_id
    out.mkdir(parents=True,exist_ok=False)
    for folder in ["audio","lpc"]:(out/folder).mkdir()
    (out/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
    write_jsonl(out/"input_manifest.jsonl",manifest)
    write_json(out/"key.json",json.loads((root/config["key"]["path"]).read_text()))
    write_json(out/"environment.json",probe_environment())
    write_json(out/"e00_checks.json",smoke_checks(cfg,key))
    write_json(out/"run_status.json",{"status":"running_quality_scan"})
    try:
        return _run(root,out,config,cfg,lp,key,manifest,started)
    except Exception as exc:
        write_json(out/"run_status.json",{"status":"failed","error":repr(exc)})
        with (out/"failures.jsonl").open("a") as f:
            f.write(json.dumps({"stage":"fatal","error":repr(exc)},ensure_ascii=False)+"\n")
        raise


def _run(root,out,config,cfg,lp,key,manifest,started):
    methods=config["evaluation"]["methods"];deltas=config["evaluation"]["deltas_db"]
    channels=config["evaluation"]["channels"];aggregations=config["evaluation"]["aggregate_frames"]
    config_hash=canonical_hash(config)
    quality,embeddings,failures,gains,lpc_info=[],[],[],[],[]
    frame_log=CSVStream(out/"lpc_frames.csv")
    raw_distribution,centered_distribution,voiced_centered=[],[],[]
    # Phase 1: scan both methods' measured pre-channel quality with one joint gain.
    for source_i,row in enumerate(manifest):
        uid=row["utterance_id"];path=root/row["waveform_path"]
        if sha256_file(path)!=row["waveform_sha256"]:raise ValueError(f"Source hash mismatch: {uid}")
        reference,sr=read_audio(path)
        if sr!=16000:raise ValueError("M3 embeds only 16 kHz prepared audio")
        profile=analyze_lpc(reference,sr,cfg,lp)
        if profile.metadata["numerical_failure_frames"]:
            raise ValueError(f"LPC numerical failures require investigation: {uid}: {profile.metadata}")
        spectrum,times=analyze(reference,sr,cfg)
        np.testing.assert_array_equal(np.rint(times*sr),np.rint(profile.times_s*sr))
        indices,frequencies=cfg.indices(sr)
        reference_feature=extract_features(reference,sr,key,cfg)
        valid=profile.status=="ok"
        voiced=valid & reference_feature.valid_mask
        raw_distribution.append(profile.envelope_db[indices][:,valid].ravel())
        centered_distribution.append(profile.centered_band_db[:,valid].ravel())
        voiced_centered.append(profile.centered_band_db[:,voiced].ravel())
        np.savez_compressed(out/"lpc"/f"{uid}.npz",weights=profile.weights,envelope_db=profile.envelope_db,
            centered_band_db=profile.centered_band_db,coefficients=profile.coefficients,times_s=profile.times_s,
            frame_power=profile.frame_power,status=profile.status,reference_voiced_mask=reference_feature.valid_mask)
        write_json(out/"lpc"/f"{uid}.json",profile.metadata)
        lpc_info.append({"utterance_id":uid,**profile.metadata})
        for i,stamp in enumerate(profile.times_s):
            frame_log.add({"utterance_id":uid,"frame_index":i,"time_s":stamp,
                "status":profile.status[i],"frame_power":profile.frame_power[i],
                "reference_voiced":int(reference_feature.valid_mask[i]),
                "raw_band_mean_db":float(profile.envelope_db[indices,i].mean()) if valid[i] else None,
                "centered_band_std_db":float(profile.centered_band_db[:,i].std()) if valid[i] else None,
                "weight_min":float(profile.weights[:,i].min()),"weight_max":float(profile.weights[:,i].max()),
                "weight_mean":float(profile.weights[:,i].mean())})
        variants={}
        negative=embed(reference,sr,key,0,cfg)[0]
        for method in methods:
            for delta in deltas:
                variants[(method,delta)]=(embed(reference,sr,key,delta,cfg)[0] if method=="SS" else
                    synthesize(modify_spectrum(spectrum,sr,key,delta,cfg,profile.weights),sr,len(reference),cfg))
        peak=max(float(abs(x).max()) for x in [negative,*variants.values()])
        gain=min(1.,config["audio"]["joint_peak_limit"]/peak)
        gains.append({"utterance_id":uid,"common_gain":gain,"maximum_unscaled_variant_peak":peak,
                      "joint_peak_limit":config["audio"]["joint_peak_limit"]})
        for (method,delta),positive in variants.items():
            for label,audio in [(0,negative),(1,positive)]:
                destination=out/"audio"/f"{uid}_{method}_d{delta:g}_label{label}_embed.wav"
                write_audio(destination,audio*gain,sr)
                saved,_=read_audio(destination)
                embeddings.append({"utterance_id":uid,"method":method,"delta_db":float(delta),"label":label,
                    "common_gain":gain,"waveform_path":str(destination.relative_to(root)),
                    "waveform_sha256":sha256_file(destination),"pcm_sha256":pcm_hash(saved)})
                if label==1:
                    q=quality_scores(reference*gain,saved,sr,config["evaluation"]["compute_pesq"])
                    quality.append({"utterance_id":uid,"parent_utterance_id":row["parent_utterance_id"],
                        "method":method,"delta_db":float(delta),"common_gain":gain,
                        "pesq_nb":q["pesq_nb"],"pesq_wb":q["pesq_wb"],"embedding_snr_db":q["embedding_snr_db"],
                        "scope":"pre_channel_embedding_quality"})
                    failures.extend({"utterance_id":uid,"method":method,"delta_db":delta,"stage":"quality",**f}
                                    for f in q["quality_failures"])
        if source_i==0:
            actual={}
            for method in methods:
                p=out/"audio"/f"{uid}_{method}_d3_label1_embed.wav"
                audio,_=read_audio(p);actual[method]=log_magnitude(analyze(audio,sr,cfg)[0])-log_magnitude(analyze(reference*gain,sr,cfg)[0])
            np.savez_compressed(out/"diagnostic.npz",reference_spectrum_db=log_magnitude(spectrum),
                envelope_db=profile.envelope_db,centered_band_db=profile.centered_band_db,weights=profile.weights,
                times_s=times,band_indices=indices,frequency_hz=frequencies,frame_power=profile.frame_power,
                target_ss_db=np.broadcast_to(3*key[:,None],profile.weights.shape),
                target_lpc_db=3*key[:,None]*profile.weights,actual_ss_db=actual["SS"][indices],actual_lpc_db=actual["SS_LPC"][indices])
        print(f"Quality [{source_i+1}/{len(manifest)}] {uid}",flush=True)
    frame_log.close()
    write_csv(out/"quality_scores.csv",quality);write_csv(out/"common_gains.csv",gains)
    write_jsonl(out/"embedding_manifest.jsonl",embeddings)
    write_jsonl(out/"lpc_manifest.jsonl",lpc_info);write_jsonl(out/"failures.jsonl",failures)
    raw=np.concatenate(raw_distribution);centered=np.concatenate(centered_distribution);voiced=np.concatenate(voiced_centered)
    np.savez_compressed(out/"lpc_distributions.npz",raw_db=raw,centered_db=centered,voiced_centered_db=voiced)
    write_json(out/"lpc_summary.json",{"config":asdict(lp),"segments":len(manifest),
        "frames":sum(r["frames"] for r in lpc_info),"valid_lpc_frames":sum(r["valid_lpc_frames"] for r in lpc_info),
        "low_energy_frames":sum(len(r["low_energy_frames"]) for r in lpc_info),"numerical_failure_frames":0,
        "raw_mean_db":float(raw.mean()),"raw_std_db":float(raw.std()),
        "centered_mean_db":float(centered.mean()),"centered_std_db":float(centered.std()),
        "voiced_centered_std_db":float(voiced.std()),"fixed_gaussian_sigma_db":lp.gaussian_sigma_db,
        "empirical_sigma_was_used_for_weights":False})
    match=config["evaluation"]["quality_match"]
    selection=choose_quality_pair(quality,match["target_nb_pesq"],match["tolerance"])
    selection["quality_csv_sha256"]=sha256_file(out/"quality_scores.csv")
    write_json(out/"quality_selection.json",selection)
    write_json(out/"run_status.json",{"status":"running_channels","quality_pair_frozen":selection})
    print("Quality pair frozen before channel scores: "+json.dumps(selection),flush=True)
    # Phase 2: fixed channels and reference-free detector for every method/strength/label.
    source_index={r["utterance_id"]:r for r in manifest};gain_index={r["utterance_id"]:r["common_gain"] for r in gains}
    buckets=defaultdict(list);artifacts=[];channel_metadata=[];coverage=[]
    frames_stream=CSVStream(out/"frame_scores.csv");groups_stream=CSVStream(out/"aggregate_scores.csv")
    for e_i,item in enumerate(embeddings):
        uid=item["utterance_id"];row=source_index[uid]
        saved,sr=read_audio(root/item["waveform_path"])
        reference,_=read_audio(root/row["waveform_path"]);reference*=gain_index[uid]
        for channel in channels:
            base={"utterance_id":uid,"parent_utterance_id":row["parent_utterance_id"],"speaker_id":row["speaker_id"],
                  "method":item["method"],"delta_db":item["delta_db"],"channel":channel,"label":item["label"]}
            destination=out/"audio"/f"{uid}_{item['method']}_d{item['delta_db']:g}_label{item['label']}_{channel}.wav"
            received,rate,meta=apply_channel(saved,sr,channel,destination,reference,uid,config["data"]["seed"])
            feature=extract_features(received,rate,key,cfg,vad=config["evaluation"]["vad"]["mode"],
                                      vad_aggressiveness=config["evaluation"]["vad"]["aggressiveness"])
            scores=score_frames(feature.features);valid=feature.valid_mask
            channel_metadata.append({**base,**meta})
            artifacts.append({**base,"sample_rate":rate,"samples":len(received),"waveform_path":str(destination.relative_to(root)),
                "waveform_sha256":sha256_file(destination),"pcm_sha256":pcm_hash(received),"peak":float(abs(received).max()),
                "frame_scores_sha256":pcm_hash(scores),"valid_mask_sha256":hashlib.sha256(valid.tobytes()).hexdigest(),
                "config_hash":config_hash})
            coverage.append({**base,**feature.metadata,"voiced_fraction":float(valid.mean())})
            for i,(score,stamp,is_valid) in enumerate(zip(scores,feature.times_s,valid)):
                frames_stream.add({**base,"frame_index":i,"time_s":float(stamp),"valid":int(is_valid),"score":float(score)})
            if not valid.any():failures.append({**base,"stage":"detection","error":"No voiced frames"})
            for count in aggregations:
                groups=aggregate(scores[valid],feature.times_s[valid],count)
                for group in groups:
                    groups_stream.add({**base,**group})
                    buckets[(item["method"],item["delta_db"],channel,count)].append((uid,item["label"],group["score"],group["group_index"]))
                if not groups:failures.append({**base,"stage":"aggregation","frames":count,"error":"Insufficient voiced frames"})
        if (e_i+1)%32==0:print(f"Channels [{e_i+1}/{len(embeddings)}]",flush=True)
    frames_stream.close();groups_stream.close()
    write_jsonl(out/"audio_manifest.jsonl",artifacts);write_jsonl(out/"channel_manifest.jsonl",channel_metadata)
    write_csv(out/"coverage.csv",coverage)
    metric_rows=[];curves=CSVStream(out/"threshold_curves.csv")
    for (method,delta,channel,count),rows in sorted(buckets.items()):
        labels=np.array([r[1] for r in rows]);scores=np.array([r[2] for r in rows])
        stats=equal_error_rate(labels,scores);stats.pop("threshold_role")
        first=[r for r in rows if r[3]==0]
        first_eer=equal_error_rate(np.array([r[1] for r in first]),np.array([r[2] for r in first]))["eer"]
        paired={r[0] for r in rows if r[1]==0}&{r[0] for r in rows if r[1]==1}
        base={"method":method,"delta_db":delta,"channel":channel,"frames":count}
        metric_rows.append({**base,**stats,"eer_first_group_per_segment":first_eer,"paired_eligible_segments":len(paired),
            "negative_groups":int((labels==0).sum()),"positive_groups":int((labels==1).sum())})
        curve=detection_curve(labels,scores)
        for t,far,frr in zip(curve["threshold"],curve["far"],curve["frr"]):curves.add({**base,"threshold":t,"far":far,"frr":frr})
    curves.close();write_csv(out/"metrics.csv",metric_rows);write_jsonl(out/"failures.jsonl",failures)
    write_json(out/"metrics.json",{"experiment_kind":"lpc_comparison_v1","experiment_id":config["experiment_id"],
        "run_id":out.name,"scope":"development_smoke_only; quality matched on same smoke; no held-out test",
        "segments":len(manifest),"parents":len({r["parent_utterance_id"] for r in manifest}),
        "speakers":len({r["speaker_id"] for r in manifest}),"config_hash":config_hash,
        "input_manifest_sha256":sha256_file(out/"input_manifest.jsonl"),"key_sha256":key_hash(key),
        "audio_artifacts":len(artifacts),"embedding_artifacts":len(embeddings),"frame_score_rows":frames_stream.count,
        "aggregate_score_rows":groups_stream.count,"metric_rows":len(metric_rows),"failures":len(failures),
        "joint_gain_attenuated_segments":sum(r["common_gain"]<1 for r in gains),"elapsed_s":time.perf_counter()-started,
        "quality_selection":selection,"threshold_policy":"descriptive EER only; no deployment threshold","metrics":metric_rows})
    from .lpc_validation import validate_comparison
    from .lpc_reporting import make_lpc_report
    write_json(out/"validation.json",validate_comparison(out))
    make_lpc_report(out)
    write_json(out/"run_status.json",{"status":"complete","validation_passed":True})
    return out
