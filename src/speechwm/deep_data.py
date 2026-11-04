from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .audio import read_audio, write_audio
from .channels import apply_channel
from .comparison import pcm_hash
from .embedding import embed
from .evaluation import write_csv
from .features import extract_features
from .spectral import SpectralConfig
from .utils import read_jsonl, sha256_file, write_json, write_jsonl

SPLITS=("dev_train","dev_validation","dev_evaluation")


def audit_splits(rows):
    if not rows or {r["split"] for r in rows}!=set(SPLITS):raise ValueError("Require all three development partitions")
    for field in ["parent_utterance_id","speaker_id","source_sha256","waveform_sha256"]:
        assignments=defaultdict(set)
        for row in rows:assignments[row[field]].add(row["split"])
        if any(len(v)!=1 for v in assignments.values()):raise ValueError(f"Cross-split leakage: {field}")
    if len({r["utterance_id"] for r in rows})!=len(rows):raise ValueError("Duplicate segment ID")
    return {split:{"segments":sum(r["split"]==split for r in rows),
            "parents":sorted({r["parent_utterance_id"] for r in rows if r["split"]==split}),
            "speakers":sorted({r["speaker_id"] for r in rows if r["split"]==split}),
            "duration_s":sum(r["duration_s"] for r in rows if r["split"]==split)} for split in SPLITS}


def prepare_deep_data(root,out,config,key):
    original=read_jsonl(root/config["data"]["manifest"])
    if any(r["split"]!="smoke" for r in original):raise ValueError("M4 development runner accepts smoke sources only")
    mapping=config["data"]["parent_splits"]
    if set(mapping)!={r["parent_utterance_id"] for r in original}:raise ValueError("Explicit parent split map must cover every source")
    rows=[{**r,"original_split":r["split"],"split":mapping[r["parent_utterance_id"]]} for r in original]
    audit=audit_splits(rows);write_json(out/"split_audit.json",audit);write_jsonl(out/"split_manifest.jsonl",rows)
    cfg=SpectralConfig(**config["spectral"])
    for name in ["audio","features"]:(out/name).mkdir()
    features=[];artifacts=[];metadata=[];coverage=[];gains=[]
    for index,row in enumerate(rows):
        uid=row["utterance_id"]
        if sha256_file(root/row["waveform_path"])!=row["waveform_sha256"]:raise ValueError("Prepared source hash mismatch")
        reference,sr=read_audio(root/row["waveform_path"])
        if sr!=16000:raise ValueError("Expected 16 kHz embedding source")
        variants={d:embed(reference,sr,key,d,cfg)[0] for d in [0,*[c["delta_db"] for c in config["conditions"]]]}
        peak=max(float(abs(v).max()) for v in variants.values())
        gain=min(1.,config["audio"]["joint_peak_limit"]/peak);reference*=gain
        gains.append({"utterance_id":uid,"common_gain":gain,"maximum_unscaled_peak":peak})
        for condition in config["conditions"]:
            for kind in ["main","zero_control"]:
                for label in [0,1]:
                    delta=condition["delta_db"] if kind=="main" and label else 0
                    stem=f"{uid}_{condition['id']}_{kind}_label{label}"
                    embedded=out/"audio"/(stem+"_embed.wav")
                    write_audio(embedded,variants[delta]*gain,sr);saved,_=read_audio(embedded)
                    received_path=out/"audio"/(stem+".wav")
                    received,rate,channel_meta=apply_channel(saved,sr,condition["channel"],received_path,reference,uid,20261017)
                    f=extract_features(received,rate,key,cfg,vad=config["evaluation"]["vad"]["mode"],
                        vad_aggressiveness=config["evaluation"]["vad"]["aggressiveness"])
                    if f.valid_mask.sum()<max(config["evaluation"]["aggregate_frames"]):raise ValueError("Insufficient speech for M4 aggregation")
                    cache=out/"features"/(stem+".npz")
                    np.savez_compressed(cache,features=f.features[f.valid_mask].astype(np.float32),
                        features_f64=f.features,valid_mask=f.valid_mask,times_s=f.times_s)
                    base={"id":stem,"utterance_id":uid,"parent_utterance_id":row["parent_utterance_id"],
                        "speaker_id":row["speaker_id"],"split":row["split"],"condition":condition["id"],
                        "channel":condition["channel"],"kind":kind,"label":label,"actual_embed_delta_db":delta,
                        "nominal_delta_db":condition["delta_db"],"common_gain":gain}
                    artifacts.append({**base,"waveform_path":str(received_path.relative_to(root)),
                        "waveform_sha256":sha256_file(received_path),"pcm_sha256":pcm_hash(received),
                        "sample_rate":rate,"samples":len(received),"embedded_path":str(embedded.relative_to(root)),
                        "embedded_sha256":sha256_file(embedded),"peak":float(abs(received).max())})
                    features.append({**base,"features_path":str(cache.relative_to(out)),"features_sha256":sha256_file(cache),
                        "valid_frames":int(f.valid_mask.sum()),"waveform_sha256":sha256_file(received_path)})
                    metadata.append({**base,**channel_meta})
                    coverage.append({**base,**f.metadata,"voiced_fraction":float(f.valid_mask.mean())})
        print(f"Features [{index+1}/{len(rows)}] {uid} -> {row['split']}",flush=True)
    write_jsonl(out/"feature_manifest.jsonl",features);write_jsonl(out/"audio_manifest.jsonl",artifacts)
    write_jsonl(out/"channel_manifest.jsonl",metadata);write_csv(out/"coverage.csv",coverage)
    write_csv(out/"common_gains.csv",gains)
    return features


def load_entries(out,split,condition,kind="main"):
    entries=[]
    for row in read_jsonl(out/"feature_manifest.jsonl"):
        if row["split"]!=split or row["condition"]!=condition or row["kind"]!=kind:continue
        if sha256_file(out/row["features_path"])!=row["features_sha256"]:raise ValueError("Feature cache hash mismatch")
        with np.load(out/row["features_path"],allow_pickle=False) as f:
            entries.append({**row,"x":f["features"],"times":f["times_s"][f["valid_mask"]],
                            "correlation":f["features_f64"][f["valid_mask"]].mean(axis=1),
                            "frame_indices":np.flatnonzero(f["valid_mask"])})
    return entries
