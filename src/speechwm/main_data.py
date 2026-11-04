"""M5 paired waveforms and shared features. Test artifacts are prepared only after the gate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .audio import read_audio, write_audio
from .channels import apply_channel
from .comparison import pcm_hash
from .embedding import modify_spectrum
from .features import extract_features
from .lpc import LPCConfig, analyze_lpc
from .metrics import quality_scores
from .spectral import SpectralConfig, analyze, synthesize
from .utils import read_jsonl, sha256_file, write_json, write_jsonl


def variants_for(audio,config,key):
    cfg=SpectralConfig(**config["spectral"]);profile=analyze_lpc(audio,16000,cfg,LPCConfig(**config["lpc"]))
    if profile.metadata["numerical_failure_frames"]:raise ValueError("LPC numerical failures")
    spectrum,_=analyze(audio,16000,cfg)
    variants={(method,delta):synthesize(modify_spectrum(spectrum,16000,key,delta,cfg,
                profile.weights if method=="SS_LPC" else None),16000,len(audio),cfg)
              for method in ["SS","SS_LPC"] for delta in config["evaluation"]["quality_deltas_db"]}
    peak=max(float(abs(x).max()) for x in variants.values())
    gain=min(1.,config["audio"]["joint_peak_limit"]/peak)
    return {k:v*gain for k,v in variants.items()},gain,peak,profile


def control_ids(rows,count):
    from .formal_data import order_key
    ids=set()
    for gender in ["F","M"]:
        speakers=sorted({r["speaker_id"] for r in rows if r["gender"]==gender},key=lambda s:order_key(20261018,"zero",s))
        if len(speakers)<count//2:raise ValueError("Not enough control speakers")
        for speaker in speakers[:count//2]:
            candidates=[r for r in rows if r["speaker_id"]==speaker]
            ids.add(min(candidates,key=lambda r:order_key(20261018,"zero",r["utterance_id"]))["utterance_id"])
    return ids


def create_received(root,out,config,key,row,condition,kind,label,method,delta,embedded,reference):
    uid=row["utterance_id"];cid=condition["id"];stem=f"{uid}_{cid}_{kind}_label{label}"
    destination=out/"audio"/(stem+".wav")
    x,sr=read_audio(embedded)
    received,rate,metadata=apply_channel(x,sr,condition["channel"],destination,reference,uid,20261017)
    cfg=SpectralConfig(**config["spectral"])
    f=extract_features(received,rate,key,cfg,vad=config["evaluation"]["vad"]["mode"],
                       vad_aggressiveness=config["evaluation"]["vad"]["aggressiveness"])
    cache=out/"features"/(stem+".npz")
    np.savez_compressed(cache,features=f.features[f.valid_mask].astype(np.float32),features_f64=f.features,
                        valid_mask=f.valid_mask,times_s=f.times_s)
    base={k:row[k] for k in ["utterance_id","parent_utterance_id","speaker_id","gender","split"]}
    base.update(id=stem,condition=cid,channel=condition["channel"],kind=kind,label=label,method=method,delta_db=delta,
                nominal_delta_db=condition.get("delta_db",delta))
    artifact={**base,"waveform_path":str(destination.relative_to(root)),"waveform_sha256":sha256_file(destination),
        "pcm_sha256":pcm_hash(received),"embedded_path":str(embedded.relative_to(root)),
        "embedded_sha256":sha256_file(embedded),"sample_rate":rate,"samples":len(received),"peak":float(abs(received).max())}
    feature={**base,"features_path":str(cache.relative_to(out)),"features_sha256":sha256_file(cache),
        "valid_frames":int(f.valid_mask.sum()),"waveform_sha256":artifact["waveform_sha256"]}
    coverage={**base,**f.metadata,"supports_30_frames":int(f.valid_mask.sum())>=30}
    return artifact,feature,{**base,**metadata},coverage


def append_manifest(out,name,rows):
    path=out/name;previous=read_jsonl(path) if path.exists() else []
    write_jsonl(path,previous+rows)


def prepare_partition(root: Path,out: Path,config,key,split,selection=None):
    if split=="test" and not (out/"evaluation_gate.json").exists():raise ValueError("Test preparation requires frozen evaluation gate")
    rows=[r for r in read_jsonl(root/config["data"]["manifest"]) if r["split"]==split]
    if not rows:raise ValueError("Empty partition")
    zero=control_ids(rows,config["evaluation"]["zero_control_utterances_per_split"]) if split!="train" else set()
    artifacts=[];features=[];metadata=[];coverage=[];quality=[];references=[]
    for i,row in enumerate(rows):
        uid=row["utterance_id"]
        if sha256_file(root/row["waveform_path"])!=row["waveform_sha256"]:raise ValueError("Source hash mismatch")
        audio,sr=read_audio(root/row["waveform_path"])
        if sr!=16000:raise ValueError("Embedding requires 16 kHz")
        variants,gain,peak,profile=variants_for(audio,config,key)
        reference_path=out/"references"/(uid+".wav");write_audio(reference_path,audio*gain,sr)
        reference,_=read_audio(reference_path)
        references.append({**{k:row[k] for k in ["utterance_id","split","speaker_id","gender"]},"common_gain":gain,
            "maximum_unscaled_peak":peak,"reference_path":str(reference_path.relative_to(root)),
            "reference_sha256":sha256_file(reference_path),"lpc_metadata":profile.metadata})
        embedded={}
        choices=variants.keys() if split!="train" else [("SS",0),("SS",1),("SS",3)]
        for method,delta in choices:
            path=out/"embedded"/f"{uid}_{method}_d{delta:g}.wav";write_audio(path,variants[(method,delta)],sr)
            embedded[(method,delta)]=path
            if split!="train":
                saved,_=read_audio(path);q=quality_scores(reference,saved,sr)
                quality.append({"utterance_id":uid,"parent_utterance_id":row["parent_utterance_id"],"speaker_id":row["speaker_id"],
                    "gender":row["gender"],"split":split,"method":method,"delta_db":delta,
                    "pesq_nb":q["pesq_nb"],"pesq_wb":q["pesq_wb"],"embedding_snr_db":q["embedding_snr_db"],
                    "common_gain":gain,"scope":"pre_channel_embedding_quality","quality_failures":q["quality_failures"],
                    "embedded_path":str(path.relative_to(root)),"embedded_sha256":sha256_file(path)})
        for c in config["conditions"]:
            for kind in (["main","zero_control"] if uid in zero else ["main"]):
                for label in [0,1]:
                    delta=c["delta_db"] if kind=="main" and label else 0
                    values=create_received(root,out,config,key,row,c,kind,label,"SS",delta,embedded[("SS",delta)],reference)
                    for target,value in zip([artifacts,features,metadata,coverage],values):target.append(value)
        if split!="train" and selection is not None:
            for channel in config["evaluation"]["noise_channels"]:
                for method,delta in [("SS",selection["ss_delta_db"]),("SS_LPC",selection["lpc_delta_db"])]:
                    condition={"id":f"E03_{channel.split('_')[0]}_{method}","channel":channel,"delta_db":delta}
                    for label in [0,1]:
                        strength=delta if label else 0
                        values=create_received(root,out,config,key,row,condition,"main",label,method,strength,
                                               embedded[(method,delta)] if label else embedded[("SS",0)],reference)
                        for target,value in zip([artifacts,features,metadata,coverage],values):target.append(value)
        if (i+1)%20==0 or i+1==len(rows):print(f"Prepare {split} {i+1}/{len(rows)}",flush=True)
    for name,data in [("audio_manifest.jsonl",artifacts),("feature_manifest.jsonl",features),("channel_manifest.jsonl",metadata),
                      ("coverage.jsonl",coverage),("quality_scores.jsonl",quality),("reference_manifest.jsonl",references)]:
        append_manifest(out,name,data)
    return {"split":split,"sources":len(rows),"received":len(artifacts),"quality_failures":sum(bool(r["quality_failures"]) for r in quality),
            "insufficient_30_frame_files":sum(not r["supports_30_frames"] for r in coverage)}


def prepare_validation_noise(root,out,config,key,selection):
    rows=[r for r in read_jsonl(root/config["data"]["manifest"]) if r["split"]=="validation"]
    artifacts=[];features=[];metadata=[];coverage=[]
    refs={r["utterance_id"]:r for r in read_jsonl(out/"reference_manifest.jsonl")}
    for row in rows:
        uid=row["utterance_id"];reference,_=read_audio(root/refs[uid]["reference_path"])
        for channel in config["evaluation"]["noise_channels"]:
            for method,delta in [("SS",selection["ss_delta_db"]),("SS_LPC",selection["lpc_delta_db"])]:
                for label in [0,1]:
                    strength=delta if label else 0;name=f"{uid}_{method}_d{delta:g}.wav" if label else f"{uid}_SS_d0.wav"
                    values=create_received(root,out,config,key,row,{"id":f"E03_{channel.split('_')[0]}_{method}","channel":channel,"delta_db":delta},
                                           "main",label,method,strength,out/"embedded"/name,reference)
                    for target,value in zip([artifacts,features,metadata,coverage],values):target.append(value)
    for name,data in [("audio_manifest.jsonl",artifacts),("feature_manifest.jsonl",features),("channel_manifest.jsonl",metadata),("coverage.jsonl",coverage)]:append_manifest(out,name,data)
    return len(artifacts)
