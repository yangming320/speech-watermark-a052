"""Select a fixed, speaker-disjoint public corpus before model evaluation."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

import numpy as np
import soundfile as sf
import yaml

from .audio import prepare_audio, read_audio, write_audio
from .comparison import pcm_hash
from .utils import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl

ROLES=("train","validation","test")


def order_key(seed,role,identifier):
    return hashlib.sha256(f"{seed}|{role}|{identifier}".encode()).hexdigest()


def check_disjoint(rows,excluded=()):
    if {r["split"] for r in rows}!=set(ROLES):raise ValueError("Require train/validation/test")
    for field in ["speaker_id","parent_utterance_id","source_sha256","source_pcm_sha256","pcm_sha256"]:
        assignments={}
        for r in rows:assignments.setdefault(r[field],set()).add(r["split"])
        if any(len(v)>1 for v in assignments.values()):raise ValueError(f"Cross-partition leakage: {field}")
    if len({r["utterance_id"] for r in rows})!=len(rows):raise ValueError("Duplicate utterance ID")
    if set(excluded)&{r["speaker_id"] for r in rows}:raise ValueError("Historical smoke speaker contamination")
    return {s:{"utterances":sum(r["split"]==s for r in rows),
        "speakers":len({r["speaker_id"] for r in rows if r["split"]==s}),
        "duration_s":sum(r["duration_s"] for r in rows if r["split"]==s),
        "gender_utterances":{g:sum(r["split"]==s and r["gender"]==g for r in rows) for g in ["F","M"]},
        "speaker_ids":sorted({r["speaker_id"] for r in rows if r["split"]==s},key=int)} for s in ROLES}


def select_catalog(catalog,config):
    # Selection uses metadata/duration only, never VAD, PESQ, labels or detector scores.
    seed=config["seed"];policy=config["selection"];selected=[];used=set(config["exclude_speakers"])
    for role in ROLES:
        pool=policy.get("pool_by_split",{}).get(role,"test_pool" if role=="test" else "training_pool")
        rule=policy[role]
        for gender in ["F","M"]:
            candidates={}
            for r in catalog:
                if r["pool"]==pool and r["gender"]==gender and r["speaker_id"] not in used and r["duration_s"]>=policy["clip_seconds"]:
                    candidates.setdefault(r["speaker_id"],[]).append(r)
            eligible=[s for s,v in candidates.items() if len(v)>=rule["utterances_per_speaker"]]
            eligible.sort(key=lambda s:order_key(seed,role,s))
            if len(eligible)<rule["speakers_per_gender"]:raise ValueError(f"Not enough eligible speakers: {role}/{gender}: {len(eligible)}")
            speakers=eligible[:rule["speakers_per_gender"]];used.update(speakers)
            for speaker in speakers:
                utterances=sorted(candidates[speaker],key=lambda r:order_key(seed,role,r["parent_utterance_id"]))
                selected.extend({**r,"split":role} for r in utterances[:rule["utterances_per_speaker"]])
    return selected


def prepare_formal(root: Path,config_path: Path):
    config=yaml.safe_load(config_path.read_text());policy=config["selection"]
    if policy["clip_seconds"]!=3.0 or policy["sample_rate"]!=16000:raise ValueError("M5 v1 fixes centered 3 s at 16 kHz")
    raw=root/config["raw_directory"];processed=root/config["processed_directory"];destination=root/config["manifest_directory"]
    if destination.exists():raise FileExistsError("Refusing to overwrite frozen formal dataset")
    destination.mkdir(parents=True);raw.mkdir(parents=True,exist_ok=True)
    catalog=[];archives=[]
    for source in config["archives"]:
        archive=root/config["cache_directory"]/source["name"]
        md5=hashlib.md5()
        with archive.open("rb") as f:
            while block:=f.read(1024*1024):md5.update(block)
        if md5.hexdigest()!=source["md5"]:raise ValueError("Official archive MD5 mismatch")
        archives.append({**source,"size_bytes":archive.stat().st_size,"sha256":sha256_file(archive)})
        # Copy only corpus metadata and selected originals; the full download stays in cache.
        metadata_dir=raw/source["role"];metadata_dir.mkdir(exist_ok=True)
        with tarfile.open(archive,"r:gz") as tar:
            members=tar.getmembers()
            available={m.name for m in members};origins={}
            for name in ["LICENSE.TXT","README.TXT","SPEAKERS.TXT","CHAPTERS.TXT"]:
                member="LibriSpeech/"+name
                if member in available:
                    content=tar.extractfile(member).read();origins[name]={"archive":source["name"],"member":member}
                elif name!="README.TXT" and (raw/"training_pool"/name).exists():
                    shared=raw/"training_pool"/name;content=shared.read_bytes()
                    origins[name]={"shared_official_metadata_from":str(shared.relative_to(root)),"reason":"not bundled in this archive"}
                else:continue
                (metadata_dir/name).write_bytes(content)
                origins[name]["sha256"]=hashlib.sha256(content).hexdigest()
            write_json(metadata_dir/"METADATA_SOURCE.json",origins)
            people={}
            for line in (metadata_dir/"SPEAKERS.TXT").read_text().splitlines():
                if not line.strip() or line.startswith(";"):continue
                fields=[s.strip() for s in line.split("|")]
                people[fields[0]]={"gender":fields[1],"official_subset":fields[2],"speaker_name":fields[4]}
            for member in members:
                if not member.isfile() or not member.name.endswith(".flac"):continue
                uid=PurePosixPath(member.name).stem;speaker=uid.split("-")[0]
                content=tar.extractfile(member).read();info=sf.info(io.BytesIO(content))
                if info.samplerate!=16000 or info.channels!=1:raise ValueError("Unexpected LibriSpeech format")
                catalog.append({"parent_utterance_id":uid,"speaker_id":speaker,**people[speaker],"pool":source["role"],
                    "archive":source["name"],"member":member.name,"sample_rate":info.samplerate,"samples":info.frames,
                    "duration_s":info.duration,"source_sha256":hashlib.sha256(content).hexdigest()})
        print(f"Indexed {source['name']}: {sum(r['archive']==source['name'] for r in catalog)} utterances",flush=True)
    chosen=select_catalog(catalog,config);write_jsonl(destination/"catalog.jsonl",catalog)
    write_jsonl(destination/"selection_before_audio_processing.jsonl",chosen)
    rows=[]
    for source in config["archives"]:
        archive=root/config["cache_directory"]/source["name"]
        with tarfile.open(archive,"r:gz") as tar:
            ordered=sorted([r for r in chosen if r["archive"]==source["name"]],key=lambda r:tar.getmember(r["member"]).offset_data)
            for r in ordered:
                uid=r["parent_utterance_id"];source_path=raw/r["pool"]/(uid+".flac")
                source_path.write_bytes(tar.extractfile(r["member"]).read())
                original,sr=read_audio(source_path);count=round(policy["clip_seconds"]*sr);start=(len(original)-count)//2
                audio,prep=prepare_audio(original[start:start+count],sr,policy["sample_rate"],policy["peak_limit"])
                path=processed/r["split"]/(uid+"_center3s.wav");write_audio(path,audio,sr)
                saved,_=read_audio(path)
                rows.append({"utterance_id":uid+"_center3s","parent_utterance_id":uid,"speaker_id":r["speaker_id"],
                    "gender":r["gender"],"gender_source":"official SPEAKERS.TXT","speaker_name":r["speaker_name"],
                    "split":r["split"],"dataset_id":config["dataset_id"],"source_archive":r["archive"],
                    "source_member":r["member"],"official_subset":r["official_subset"],
                    "source_path":str(source_path.relative_to(root)),"source_sha256":r["source_sha256"],
                    "source_pcm_sha256":pcm_hash(original),"waveform_path":str(path.relative_to(root)),
                    "waveform_sha256":sha256_file(path),"pcm_sha256":pcm_hash(saved),"channels":1,"sample_rate":sr,
                    "duration_s":len(audio)/sr,"source_duration_s":len(original)/sr,"start_sample_in_parent":start,
                    "end_sample_in_parent":start+count,"preparation":prep,"scope":config["scope"]})
    rows.sort(key=lambda r:(ROLES.index(r["split"]),int(r["speaker_id"]),r["utterance_id"]))
    audit=check_disjoint(rows,config["exclude_speakers"])
    write_jsonl(destination/"all.jsonl",rows)
    for role in ROLES:write_jsonl(destination/(role+".jsonl"),[r for r in rows if r["split"]==role])
    write_json(destination/"sources.json",{"archives":archives,"policy":config,"policy_hash":canonical_hash(config),
        "compressed_audio_provenance":"Distributed FLAC decoded losslessly; LibriSpeech originated from LibriVox recordings, not TIMIT studio recordings",
        "archive_downloads_excluded_from_release":"Full source tarballs cached; all selected original FLAC and corpus metadata included in data/raw"})
    write_json(destination/"split_audit.json",audit)
    write_json(destination/"freeze.json",{"dataset_id":config["dataset_id"],"manifest_sha256":sha256_file(destination/"all.jsonl"),
        "selection_sha256":sha256_file(destination/"selection_before_audio_processing.jsonl"),"policy_hash":canonical_hash(config),
        "speaker_disjoint":True,"historical_smoke_speakers_excluded":config["exclude_speakers"],
        "selected_without_detector_or_quality_scores":True,"audit":audit})
    return destination


def validate_formal(root: Path,directory: Path):
    rows=read_jsonl(directory/"all.jsonl");freeze=json.loads((directory/"freeze.json").read_text())
    policy=json.loads((directory/"sources.json").read_text())["policy"]
    assert canonical_hash(policy)==freeze["policy_hash"]
    assert sha256_file(directory/"selection_before_audio_processing.jsonl")==freeze["selection_sha256"]
    chosen=select_catalog(read_jsonl(directory/"catalog.jsonl"),policy)
    assert chosen==read_jsonl(directory/"selection_before_audio_processing.jsonl")
    expected={r["parent_utterance_id"]:(r["split"],r["speaker_id"],r["gender"],r["source_sha256"]) for r in chosen}
    assert expected=={r["parent_utterance_id"]:(r["split"],r["speaker_id"],r["gender"],r["source_sha256"]) for r in rows}
    assert sha256_file(directory/"all.jsonl")==freeze["manifest_sha256"]
    assert check_disjoint(rows,freeze["historical_smoke_speakers_excluded"])==freeze["audit"]
    for role in ROLES:assert read_jsonl(directory/(role+".jsonl"))==[r for r in rows if r["split"]==role]
    for r in rows:
        source,sr=read_audio(root/r["source_path"]);audio,rate=read_audio(root/r["waveform_path"])
        assert sr==rate==r["sample_rate"]==16000 and sha256_file(root/r["source_path"])==r["source_sha256"]
        assert sha256_file(root/r["waveform_path"])==r["waveform_sha256"]
        assert pcm_hash(source)==r["source_pcm_sha256"] and pcm_hash(audio)==r["pcm_sha256"]
        expected,prep=prepare_audio(source[r["start_sample_in_parent"]:r["end_sample_in_parent"]],sr)
        np.testing.assert_array_equal(expected.astype(np.float32).astype(np.float64),audio)
        assert prep==r["preparation"]
    return {"passed":True,"utterances_verified":len(rows),"audit":freeze["audit"]}
