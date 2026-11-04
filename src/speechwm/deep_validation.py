from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from .audio import read_audio
from .channels import inspect_opus
from .comparison import pcm_hash
from .deep_data import SPLITS, audit_splits, load_entries
from .deep_evaluation import calibrate_validation, entry_scores, fixed_threshold_rates, metric_table
from .features import aggregate, extract_features
from .keys import load_key
from .models import PARAMETERS, load_checkpoint
from .metrics import equal_error_rate
from .spectral import SpectralConfig
from .training import fit_model
from .utils import canonical_hash, read_jsonl, sha256_file


def validate_deep(out: Path,replay_training=True):
    root=next(p for p in out.resolve().parents if (p/"pyproject.toml").exists())
    config=yaml.safe_load((out/"config.yaml").read_text());summary=json.loads((out/"metrics.json").read_text())
    assert canonical_hash(config)==summary["config_hash"]
    assert sha256_file(out/"split_manifest.jsonl")==summary["split_manifest_sha256"]
    assert sha256_file(out/"feature_manifest.jsonl")==summary["feature_manifest_sha256"]
    source=read_jsonl(out/"split_manifest.jsonl");audit=audit_splits(source)
    assert audit==json.loads((out/"split_audit.json").read_text())
    for r in source:assert sha256_file(root/r["waveform_path"])==r["waveform_sha256"]
    cfg=SpectralConfig(**config["spectral"]);key=load_key(out/"key.json")
    artifacts=read_jsonl(out/"audio_manifest.jsonl");features=read_jsonl(out/"feature_manifest.jsonl")
    metadata=read_jsonl(out/"channel_manifest.jsonl")
    assert len(artifacts)==len(features)==len(metadata)==160
    feature_index={r["id"]:r for r in features};meta_index={r["id"]:r for r in metadata}
    zero_pairs=defaultdict(list);min_valid=100000;opus_count=0
    for item in artifacts:
        x,sr=read_audio(root/item["waveform_path"])
        assert sr==8000 and sr==item["sample_rate"] and len(x)==item["samples"] and abs(x).max()<1
        assert sha256_file(root/item["waveform_path"])==item["waveform_sha256"] and pcm_hash(x)==item["pcm_sha256"]
        assert sha256_file(root/item["embedded_path"])==item["embedded_sha256"]
        m=meta_index[item["id"]];assert len(x)==(m["input_samples"]+1)//2
        f=extract_features(x,sr,key,cfg,vad=config["evaluation"]["vad"]["mode"],vad_aggressiveness=config["evaluation"]["vad"]["aggressiveness"])
        entry=feature_index[item["id"]];assert sha256_file(out/entry["features_path"])==entry["features_sha256"]
        with np.load(out/entry["features_path"],allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["features_f64"],f.features)
            np.testing.assert_array_equal(saved["features"],f.features[f.valid_mask].astype(np.float32))
            np.testing.assert_array_equal(saved["valid_mask"],f.valid_mask)
            np.testing.assert_array_equal(saved["times_s"],f.times_s)
        min_valid=min(min_valid,int(f.valid_mask.sum()))
        if item["kind"]=="zero_control":zero_pairs[(item["utterance_id"],item["condition"])].append(pcm_hash(x))
        if item["condition"]=="C3":
            compressed=(root/item["waveform_path"]).with_suffix(".opus")
            assert sha256_file(compressed)==m["compressed_sha256"] and inspect_opus(compressed)==m["inspection"]
            opus_count+=1
    assert all(len(v)==2 and len(set(v))==1 for v in zero_pairs.values())
    train_summary=json.loads((out/"training_summary.json").read_text())["models"]
    models={c["id"]:{} for c in config["conditions"]};replays=0
    gate=json.loads((out/"evaluation_gate.json").read_text())
    assert gate["primary_epoch"]==25 and not gate["eval_partition_used_for_training_or_selection"]
    assert gate["config_hash"]==summary["config_hash"]
    for r in train_summary:
        path=out/r["checkpoint_path"];model,checkpoint=load_checkpoint(path,key)
        name=f"{r['condition']}_{r['architecture']}"
        assert sha256_file(path)==r["checkpoint_sha256"]==gate["checkpoint_hashes"][name]
        assert checkpoint["epoch"]==25 and checkpoint["parameters"]==PARAMETERS[r["architecture"]]
        assert checkpoint["binding"]["config_hash"]==summary["config_hash"]
        assert checkpoint["binding"]["split_manifest_sha256"]==summary["split_manifest_sha256"]
        assert not set(checkpoint["binding"]["train_parent_ids"])&set(checkpoint["binding"]["validation_parent_ids"])
        for split,name_hash in [("dev_train","train_feature_manifest_hash"),("dev_validation","validation_feature_manifest_hash")]:
            assert checkpoint["binding"][name_hash]==canonical_hash([m for m in features if m["condition"]==r["condition"] and m["kind"]=="main" and m["split"]==split])
        assert sha256_file(out/"checkpoints"/f"{name}_validation_best.pt")==r["best_checkpoint_sha256"]
        models[r["condition"]][r["architecture"]]=model
        if replay_training:
            trained=fit_model(load_entries(out,"dev_train",r["condition"]),load_entries(out,"dev_validation",r["condition"]),r["architecture"],config["training"])
            assert trained["state_sha256"]==r["state_sha256"] and trained["updates"]==r["updates"]
            history=list(csv.DictReader((out/"training"/f"{name}_history.csv").open()))
            assert len(history)==25
            for saved,current in zip(history,trained["history"]):
                for field,value in current.items():
                    if isinstance(value,str):assert saved[field]==value
                    else:assert float(saved[field])==value
            replays+=1
    # Re-infer frames and re-aggregate groups independently of the saved score CSVs.
    frame_count=0;max_error=0.
    aggregate_count=0;definition_count=0
    with (out/"frame_scores.csv").open() as stream, (out/"aggregate_scores.csv").open() as aggregate_stream, (out/"group_definitions.csv").open() as definition_stream:
        reader=csv.DictReader(stream)
        aggregate_reader=csv.DictReader(aggregate_stream);definition_reader=csv.DictReader(definition_stream)
        for condition in config["conditions"]:
            for split in SPLITS:
                for kind in ["main","zero_control"]:
                    for entry in load_entries(out,split,condition["id"],kind):
                        scores=entry_scores(entry,models[condition["id"]])
                        base={k:entry[k] for k in ["id","utterance_id","parent_utterance_id","speaker_id","split","condition","kind","label"]}
                        def check_record(saved,expected):
                            assert set(saved)==set(expected)
                            for field,value in expected.items():
                                if isinstance(value,str):assert saved[field]==value
                                else:assert float(saved[field])==value
                        for detector,values in scores.items():
                            for index,time,score in zip(entry["frame_indices"],entry["times"],values):
                                saved=next(reader)
                                for field,value in base.items():assert saved[field]==str(value)
                                assert saved["id"]==entry["id"] and saved["detector"]==detector and int(saved["frame_index"])==index
                                assert float(saved["time_s"])==time
                                error=abs(float(saved["score"])-score);assert error<=1e-6;max_error=max(max_error,error);frame_count+=1
                        for frames in config["evaluation"]["aggregate_frames"]:
                            for group in aggregate(np.zeros(len(entry["x"])),entry["times"],frames):
                                check_record(next(definition_reader),{**base,**{k:v for k,v in group.items() if k!="score"}})
                                definition_count+=1
                            for detector,values in scores.items():
                                for group in aggregate(values,entry["times"],frames):
                                    check_record(next(aggregate_reader),{**base,"detector":detector,**group})
                                    aggregate_count+=1
        assert next(reader,None) is None
        assert next(aggregate_reader,None) is None and next(definition_reader,None) is None
    rows=[]
    with (out/"aggregate_scores.csv").open() as stream:
        for r in csv.DictReader(stream):
            for field in ["label","frames","group_index","start_frame_in_valid"]:r[field]=int(r[field])
            for field in ["score","start_s","end_s"]:r[field]=float(r[field])
            rows.append(r)
    assert metric_table(rows)==summary["metrics"]
    definitions={}
    for row in rows:
        definition={k:v for k,v in row.items() if k not in {"detector","score"}}
        k=(row["id"],row["frames"],row["group_index"])
        if k in definitions:assert definition==definitions[k]
        else:definitions[k]=definition
    saved_definitions=list(csv.DictReader((out/"group_definitions.csv").open()))
    assert len(saved_definitions)==len(definitions)==definition_count
    calibration=json.loads((out/"calibration.json").read_text())
    assert gate["calibration_sha256"]==sha256_file(out/"calibration.json")
    assert calibration==calibrate_validation([r for r in rows if r["split"]=="dev_validation" and r["kind"]=="main"],gate["checkpoint_hashes"])
    rates=fixed_threshold_rates(rows,calibration)
    persisted_rates=list(csv.DictReader((out/"fixed_threshold_rates.csv").open()))
    for old,new in zip(persisted_rates,rates):
        for k,v in new.items():assert old[k]==v if isinstance(v,str) else float(old[k])==v
    assert len(rates)==len(persisted_rates)
    previous=json.loads((root/"milestones/M2/evidence/metrics.json").read_text())["metrics"]
    regression=[]
    for c in config["conditions"]:
        for frames in config["evaluation"]["aggregate_frames"]:
            values=[r for r in rows if r["condition"]==c["id"] and r["kind"]=="main" and r["detector"]=="correlation" and r["frames"]==frames]
            current=equal_error_rate(np.array([r["label"] for r in values]),np.array([r["score"] for r in values]))
            old=next(r for r in previous if r["delta_db"]==c["delta_db"] and r["channel"]==c["channel"] and r["frames"]==frames and r["detector"]=="correlation")
            assert current["eer"]==old["eer"] and current["samples"]==old["samples"]
            regression.append({"condition":c["id"],"frames":frames,"eer":current["eer"],"samples":current["samples"],"identical_to_m2":True})
    assert regression==json.loads((out/"m2_baseline_regression.json").read_text())["settings"]
    null=[r for r in summary["metrics"] if r["kind"]=="zero_control"]
    assert len(null)==144 and all(r["eer"]==.5 for r in null)
    assert frame_count==summary["frame_score_rows"] and len(rows)==summary["aggregate_score_rows"]==aggregate_count
    assert len(definitions)==summary["group_definition_rows"] and min_valid>=30
    frozen={}
    for stage in ["M1","M2","M3"]:
        folder=root/"milestones"/stage;manifest=json.loads((folder/"snapshot_manifest.json").read_text())
        for item in manifest["files"]:assert sha256_file(folder/item["path"])==item["sha256"]
        frozen[stage]=len(manifest["files"])
    assert not read_jsonl(out/"failures.jsonl")
    return {"passed":True,"received_wavs_and_feature_caches_verified":len(artifacts),"opus_files_verified":opus_count,
        "split_parent_speaker_hash_disjoint":True,"split_audit":audit,"zero_pcm_pairs":len(zero_pairs),
        "zero_eer_settings":len(null),"checkpoints_verified":len(train_summary),"models_retrained_identically":replays,
        "training_replay_requested":replay_training,"frame_score_rows_reinferred":frame_count,
        "maximum_checkpoint_logit_replay_error":max_error,"metric_rows_recomputed":len(summary["metrics"]),
        "aggregate_score_rows_recomputed":aggregate_count,
        "common_group_definitions_verified":len(definitions),"validation_only_calibrations":len(calibration["settings"]),
        "minimum_valid_frames":min_valid,"legacy_snapshot_files_verified":frozen,"failures":0,
        "m2_baseline_regression_settings":len(regression),
        "scope":"same-environment development replay; one original utterance per split, not formal performance"}
