from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from .audio import read_audio, resample_audio
from .channels import inspect_opus, paired_noise
from .comparison import pcm_hash
from .deep_data import load_entries
from .features import aggregate, extract_features
from .formal_data import validate_formal
from .main_data import variants_for
from .main_evaluation import read_statistics, scores_for, seed_summary
from .main_runner import select_validation_quality
from .metrics import quality_scores
from .models import load_checkpoint
from .spectral import SpectralConfig
from .training import fit_model
from .utils import canonical_hash, read_jsonl, sha256_file


def check_csv_record(saved,expected,score_tolerance=0.):
    assert set(saved)==set(expected)
    maximum=0.
    for field,value in expected.items():
        if isinstance(value,str):assert saved[field]==value
        else:
            error=abs(float(saved[field])-value)
            assert error<=(score_tolerance if field=="score" else 0.),(field,saved[field],value)
            if field=="score":maximum=max(maximum,error)
    return maximum


def audit_scores(out,config,models,split):
    counts={"frame_score_rows":0,"aggregate_score_rows":0,"group_definition_rows":0};maximum=0.
    with (out/f"{split}_frame_scores.csv").open() as frames, (out/f"{split}_aggregate_scores.csv").open() as groups, (out/f"{split}_group_definitions.csv").open() as definitions:
        fr=csv.DictReader(frames);gr=csv.DictReader(groups);dr=csv.DictReader(definitions)
        conditions=sorted({r["condition"] for r in read_jsonl(out/"feature_manifest.jsonl") if r["split"]==split})
        for cid in conditions:
            for kind in ["main","zero_control"]:
                for entry in load_entries(out,split,cid,kind):
                    values=list(scores_for(entry,models));base={"id":entry["id"]}
                    for detector,seed,scores in values:
                        for index,time,score in zip(entry["frame_indices"],entry["times"],scores):
                            maximum=max(maximum,check_csv_record(next(fr),{**base,"detector":detector,"seed":seed,
                                "frame_index":int(index),"time_s":float(time),"score":float(score)},1e-6));counts["frame_score_rows"]+=1
                    aggregation=config["evaluation"]["lpc_aggregate_frames"] if cid.startswith("E03_") else config["evaluation"]["aggregate_frames"]
                    for count in aggregation:
                        for group in aggregate(np.zeros(len(entry["x"])),entry["times"],count):
                            check_csv_record(next(dr),{**base,**{k:v for k,v in group.items() if k!="score"}});counts["group_definition_rows"]+=1
                        for detector,seed,scores in values:
                            for group in aggregate(scores,entry["times"],count):
                                maximum=max(maximum,check_csv_record(next(gr),{**base,"detector":detector,"seed":seed,**group},1e-6));counts["aggregate_score_rows"]+=1
        assert next(fr,None) is next(gr,None) is next(dr,None) is None
    return {**counts,"maximum_score_error":maximum}


def validate_main(out: Path,replay_seed=2026):
    root=next(p for p in out.resolve().parents if (p/"pyproject.toml").exists())
    config=yaml.safe_load((out/"config.yaml").read_text());summary=json.loads((out/"metrics.json").read_text())
    assert canonical_hash(config)==summary["config_hash"]
    dataset=validate_formal(root,root/config["data"]["dataset_directory"])
    assert sha256_file(out/"input_manifest.jsonl")==summary["dataset_manifest_sha256"]
    source={r["utterance_id"]:r for r in read_jsonl(out/"input_manifest.jsonl")}
    from .keys import load_key
    key=load_key(out/"key.json");cfg=SpectralConfig(**config["spectral"])
    gate=json.loads((out/"evaluation_gate.json").read_text())
    assert gate["primary_epoch"]==25 and gate["config_hash"]==summary["config_hash"] and not gate["test_used_for_training_selection_or_calibration"]
    assert gate["dataset_manifest_sha256"]==summary["dataset_manifest_sha256"]
    quality=read_jsonl(out/"quality_scores.jsonl");selected=json.loads((out/"quality_selection.json").read_text())
    val_quality=[r for r in quality if r["split"]=="validation"]
    assert selected==select_validation_quality(val_quality,config)
    assert gate["quality_selection_sha256"]==sha256_file(out/"quality_selection.json")
    quality_gate=json.loads((out/"quality_gate.json").read_text())
    assert not quality_gate["test_quality_computed"] and quality_gate["validation_quality_rows_hash"]==canonical_hash(val_quality)
    assert quality_gate["selection_sha256"]==gate["quality_selection_sha256"]
    references=read_jsonl(out/"reference_manifest.jsonl");reference_index={r["utterance_id"]:r for r in references}
    embedded_verified=0;quality_recomputed=0;quality_strata=set();quality_error=0.;gains=[]
    # Recompute every embedding waveform; rerun PESQ on every split/gender/method/strength stratum.
    for record in references:
        row=source[record["utterance_id"]];audio,sr=read_audio(root/row["waveform_path"])
        variants,gain,peak,profile=variants_for(audio,config,key);gains.append(gain)
        assert gain==record["common_gain"] and peak==record["maximum_unscaled_peak"] and profile.metadata==record["lpc_metadata"]
        reference,_=read_audio(root/record["reference_path"])
        assert sha256_file(root/record["reference_path"])==record["reference_sha256"]
        np.testing.assert_array_equal((audio*gain).astype(np.float32).astype(np.float64),reference)
        choices=variants.keys() if row["split"]!="train" else [("SS",0),("SS",1),("SS",3)]
        for method,delta in choices:
            path=out/"embedded"/f"{row['utterance_id']}_{method}_d{delta:g}.wav"
            saved,rate=read_audio(path);assert rate==sr
            np.testing.assert_array_equal(variants[(method,delta)].astype(np.float32).astype(np.float64),saved);embedded_verified+=1
            if row["split"]!="train":
                original=next(r for r in quality if r["utterance_id"]==row["utterance_id"] and r["method"]==method and r["delta_db"]==delta)
                assert sha256_file(path)==original["embedded_sha256"]
                stratum=(row["split"],row["gender"],method,delta)
                if stratum not in quality_strata:
                    q=quality_scores(reference,saved,sr);assert q["quality_failures"]==original["quality_failures"]
                    for field in ["pesq_nb","pesq_wb"]:
                        if q[field] is not None:
                            error=abs(q[field]-original[field]);assert error<=1e-6;quality_error=max(quality_error,error)
                    quality_recomputed+=1;quality_strata.add(stratum)
    artifacts=read_jsonl(out/"audio_manifest.jsonl");features=read_jsonl(out/"feature_manifest.jsonl")
    metadata={r["id"]:r for r in read_jsonl(out/"channel_manifest.jsonl")};fi={r["id"]:r for r in features}
    assert len(artifacts)==len(features)==len(metadata)
    pcm={};opus=0;noise=0;minimum=100000;zero_pairs={}
    for r in artifacts:
        row=source[r["utterance_id"]]
        for field in ["speaker_id","parent_utterance_id","gender","split"]:assert r[field]==row[field]
        channel=metadata[r["id"]];x,sr=read_audio(root/r["waveform_path"])
        assert sr==8000 and len(x)==r["samples"]==24000 and abs(x).max()<1
        assert sha256_file(root/r["waveform_path"])==r["waveform_sha256"] and pcm_hash(x)==r["pcm_sha256"]
        assert sha256_file(root/r["embedded_path"])==r["embedded_sha256"]
        expected_delta=0
        if r["label"] and r["kind"]=="main":
            expected_delta=(selected["lpc_delta_db"] if r["method"]=="SS_LPC" else selected["ss_delta_db"]) if r["condition"].startswith("E03_") else next(c["delta_db"] for c in config["conditions"] if c["id"]==r["condition"])
        assert r["delta_db"]==expected_delta
        if r["channel"].startswith("noise"):
            reference,_=read_audio(root/reference_index[r["utterance_id"]]["reference_path"])
            snr=20 if r["channel"].startswith("noise20") else 15
            vector,stats=paired_noise(reference,r["utterance_id"],20261017,snr)
            assert stats["noise_sha256"]==channel["noise_sha256"]
            embedded,_=read_audio(root/r["embedded_path"])
            np.testing.assert_array_equal(resample_audio(embedded+vector,16000,8000).astype(np.float32).astype(np.float64),x);noise+=1
        if r["channel"]=="opus8k_8kbps":
            compressed=(root/r["waveform_path"]).with_suffix(".opus")
            assert sha256_file(compressed)==channel["compressed_sha256"] and inspect_opus(compressed)==channel["inspection"];opus+=1
        f=extract_features(x,sr,key,cfg,vad=config["evaluation"]["vad"]["mode"],vad_aggressiveness=config["evaluation"]["vad"]["aggressiveness"])
        entry=fi[r["id"]];assert sha256_file(out/entry["features_path"])==entry["features_sha256"]
        with np.load(out/entry["features_path"],allow_pickle=False) as cache:
            np.testing.assert_array_equal(cache["features_f64"],f.features)
            np.testing.assert_array_equal(cache["features"],f.features[f.valid_mask].astype(np.float32))
            np.testing.assert_array_equal(cache["times_s"],f.times_s);np.testing.assert_array_equal(cache["valid_mask"],f.valid_mask)
        minimum=min(minimum,int(f.valid_mask.sum()))
        if r["kind"]=="zero_control":zero_pairs.setdefault((r["utterance_id"],r["condition"]),[]).append(r["pcm_sha256"])
        if not r["label"]:
            pair=(r["utterance_id"],r["channel"])
            if pair in pcm:assert pcm[pair]==r["pcm_sha256"]
            else:pcm[pair]=r["pcm_sha256"]
    assert all(len(v)==2 and len(set(v))==1 for v in zero_pairs.values())
    training=json.loads((out/"training_summary.json").read_text());assert training["complete"] and len(training["models"])==12
    models={c["id"]:{} for c in config["conditions"]};replays=[]
    for r in training["models"]:
        path=out/r["checkpoint_path"];model,checkpoint=load_checkpoint(path,key)
        assert sha256_file(path)==r["checkpoint_sha256"]==gate["checkpoint_hashes"][r["name"]]
        assert checkpoint["epoch"]==25 and checkpoint["state_sha256"]==r["state_sha256"]
        assert checkpoint["binding"]["config_hash"]==summary["config_hash"]
        assert checkpoint["binding"]["split_manifest_sha256"]==summary["dataset_manifest_sha256"]
        assert checkpoint["training_config"]["seed"]==r["seed"]
        for split,field in [("train","train_parent_ids"),("validation","validation_parent_ids")]:
            assert checkpoint["binding"][field]==sorted({x["parent_utterance_id"] for x in source.values() if x["split"]==split})
        for split,field in [("train","train_feature_manifest_hash"),("validation","validation_feature_manifest_hash")]:
            assert checkpoint["binding"][field]==canonical_hash([x for x in features if x["condition"]==r["condition"] and x["split"]==split and x["kind"]=="main"])
        assert sha256_file(out/"checkpoints"/(r["name"]+"_validation_best.pt"))==r["best_checkpoint_sha256"]
        models[r["condition"]].setdefault(r["seed"],{})[r["architecture"]]=model
        history=list(csv.DictReader((out/"training"/(r["name"]+"_history.csv")).open()));assert len(history)==25
        if r["seed"]==replay_seed:
            print(f"Replay {r['name']}",flush=True,file=sys.stderr)
            repeated=fit_model(load_entries(out,"train",r["condition"]),load_entries(out,"validation",r["condition"]),r["architecture"],checkpoint["training_config"],train_split="train",validation_split="validation")
            assert repeated["state_sha256"]==r["state_sha256"] and repeated["updates"]==r["updates"]
            for old,new in zip(history,repeated["history"]):check_csv_record(old,new)
            replays.append(r["name"])
    # CSV-to-model checks precede CSV-to-metrics checks.
    score_audits={};metrics=[];rates=[];calibration=json.loads((out/"calibration.json").read_text())
    assert sha256_file(out/"calibration.json")==gate["calibration_sha256"]
    for split in ["validation","test"]:
        score_audits[split]=audit_scores(out,config,models,split)
        for k,v in summary[split+"_counts"].items():assert score_audits[split][k]==v
        statistics=read_statistics(out,split);metrics+=statistics.metrics();rates+=statistics.rates(calibration)
        if split=="validation":assert calibration==statistics.calibration(gate["checkpoint_hashes"])
        del statistics
    assert metrics==summary["metrics"]
    for name,rows in [("metrics.csv",metrics),("seed_summary.csv",seed_summary(metrics)),("fixed_threshold_rates.csv",rates)]:
        with (out/name).open() as f:
            reader=csv.DictReader(f)
            for r in rows:check_csv_record(next(reader),r)
            assert next(reader,None) is None
    null=[r for r in metrics if r["kind"]=="zero_control"];assert null and all(r["eer"]==.5 for r in null)
    historical={}
    for stage in ["M1","M2","M3","M4"]:
        folder=root/"milestones"/stage;manifest=json.loads((folder/"snapshot_manifest.json").read_text())
        for entry in manifest["files"]:assert sha256_file(folder/entry["path"])==entry["sha256"]
        historical[stage]=len(manifest["files"])
    assert not read_jsonl(out/"failures.jsonl")
    return {"passed":True,"dataset_validation":dataset,"received_and_feature_caches_verified":len(artifacts),"embedded_wavs_recomputed":embedded_verified,
        "opus_verified":opus,"noise_wavs_recomputed":noise,"negative_pcm_groups_shared":len(pcm),"zero_pcm_pairs":len(zero_pairs),
        "zero_eer_settings":len(null),"checkpoints_verified":12,"fully_retrained_settings":replays,"replay_seed":replay_seed,
        "score_audits":score_audits,"metric_rows_recomputed":len(metrics),"validation_thresholds_verified":len(calibration["settings"]),
        "pesq_strata_recomputed":quality_recomputed,"pesq_maximum_error":quality_error,"minimum_valid_frames":minimum,
        "attenuated_source_clips":sum(g<1 for g in gains),"historical_snapshot_files_verified":historical,
        "scope":f"All checkpoint inference audited; full training replay seed={replay_seed}, replayed {len(replays)} settings; no full replay for other settings"}
