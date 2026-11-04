from __future__ import annotations

import json
from pathlib import Path
import resource
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml

from .channels import codec_runtime
from .comparison import choose_quality_pair
from .environment import probe_environment
from .evaluation import write_csv
from .formal_data import validate_formal
from .keys import load_key
from .main_data import prepare_partition, prepare_validation_noise
from .main_evaluation import score_partition, seed_summary
from .main_training import shuffled_label_control, train_models
from .training import synthetic_sanity
from .utils import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl


def select_validation_quality(rows,config):
    if not rows or any(r["split"]!="validation" for r in rows):raise ValueError("Quality selection accepts validation only")
    if any(r["quality_failures"] for r in rows):raise ValueError("Validation PESQ failure requires investigation")
    selected=choose_quality_pair(rows,config["evaluation"]["quality_target_nb"],config["evaluation"]["quality_tolerance"])
    return {**selected,"scope":"validation only; fixed global strengths applied unchanged to test", "source_split":"validation",
            "validation_quality_rows_hash":canonical_hash(rows),"selected_before_test_processing":True}


def run_main(root: Path,config_path: Path,run_id=None):
    config=yaml.safe_load(config_path.read_text())
    if config["experiment_kind"]!="formal_main_v1" or config["training"]["seeds"]!=[2026,2027,2028] or config["training"]["epochs"]!=25:
        raise ValueError("M5 v1 fixes three seeds and 25 epochs")
    if config["training"]["device"]!="cpu" or config["training"]["normalization"]!="none":raise ValueError("Unsupported training protocol")
    dataset=root/config["data"]["dataset_directory"];data_audit=validate_formal(root,dataset)
    run_id=run_id or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    if not run_id or Path(run_id).name!=run_id or run_id in {".",".."}:raise ValueError("Invalid run ID")
    out=root/config["output"]["root"]/config["experiment_id"]/run_id;out.mkdir(parents=True,exist_ok=False)
    for name in ["audio","embedded","references","features","checkpoints","training"]:(out/name).mkdir()
    (out/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
    rows=read_jsonl(root/config["data"]["manifest"]);write_jsonl(out/"input_manifest.jsonl",rows)
    write_json(out/"dataset_validation.json",data_audit)
    write_json(out/"dataset_freeze.json",json.loads((dataset/"freeze.json").read_text()))
    key=load_key(root/config["key"]["path"]);write_json(out/"key.json",json.loads((root/config["key"]["path"]).read_text()))
    write_json(out/"environment.json",probe_environment());write_json(out/"codec_runtime.json",codec_runtime())
    write_jsonl(out/"failures.jsonl",[]);start=time.perf_counter();phases=[]
    def status(phase):write_json(out/"run_status.json",{"status":phase});print(phase,flush=True)
    try:
        write_json(out/"sanity.json",synthetic_sanity())
        status("preparing train and validation; test integrity checked, no test experiment artifacts")
        phases.append(prepare_partition(root,out,config,key,"train"))
        phases.append(prepare_partition(root,out,config,key,"validation"))
        quality=read_jsonl(out/"quality_scores.jsonl");selection=select_validation_quality(quality,config)
        write_json(out/"quality_selection.json",selection)
        write_json(out/"quality_gate.json",{"source_split":"validation","selection_sha256":sha256_file(out/"quality_selection.json"),
                    "validation_quality_rows_hash":canonical_hash(quality),"test_quality_computed":False})
        prepare_validation_noise(root,out,config,key,selection)
        status("training twelve primary settings; no test scores")
        models,hashes=train_models(out,config,key)
        shuffled_label_control(out,config)
        status("validation calibration")
        validation,validation_counts=score_partition(out,config,models,"validation")
        val_metrics=validation.metrics();calibration=validation.calibration(hashes)
        write_json(out/"calibration.json",calibration)
        rates=validation.rates(calibration);validation.write_curves(out/"validation_threshold_curves.csv");del validation
        gate={"status":"frozen_before_test_processing_and_scoring","config_hash":canonical_hash(config),
            "dataset_manifest_sha256":sha256_file(root/config["data"]["manifest"]),"primary_epoch":25,"checkpoint_hashes":hashes,
            "quality_selection_sha256":sha256_file(out/"quality_selection.json"),"calibration_sha256":sha256_file(out/"calibration.json"),
            "test_used_for_training_selection_or_calibration":False}
        write_json(out/"evaluation_gate.json",gate)
        status("frozen test E02/E03/E04/E05")
        phases.append(prepare_partition(root,out,config,key,"test",selection))
        test,test_counts=score_partition(out,config,models,"test")
        metrics=val_metrics+test.metrics();rates+=test.rates(calibration);test.write_curves(out/"test_threshold_curves.csv");del test
        for name,digest in hashes.items():assert sha256_file(out/"checkpoints"/(name+"_epoch25.pt"))==digest
        quality=read_jsonl(out/"quality_scores.jsonl")
        write_csv(out/"quality_scores.csv",[{**r,"quality_failures":json.dumps(r["quality_failures"])} for r in quality])
        write_csv(out/"metrics.csv",metrics);write_csv(out/"seed_summary.csv",seed_summary(metrics));write_csv(out/"fixed_threshold_rates.csv",rates)
        summary={"experiment_kind":"formal_main_v1","experiment_id":config["experiment_id"],"run_id":run_id,
            "scope":"Fixed speaker-disjoint LibriSpeech alternative corpus, centered 3-second clips; not original TIMIT benchmark",
            "config_hash":canonical_hash(config),"dataset_manifest_sha256":gate["dataset_manifest_sha256"],"models":12,"primary_epoch":25,
            "seeds":config["training"]["seeds"],"data_audit":data_audit,"phase_counts":phases,
            "validation_counts":validation_counts,"test_counts":test_counts,"metrics":metrics,"metric_rows":len(metrics),
            "quality_rows":len(quality),"quality_failures":sum(bool(r["quality_failures"]) for r in quality),
            "quality_selection":selection,"elapsed_s_before_audit":time.perf_counter()-start,
            "peak_process_rss_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)}
        write_json(out/"metrics.json",summary)
        from .main_validation import validate_main
        from .main_reporting import make_main_report
        status("auditing saved test and validation artifacts")
        write_json(out/"validation.json",validate_main(out,replay_seed=2026))
        make_main_report(out);status("complete")
        return out
    except Exception as exc:
        write_json(out/"run_status.json",{"status":"failed","error":repr(exc)})
        write_jsonl(out/"failures.jsonl",[{"stage":"fatal","error":repr(exc)}]);raise
