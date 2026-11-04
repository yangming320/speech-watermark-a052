from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
import yaml

from .channels import codec_runtime
from .deep_data import load_entries, prepare_deep_data
from .deep_evaluation import calibrate_validation, grouped_scores, save_evaluation
from .environment import probe_environment
from .evaluation import write_csv
from .keys import load_key
from .models import PARAMETERS, load_checkpoint, make_contract, state_digest
from .training import fit_model, synthetic_sanity
from .utils import canonical_hash, read_jsonl, sha256_file, write_json, write_jsonl


def train_checkpoints(out,config,key):
    (out/"checkpoints").mkdir();(out/"training").mkdir()
    rows=[];models={};hashes={}
    manifest=read_jsonl(out/"feature_manifest.jsonl")
    for condition in config["conditions"]:
        cid=condition["id"];models[cid]={}
        train=load_entries(out,"dev_train",cid);val=load_entries(out,"dev_validation",cid)
        for arch in config["training"]["architectures"]:
            fit_started=time.perf_counter()
            result=fit_model(train,val,arch,config["training"])
            fit_seconds=time.perf_counter()-fit_started
            write_csv(out/"training"/f"{cid}_{arch}_history.csv",result["history"])
            binding={"split_manifest_sha256":sha256_file(out/"split_manifest.jsonl"),
                "train_feature_manifest_hash":canonical_hash([r for r in manifest if r["condition"]==cid and r["kind"]=="main" and r["split"]=="dev_train"]),
                "validation_feature_manifest_hash":canonical_hash([r for r in manifest if r["condition"]==cid and r["kind"]=="main" and r["split"]=="dev_validation"]),
                "train_parent_ids":sorted({r["parent_utterance_id"] for r in train}),
                "validation_parent_ids":sorted({r["parent_utterance_id"] for r in val}),
                "config_hash":canonical_hash(config),"torch_version":str(torch.__version__)}
            contract=make_contract(key,config,condition)
            base={"schema_version":1,"architecture":arch,"parameters":PARAMETERS[arch],"contract":contract,
                  "contract_hash":canonical_hash(contract),"binding":binding,"training_config":config["training"],
                  "scope":"public smoke; CPU development model"}
            final=out/"checkpoints"/f"{cid}_{arch}_epoch25.pt"
            torch.save({**base,"epoch":config["training"]["epochs"],"selection":"preset final epoch",
                "state_dict":result["final_state"],"state_sha256":result["state_sha256"],
                "optimizer_state":result["optimizer_state"],"rng_state":result["rng_state"],"updates":result["updates"]},final)
            best=out/"checkpoints"/f"{cid}_{arch}_validation_best.pt"
            torch.save({**base,"epoch":result["best_epoch"],"selection":"validation BCE minimum; diagnostic only",
                "state_dict":result["best_state"],"state_sha256":state_digest(result["best_state"])},best)
            model,checkpoint=load_checkpoint(final,key);models[cid][arch]=model
            name=f"{cid}_{arch}";hashes[name]=sha256_file(final)
            rows.append({"condition":cid,"architecture":arch,"parameters":PARAMETERS[arch],"epoch":checkpoint["epoch"],
                "seed":config["training"]["seed"],"updates":result["updates"],"maximum_gradient_norm":result["maximum_gradient_norm"],
                "initial_state_sha256":result["initial_state_sha256"],"state_sha256":result["state_sha256"],
                "checkpoint_path":str(final.relative_to(out)),"checkpoint_sha256":sha256_file(final),
                "best_epoch_diagnostic":result["best_epoch"],"best_checkpoint_sha256":sha256_file(best),
                "first_train_bce":result["history"][0]["train_bce"],"last_train_bce":result["history"][-1]["train_bce"],
                "last_validation_bce":result["history"][-1]["validation_bce"]})
            rows[-1]["fit_seconds"]=fit_seconds
            print(f"Trained {name}: epoch {checkpoint['epoch']}, train BCE {rows[-1]['last_train_bce']:.6f}, val BCE {rows[-1]['last_validation_bce']:.6f}",flush=True)
    write_json(out/"training_summary.json",{"models":rows,"primary":"epoch25 for all four; validation-best checkpoints diagnostic only"})
    return models,hashes


def run_deep(root: Path,config_path: Path,run_id=None):
    started=time.perf_counter();config=yaml.safe_load(config_path.read_text())
    if config["experiment_kind"]!="deep_detection_v1" or config["training"]["normalization"]!="none":
        raise ValueError("Unsupported deep experiment protocol")
    if config["training"]["device"]!="cpu" or config["training"]["epochs"]!=25:raise ValueError("M4 protocol fixes CPU and 25 epochs")
    run_id=run_id or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d-%H%M%S")
    if not run_id or Path(run_id).name!=run_id or run_id in {".",".."}:raise ValueError("Invalid run ID")
    out=root/config["output"]["root"]/config["experiment_id"]/run_id;out.mkdir(parents=True,exist_ok=False)
    (out/"config.yaml").write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True))
    key=load_key(root/config["key"]["path"])
    write_json(out/"key.json",json.loads((root/config["key"]["path"]).read_text()))
    write_json(out/"environment.json",probe_environment());write_json(out/"codec_runtime.json",codec_runtime())
    write_jsonl(out/"failures.jsonl",[])
    try:
        write_json(out/"sanity.json",synthetic_sanity(config["training"]["seed"]))
        prepare_deep_data(root,out,config,key)
        write_json(out/"run_status.json",{"status":"training; evaluation scores not yet computed"})
        models,hashes=train_checkpoints(out,config,key)
        validation_rows=grouped_scores(out,config,models,["dev_validation"],["main"])
        calibration=calibrate_validation(validation_rows,hashes);write_json(out/"calibration.json",calibration)
        gate={"status":"frozen_before_dev_evaluation","primary_epoch":25,"config_hash":canonical_hash(config),
              "checkpoint_hashes":hashes,"calibration_sha256":sha256_file(out/"calibration.json"),
              "eval_partition_used_for_training_or_selection":False}
        write_json(out/"evaluation_gate.json",gate)
        metrics,counts=save_evaluation(out,config,models,calibration)
        for name,digest in hashes.items():assert sha256_file(out/"checkpoints"/f"{name}_epoch25.pt")==digest
        summary={"experiment_kind":"deep_detection_v1","experiment_id":config["experiment_id"],"run_id":run_id,
            "scope":"three public source utterances; one parent per development partition; no formal held-out benchmark",
            "config_hash":canonical_hash(config),"split_manifest_sha256":sha256_file(out/"split_manifest.jsonl"),
            "feature_manifest_sha256":sha256_file(out/"feature_manifest.jsonl"),"primary_split":"dev_evaluation",
            "primary_epoch":25,"models":4,"received_wavs":160,"metric_rows":len(metrics),**counts,
            "failures":0,"elapsed_s":time.perf_counter()-started,"metrics":metrics}
        write_json(out/"metrics.json",summary)
        from .deep_validation import validate_deep
        from .deep_reporting import make_deep_report
        write_json(out/"validation.json",validate_deep(out,replay_training=True))
        make_deep_report(out);write_json(out/"run_status.json",{"status":"complete","validation_passed":True})
        return out
    except Exception as exc:
        write_json(out/"run_status.json",{"status":"failed","error":repr(exc)})
        write_jsonl(out/"failures.jsonl",[{"stage":"fatal","error":repr(exc)}]);raise
