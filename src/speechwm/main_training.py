from __future__ import annotations

import time

import numpy as np
import torch

from .deep_data import load_entries
from .evaluation import write_csv
from .models import PARAMETERS, load_checkpoint, make_contract, state_digest
from .training import evaluate_entries, fit_model
from .utils import canonical_hash, read_jsonl, sha256_file, write_json


def train_models(out,config,key):
    rows=[];models={};hashes={};manifest=read_jsonl(out/"feature_manifest.jsonl")
    for condition in config["conditions"]:
        cid=condition["id"];models[cid]={}
        train=load_entries(out,"train",cid);validation=load_entries(out,"validation",cid)
        for seed in config["training"]["seeds"]:
            models[cid][seed]={}
            training={**config["training"],"seed":seed}
            for arch in training["architectures"]:
                name=f"{cid}_{arch}_seed{seed}";start=time.perf_counter()
                print(f"Training {name}: {sum(len(e['x']) for e in train)} frames, 25 epochs",flush=True)
                result=fit_model(train,validation,arch,training,train_split="train",validation_split="validation")
                seconds=time.perf_counter()-start
                write_csv(out/"training"/(name+"_history.csv"),result["history"])
                contract=make_contract(key,config,condition)
                contract["scope"]="Fixed LibriSpeech alternative corpus; no production threshold for arbitrary inputs"
                binding={"config_hash":canonical_hash(config),"split_manifest_sha256":sha256_file(out/"input_manifest.jsonl"),
                    "train_parent_ids":sorted({e["parent_utterance_id"] for e in train}),
                    "validation_parent_ids":sorted({e["parent_utterance_id"] for e in validation}),
                    "train_feature_manifest_hash":canonical_hash([r for r in manifest if r["condition"]==cid and r["split"]=="train" and r["kind"]=="main"]),
                    "validation_feature_manifest_hash":canonical_hash([r for r in manifest if r["condition"]==cid and r["split"]=="validation" and r["kind"]=="main"]),
                    "torch_version":str(torch.__version__)}
                base={"schema_version":1,"architecture":arch,"parameters":PARAMETERS[arch],"contract":contract,
                      "contract_hash":canonical_hash(contract),"binding":binding,"training_config":training}
                path=out/"checkpoints"/(name+"_epoch25.pt")
                torch.save({**base,"epoch":25,"selection":"preset final epoch","state_dict":result["final_state"],
                    "state_sha256":result["state_sha256"],"optimizer_state":result["optimizer_state"],"rng_state":result["rng_state"],"updates":result["updates"]},path)
                best=out/"checkpoints"/(name+"_validation_best.pt")
                torch.save({**base,"epoch":result["best_epoch"],"selection":"validation BCE minimum; diagnostic only",
                    "state_dict":result["best_state"],"state_sha256":state_digest(result["best_state"])},best)
                model,_=load_checkpoint(path,key);models[cid][seed][arch]=model;hashes[name]=sha256_file(path)
                last=result["history"][-1]
                rows.append({"name":name,"condition":cid,"architecture":arch,"seed":seed,"epoch":25,"parameters":PARAMETERS[arch],
                    "updates":result["updates"],"state_sha256":result["state_sha256"],"initial_state_sha256":result["initial_state_sha256"],
                    "maximum_gradient_norm":result["maximum_gradient_norm"],"fit_seconds":seconds,
                    "checkpoint_path":str(path.relative_to(out)),"checkpoint_sha256":hashes[name],"best_epoch_diagnostic":result["best_epoch"],
                    "best_checkpoint_sha256":sha256_file(best),"first_train_bce":result["history"][0]["train_bce"],
                    "last_train_bce":last["train_bce"],"last_validation_bce":last["validation_bce"],"train_frames":last["train_frames"],
                    "positive_train_frames":last["positive_train_frames"],"validation_frames":last["validation_frames"]})
                write_json(out/"training_summary.json",{"models":rows,"primary_epoch":25,"expected_models":12,"complete":len(rows)==12})
                print(f"Trained {name} in {seconds:.2f}s: train BCE {last['train_bce']:.6f}, validation {last['validation_bce']:.6f}",flush=True)
    return models,hashes


def shuffled_label_control(out,config):
    training={**config["training"],"seed":2026};entries=load_entries(out,"train","C1")
    labels=np.random.default_rng(20261018).permutation([e["label"] for e in entries])
    shuffled=[{**e,"label":int(label)} for e,label in zip(entries,labels)]
    validation=load_entries(out,"validation","C1")
    result=fit_model(shuffled,validation,"A",training,train_split="train",validation_split="validation")
    stats=evaluate_entries(result["model"],"A",validation)
    write_csv(out/"training/C1_A_shuffled_label_control_history.csv",result["history"])
    record={"scope":"Diagnostic control only; file labels shuffled in train, true validation labels unchanged; not a primary model",
        "condition":"C1","architecture":"A","seed":2026,"shuffle_seed":20261018,"epoch":25,
        "original_labels_hash":canonical_hash([e["label"] for e in entries]),"shuffled_labels_hash":canonical_hash(labels.tolist()),
        "changed_file_labels":sum(e["label"]!=int(label) for e,label in zip(entries,labels)),"validation":stats,
        "state_sha256":result["state_sha256"],"test_used":False}
    write_json(out/"shuffled_label_control.json",record);return record
