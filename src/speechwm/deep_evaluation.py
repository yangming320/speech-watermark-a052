from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

from .comparison import CSVStream
from .deep_data import SPLITS, load_entries
from .evaluation import write_csv
from .features import aggregate
from .metrics import detection_curve, equal_error_rate
from .models import predict_logits
from .utils import write_json


def entry_scores(entry,models):
    return {"correlation":entry["correlation"],
            **{f"model_{arch}":predict_logits(model,arch,entry["x"]) for arch,model in models.items()}}


def grouped_scores(out,config,models,splits,kinds,streams=None):
    rows=[]
    for condition in config["conditions"]:
        cid=condition["id"]
        for split in splits:
            for kind in kinds:
                for entry in load_entries(out,split,cid,kind):
                    scores=entry_scores(entry,models[cid])
                    base={k:entry[k] for k in ["id","utterance_id","parent_utterance_id","speaker_id","split","condition","kind","label"]}
                    if streams:
                        for detector,values in scores.items():
                            for index,time,score in zip(entry["frame_indices"],entry["times"],values):
                                streams[0].add({**base,"detector":detector,"frame_index":int(index),"time_s":float(time),"score":float(score)})
                    for frames in config["evaluation"]["aggregate_frames"]:
                        if streams:
                            for group in aggregate(np.zeros(len(entry["x"])),entry["times"],frames):
                                streams[2].add({**base,**{k:v for k,v in group.items() if k!="score"}})
                        for detector,values in scores.items():
                            for group in aggregate(values,entry["times"],frames):
                                record={**base,"detector":detector,**group};rows.append(record)
                                if streams:streams[1].add(record)
    return rows


def metric_table(rows):
    buckets=defaultdict(list)
    for r in rows:buckets[(r["condition"],r["split"],r["kind"],r["detector"],r["frames"])].append(r)
    metrics=[]
    for (condition,split,kind,detector,frames),values in sorted(buckets.items()):
        labels=np.array([r["label"] for r in values]);scores=np.array([r["score"] for r in values])
        stats=equal_error_rate(labels,scores);stats.pop("threshold_role")
        first=[r for r in values if r["group_index"]==0]
        first_eer=equal_error_rate(np.array([r["label"] for r in first]),np.array([r["score"] for r in first]))["eer"]
        paired={r["utterance_id"] for r in values if r["label"]==0}&{r["utterance_id"] for r in values if r["label"]==1}
        metrics.append({"condition":condition,"split":split,"kind":kind,"detector":detector,"frames":frames,
            **stats,"eer_first_group_per_segment":first_eer,"paired_eligible_segments":len(paired),
            "negative_groups":int((labels==0).sum()),"positive_groups":int((labels==1).sum())})
    return metrics


def calibrate_validation(rows,checkpoint_hashes):
    if not rows or any(r["split"]!="dev_validation" or r["kind"]!="main" for r in rows):
        raise ValueError("Calibration must use validation main data only")
    return {"scope":"single-parent development calibration; not for arbitrary uploads",
        "source_split":"dev_validation","checkpoint_hashes":checkpoint_hashes,
        "threshold_rule":"interpolated validation EER threshold, then measure actual discrete FAR/FRR",
        "settings":[{"condition":r["condition"],"detector":r["detector"],"frames":r["frames"],
                     "threshold":r["eer_threshold_interpolated"],"validation_eer":r["eer"]} for r in metric_table(rows)]}


def fixed_threshold_rates(rows,calibration):
    thresholds={(r["condition"],r["detector"],r["frames"]):r["threshold"] for r in calibration["settings"]}
    buckets=defaultdict(list)
    for r in rows:
        if r["kind"]=="main":buckets[(r["condition"],r["split"],r["detector"],r["frames"])].append(r)
    result=[]
    for (condition,split,detector,frames),values in sorted(buckets.items()):
        threshold=thresholds[(condition,detector,frames)]
        neg=[r["score"] for r in values if r["label"]==0];pos=[r["score"] for r in values if r["label"]==1]
        result.append({"condition":condition,"split":split,"detector":detector,"frames":frames,"threshold":threshold,
            "threshold_source":"dev_validation","far":float(np.mean(np.asarray(neg)>=threshold)),
            "frr":float(np.mean(np.asarray(pos)<threshold)),"negative_groups":len(neg),"positive_groups":len(pos)})
    return result


def save_evaluation(out,config,models,calibration):
    streams=[CSVStream(out/name) for name in ["frame_scores.csv","aggregate_scores.csv","group_definitions.csv"]]
    rows=grouped_scores(out,config,models,SPLITS,["main","zero_control"],streams)
    for stream in streams:stream.close()
    metrics=metric_table(rows);write_csv(out/"metrics.csv",metrics)
    write_csv(out/"fixed_threshold_rates.csv",fixed_threshold_rates(rows,calibration))
    curves=CSVStream(out/"threshold_curves.csv")
    buckets=defaultdict(list)
    for r in rows:buckets[(r["condition"],r["split"],r["kind"],r["detector"],r["frames"])].append(r)
    for setting,values in sorted(buckets.items()):
        base=dict(zip(["condition","split","kind","detector","frames"],setting))
        curve=detection_curve(np.array([r["label"] for r in values]),np.array([r["score"] for r in values]))
        for threshold,far,frr in zip(curve["threshold"],curve["far"],curve["frr"]):
            curves.add({**base,"threshold":threshold,"far":far,"frr":frr})
    curves.close()
    # This baseline-only all-smoke regression is separate from the held-partition model table.
    original=json.loads((next(p for p in out.resolve().parents if (p/"pyproject.toml").exists())/"milestones/M2/evidence/metrics.json").read_text())["metrics"]
    regression=[]
    for c in config["conditions"]:
        for frames in config["evaluation"]["aggregate_frames"]:
            v=[r for r in rows if r["condition"]==c["id"] and r["kind"]=="main" and r["detector"]=="correlation" and r["frames"]==frames]
            current=equal_error_rate(np.array([r["label"] for r in v]),np.array([r["score"] for r in v]))
            previous=next(r for r in original if r["delta_db"]==c["delta_db"] and r["channel"]==c["channel"] and r["frames"]==frames and r["detector"]=="correlation")
            assert current["eer"]==previous["eer"] and current["samples"]==previous["samples"]
            regression.append({"condition":c["id"],"frames":frames,"eer":current["eer"],"samples":current["samples"],"identical_to_m2":True})
    write_json(out/"m2_baseline_regression.json",{"passed":True,"settings":regression,
        "scope":"all-smoke correlation-only regression; do not compare this population with dev_evaluation model results"})
    return metrics,{"frame_score_rows":streams[0].count,"aggregate_score_rows":streams[1].count,"group_definition_rows":streams[2].count}
