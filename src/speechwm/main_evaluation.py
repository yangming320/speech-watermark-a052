"""Stream M5 scores; keep only numeric score buckets rather than millions of dictionaries."""
from __future__ import annotations

from collections import defaultdict
import csv

import numpy as np

from .comparison import CSVStream
from .deep_data import load_entries
from .features import aggregate
from .metrics import detection_curve, equal_error_rate
from .models import predict_logits
from .utils import read_jsonl


def scores_for(entry,models):
    yield "correlation",0,entry["correlation"]
    for seed,architectures in sorted(models.get(entry["condition"],{}).items()):
        for arch,model in sorted(architectures.items()):yield f"model_{arch}",seed,predict_logits(model,arch,entry["x"])


def score_key(condition,split,kind,detector,seed,frames):return (condition,split,kind,detector,seed,frames)


class Statistics:
    def __init__(self):self.buckets={}

    def add(self,entry,detector,seed,frames,group):
        key=score_key(entry["condition"],entry["split"],entry["kind"],detector,seed,frames)
        if key not in self.buckets:self.buckets[key]={"labels":[],"scores":[],"first_labels":[],"first_scores":[],"uids":[set(),set()],"speakers":set()}
        b=self.buckets[key];label=entry["label"]
        b["labels"].append(label);b["scores"].append(group["score"]);b["uids"][label].add(entry["utterance_id"]);b["speakers"].add(entry["speaker_id"])
        if group["group_index"]==0:b["first_labels"].append(label);b["first_scores"].append(group["score"])

    def metrics(self):
        rows=[]
        for key,b in sorted(self.buckets.items()):
            labels=np.asarray(b["labels"]);scores=np.asarray(b["scores"])
            r=dict(zip(["condition","split","kind","detector","seed","frames"],key))
            e=equal_error_rate(labels,scores);e.pop("threshold_role")
            rows.append({**r,**e,"eer_first_group_per_utterance":equal_error_rate(np.asarray(b["first_labels"]),np.asarray(b["first_scores"]))["eer"],
                "negative_groups":int((labels==0).sum()),"positive_groups":int((labels==1).sum()),
                "paired_eligible_utterances":len(b["uids"][0]&b["uids"][1]),"speakers":len(b["speakers"])})
        return rows

    def calibration(self,checkpoint_hashes):
        if any(k[1]!="validation" for k in self.buckets):raise ValueError("Calibration accepts validation only")
        return {"source_split":"validation","checkpoint_hashes":checkpoint_hashes,
            "scope":"speaker-held-out validation calibration on fixed alternative corpus; arbitrary uploads still scores_only",
            "rule":"validation interpolated EER threshold; score >= threshold is positive",
            "settings":[{k:r[k] for k in ["condition","detector","seed","frames"]}|{"threshold":r["eer_threshold_interpolated"],"validation_eer":r["eer"]}
                        for r in self.metrics() if r["kind"]=="main"]}

    def rates(self,calibration):
        thresholds={(r["condition"],r["detector"],r["seed"],r["frames"]):r["threshold"] for r in calibration["settings"]};rows=[]
        for key,b in sorted(self.buckets.items()):
            condition,split,kind,detector,seed,frames=key
            if kind!="main":continue
            threshold=thresholds[(condition,detector,seed,frames)]
            labels=np.asarray(b["labels"]);scores=np.asarray(b["scores"])
            rows.append({"condition":condition,"split":split,"detector":detector,"seed":seed,"frames":frames,"threshold":threshold,
                "threshold_source":"validation","far":float(np.mean(scores[labels==0]>=threshold)),"frr":float(np.mean(scores[labels==1]<threshold)),
                "negative_groups":int((labels==0).sum()),"positive_groups":int((labels==1).sum())})
        return rows

    def write_curves(self,path):
        stream=CSVStream(path)
        for key,b in sorted(self.buckets.items()):
            base=dict(zip(["condition","split","kind","detector","seed","frames"],key))
            curve=detection_curve(np.asarray(b["labels"]),np.asarray(b["scores"]))
            for t,far,frr in zip(curve["threshold"],curve["far"],curve["frr"]):stream.add({**base,"threshold":t,"far":far,"frr":frr})
        stream.close();return stream.count


def score_partition(out,config,models,split):
    if split=="test" and not (out/"evaluation_gate.json").exists():raise ValueError("Test scoring requires frozen gate")
    streams=[CSVStream(out/f"{split}_{name}.csv") for name in ["frame_scores","aggregate_scores","group_definitions"]]
    statistics=Statistics();conditions=sorted({r["condition"] for r in read_jsonl(out/"feature_manifest.jsonl") if r["split"]==split})
    for cid in conditions:
        for kind in ["main","zero_control"]:
            for entry in load_entries(out,split,cid,kind):
                frame_counts=config["evaluation"]["lpc_aggregate_frames"] if cid.startswith("E03_") else config["evaluation"]["aggregate_frames"]
                scores=list(scores_for(entry,models));base={"id":entry["id"]}
                for detector,seed,values in scores:
                    for index,time,value in zip(entry["frame_indices"],entry["times"],values):
                        streams[0].add({**base,"detector":detector,"seed":seed,"frame_index":int(index),"time_s":float(time),"score":float(value)})
                for frames in frame_counts:
                    for group in aggregate(np.zeros(len(entry["x"])),entry["times"],frames):streams[2].add({**base,**{k:v for k,v in group.items() if k!="score"}})
                    for detector,seed,values in scores:
                        for group in aggregate(values,entry["times"],frames):
                            streams[1].add({**base,"detector":detector,"seed":seed,**group})
                            statistics.add(entry,detector,seed,frames,group)
        print(f"Scored {split}/{cid}",flush=True)
    for stream in streams:stream.close()
    counts=dict(zip(["frame_score_rows","aggregate_score_rows","group_definition_rows"],[s.count for s in streams]))
    return statistics,counts


def read_statistics(out,split):
    metadata={r["id"]:r for r in read_jsonl(out/"feature_manifest.jsonl")}
    stats=Statistics()
    with (out/f"{split}_aggregate_scores.csv").open() as f:
        for r in csv.DictReader(f):stats.add(metadata[r["id"]],r["detector"],int(r["seed"]),int(r["frames"]),
            {"score":float(r["score"]),"group_index":int(r["group_index"])})
    return stats


def seed_summary(metrics):
    buckets=defaultdict(list)
    for r in metrics:
        if r["split"]=="test" and r["kind"]=="main":buckets[(r["condition"],r["detector"],r["frames"])].append(r)
    rows=[]
    for (cid,detector,frames),values in sorted(buckets.items()):
        x=np.asarray([r["eer"] for r in values]);first=np.asarray([r["eer_first_group_per_utterance"] for r in values])
        rows.append({"condition":cid,"detector":detector,"frames":frames,"seeds":len(values),"eer_mean":float(x.mean()),
            "eer_std_sample":float(x.std(ddof=1)) if len(x)>1 else 0.,"eer_min":float(x.min()),"eer_max":float(x.max()),
            "first_group_eer_mean":float(first.mean()),"first_group_eer_std_sample":float(first.std(ddof=1)) if len(first)>1 else 0.,
            "negative_groups":values[0]["negative_groups"],"positive_groups":values[0]["positive_groups"],
            "paired_eligible_utterances":values[0]["paired_eligible_utterances"],"speakers":values[0]["speakers"]})
    return rows
