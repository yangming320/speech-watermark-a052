import numpy as np
import pytest

from speechwm.formal_data import check_disjoint, select_catalog
from speechwm.main_data import control_ids, prepare_partition
from speechwm.main_evaluation import Statistics, seed_summary, score_partition
from speechwm.main_runner import select_validation_quality
from speechwm.training import fit_model


def selection_fixture():
    config={"seed":20261018,"exclude_speakers":["old"],"selection":{"clip_seconds":3,
        **{s:{"speakers_per_gender":1,"utterances_per_speaker":2} for s in ["train","validation","test"]}}}
    catalog=[]
    for pool in ["training_pool","test_pool"]:
        for g in ["F","M"]:
            for i in range(4):
                speaker=f"{pool}_{g}_{i}"
                for j in range(3):catalog.append({"pool":pool,"speaker_id":speaker,"gender":g,
                    "duration_s":4,"parent_utterance_id":f"{speaker}-{j}"})
    return config,catalog


def test_formal_selection_is_balanced_deterministic_and_speaker_disjoint():
    c,rows=selection_fixture();a=select_catalog(rows,c);b=select_catalog(list(reversed(rows)),c)
    assert a==b and len(a)==12
    for split in ["train","validation","test"]:
        assert sum(r["split"]==split and r["gender"]=="F" for r in a)==2
    speakers={s:{r["speaker_id"] for r in a if r["split"]==s} for s in ["train","validation","test"]}
    assert not speakers["train"]&speakers["validation"] and not speakers["test"]&(speakers["train"]|speakers["validation"])


def test_formal_selection_rejects_insufficient_speakers_and_excludes_history():
    c,rows=selection_fixture();c["exclude_speakers"]=[r["speaker_id"] for r in rows]
    with pytest.raises(ValueError,match="Not enough"):select_catalog(rows,c)


def test_formal_selection_honors_separate_validation_source_pool():
    c,rows=selection_fixture()
    validation=[{**r,"pool":"validation_pool","speaker_id":"v_"+r["speaker_id"],
                 "parent_utterance_id":"v_"+r["parent_utterance_id"]} for r in rows if r["pool"]=="training_pool"]
    c["selection"]["pool_by_split"]={"train":"training_pool","validation":"validation_pool","test":"test_pool"}
    selected=select_catalog(rows+validation,c)
    assert all(r["pool"]=="validation_pool" for r in selected if r["split"]=="validation")


@pytest.mark.parametrize("field",["speaker_id","parent_utterance_id","source_sha256","source_pcm_sha256","pcm_sha256"])
def test_formal_duplicate_audit_rejects_every_leakage_identity(field):
    rows=[{"utterance_id":str(i),"speaker_id":str(i),"parent_utterance_id":str(i),"source_sha256":str(i),
           "source_pcm_sha256":str(i),"pcm_sha256":str(i),"split":s,"duration_s":3,"gender":"F"}
          for i,s in enumerate(["train","validation","test"])]
    check_disjoint(rows);rows[2][field]=rows[0][field]
    with pytest.raises(ValueError,match="leakage"):check_disjoint(rows)


def test_formal_audit_excludes_historical_speakers():
    rows=[{"utterance_id":str(i),"speaker_id":str(i),"parent_utterance_id":str(i),"source_sha256":str(i),
           "source_pcm_sha256":str(i),"pcm_sha256":str(i),"split":s,"duration_s":3,"gender":"M"}
          for i,s in enumerate(["train","validation","test"])]
    with pytest.raises(ValueError,match="contamination"):check_disjoint(rows,["0"])


def statistics(split,offset=0):
    stat=Statistics()
    for i,(label,score) in enumerate([(0,0.),(0,1.),(1,2.),(1,3.)]):
        entry={"condition":"C1","split":split,"kind":"main","label":label,"utterance_id":str(i%2),"speaker_id":str(i%2)}
        stat.add(entry,"model_A",2026,1,{"score":score+offset,"group_index":0})
    return stat


def test_main_threshold_is_validation_only_and_fixed_for_shifted_test():
    val=statistics("validation");cal=val.calibration({});test=statistics("test",10)
    with pytest.raises(ValueError):test.calibration({})
    rate=test.rates(cal)[0]
    assert rate["threshold"]==cal["settings"][0]["threshold"] and rate["far"]==1 and rate["frr"]==0


def test_main_seed_summary_uses_sample_standard_deviation():
    base=statistics("test").metrics()[0]
    rows=[{**base,"seed":2026+i,"eer":value} for i,value in enumerate([.1,.2,.3])]
    summary=seed_summary(rows)[0]
    assert summary["eer_mean"]==pytest.approx(.2) and summary["eer_std_sample"]==pytest.approx(.1)
    assert summary["seeds"]==3


@pytest.mark.parametrize("operation",["prepare","score"])
def test_test_partition_requires_frozen_gate(tmp_path,operation):
    with pytest.raises(ValueError,match="gate"):
        if operation=="prepare":prepare_partition(tmp_path,tmp_path,{},None,"test")
        else:score_partition(tmp_path,{}, {},"test")


def test_quality_matching_rejects_test_rows():
    with pytest.raises(ValueError,match="validation only"):
        select_validation_quality([{"split":"test"}],{})


def test_zero_controls_cover_distinct_speakers_and_gender_equally():
    rows=[{"utterance_id":f"{g}{i}_{j}","speaker_id":f"{g}{i}","gender":g} for g in ["F","M"] for i in range(3) for j in range(2)]
    selected=control_ids(rows,4)
    assert len(selected)==4 and sum(uid.startswith("F") for uid in selected)==2
    assert len({r["speaker_id"] for r in rows if r["utterance_id"] in selected})==4


def test_formal_trainer_accepts_only_train_validation_roles():
    from test_deep import configuration, data
    train=data("train","train");validation=data("validation","val")
    result=fit_model(train,validation,"A",configuration(),train_split="train",validation_split="validation")
    assert result["updates"]==6
    validation[0]["split"]="test"
    with pytest.raises(ValueError):fit_model(train,validation,"A",configuration(),train_split="train",validation_split="validation")
    with pytest.raises(ValueError):fit_model(train,validation,"A",configuration(),train_split="train",validation_split="test")
