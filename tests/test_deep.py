import json
import numpy as np
import pytest
import torch

from speechwm.deep_data import audit_splits
from speechwm.deep_evaluation import calibrate_validation, fixed_threshold_rates
from speechwm.keys import make_key
from speechwm.models import ModelA, ModelB, build_model, load_checkpoint, make_contract, predict_logits, state_digest
from speechwm.training import deterministic_cpu, fit_model, synthetic_sanity
from speechwm.utils import canonical_hash


def configuration():
    return {"seed":2026,"cpu_threads":1,"epochs":3,"batch_size":64,
            "optimizer":{"lr":.001,"alpha":.99,"eps":1e-8,"weight_decay":0,"momentum":0,"centered":False,"foreach":False}}


def data(split,parent):
    rng=np.random.default_rng(5);entries=[]
    for label in [0,1]:
        x=(rng.normal(0,.2,(64,63))+2*label-1).astype(np.float32)
        entries.append({"x":x,"times":np.arange(64)*.01,"label":label,"split":split,
                        "parent_utterance_id":parent,"speaker_id":parent,"kind":"main"})
    return entries


@pytest.mark.parametrize("architecture,count",[("A",6209),("B",15713)])
def test_exact_network_shapes_and_parameter_counts(architecture,count):
    model=build_model(architecture)
    assert sum(p.numel() for p in model.parameters())==count
    x=torch.randn(5,63) if architecture=="A" else torch.randn(5,1,63)
    assert model(x).shape==(5,1)
    if architecture=="B":
        assert model.conv(x).shape==(5,16,61)
        assert sum(isinstance(m,torch.nn.ReLU) for m in model.modules())==1
        with pytest.raises(ValueError):model(torch.randn(5,63))
    else:
        with pytest.raises(ValueError):model(torch.randn(5,64))


def test_synthetic_fit_sanity_for_both_networks():
    result=synthetic_sanity()
    assert all(r["training_accuracy"]==1 and r["final_bce"]<.01 for r in result.values())


@pytest.mark.parametrize("architecture",["A","B"])
def test_training_repeat_is_identical_and_uses_all_frames(architecture):
    train=data("dev_train","train");val=data("dev_validation","val")
    first=fit_model(train,val,architecture,configuration());second=fit_model(train,val,architecture,configuration())
    assert first["state_sha256"]==second["state_sha256"] and first["history"]==second["history"]
    assert first["updates"]==6 and first["maximum_gradient_norm"]>0
    assert first["history"][-1]["train_bce"]<first["history"][0]["train_bce"]


@pytest.mark.parametrize("problem",["eval","parent","speaker","null"])
def test_training_rejects_leakage_and_control_data(problem):
    train=data("dev_train","train");val=data("dev_validation","val")
    if problem=="eval":val[0]["split"]="dev_evaluation"
    elif problem=="parent":val[0]["parent_utterance_id"]="train"
    elif problem=="speaker":val[0]["speaker_id"]="train"
    else:train[0]["kind"]="zero_control"
    with pytest.raises(ValueError):fit_model(train,val,"A",configuration())


@pytest.mark.parametrize("field",["parent_utterance_id","speaker_id","source_sha256","waveform_sha256"])
def test_split_audit_rejects_source_or_hash_leakage(field):
    rows=[{"utterance_id":str(i),"parent_utterance_id":str(i),"speaker_id":str(i),"source_sha256":str(i),
           "waveform_sha256":str(i),"split":s,"duration_s":1} for i,s in enumerate(["dev_train","dev_validation","dev_evaluation"])]
    audit_splits(rows);rows[1][field]=rows[0][field]
    with pytest.raises(ValueError,match="leakage"):audit_splits(rows)


def save_checkpoint(path,architecture="A",vad="webrtc"):
    from speechwm.spectral import SpectralConfig
    from dataclasses import asdict
    key=make_key();config={"spectral":asdict(SpectralConfig()),"evaluation":{"vad":{"mode":vad,"aggressiveness":2},"aggregate_frames":[1,30]}}
    contract=make_contract(key,config,{"id":"C1","delta_db":1,"channel":"downsample_8k"})
    model=build_model(architecture);state=model.state_dict()
    checkpoint={"schema_version":1,"architecture":architecture,"contract":contract,"contract_hash":canonical_hash(contract),
        "state_dict":state,"state_sha256":state_digest(state),"training_config":{"cpu_threads":1},"epoch":25}
    torch.save(checkpoint,path);return checkpoint,key


def test_safe_checkpoint_reload_and_contract_key_checks(tmp_path):
    path=tmp_path/"model.pt";checkpoint,key=save_checkpoint(path,"B")
    model,loaded=load_checkpoint(path,key)
    x=np.random.default_rng(2).normal(size=(50,63)).astype(np.float32)
    np.testing.assert_array_equal(predict_logits(model,"B",x),predict_logits(model,"B",x))
    with pytest.raises(ValueError,match="key mismatch"):load_checkpoint(path,make_key(99))
    checkpoint["contract"]["normalization"]="train_standardized";checkpoint["contract_hash"]=canonical_hash(checkpoint["contract"])
    torch.save(checkpoint,path)
    with pytest.raises(ValueError,match="contract"):load_checkpoint(path,key)


def test_missing_or_corrupt_checkpoint_has_no_detector_fallback(tmp_path):
    with pytest.raises(FileNotFoundError):load_checkpoint(tmp_path/"missing.pt")
    path=tmp_path/"model.pt";checkpoint,key=save_checkpoint(path)
    checkpoint["state_dict"][next(iter(checkpoint["state_dict"]))][0,0]+=1
    torch.save(checkpoint,path)
    with pytest.raises(ValueError,match="tensor hash"):load_checkpoint(path,key)


def test_calibration_uses_validation_only_and_thresholds_remain_fixed():
    rows=[{"condition":"C1","split":"dev_validation","kind":"main","detector":"correlation","frames":1,
           "label":label,"score":score,"group_index":0,"utterance_id":str(i)} for i,(label,score) in enumerate([(0,0.),(0,1.),(1,2.),(1,3.)])]
    calibration=calibrate_validation(rows,{})
    threshold=calibration["settings"][0]["threshold"]
    evaluated=[{**r,"split":"dev_evaluation","score":r["score"]+10} for r in rows]
    rates=fixed_threshold_rates(evaluated,calibration)
    assert rates[0]["threshold"]==threshold and rates[0]["far"]==1 and rates[0]["frr"]==0
    with pytest.raises(ValueError):calibrate_validation(evaluated,{})


def test_zero_control_has_identical_neural_scores():
    x=np.random.default_rng(1).normal(size=(63,63)).astype(np.float32)
    for architecture in ["A","B"]:
        model=build_model(architecture)
        np.testing.assert_array_equal(predict_logits(model,architecture,x),predict_logits(model,architecture,x.copy()))


@pytest.mark.parametrize("architecture",["A","B"])
def test_deep_cli_is_reference_free_and_matches_checkpoint(tmp_path,monkeypatch,capsys,architecture):
    from speechwm.audio import read_audio, write_audio
    from speechwm.cli import main
    from speechwm.features import aggregate
    from speechwm.keys import save_key
    from speechwm.models import infer_audio
    checkpoint=tmp_path/"model.pt";_,key=save_checkpoint(checkpoint,architecture,vad="none")
    save_key(tmp_path/"key.json")
    source=tmp_path/"removed_original.wav";received=tmp_path/"received.wav"
    signal=np.random.default_rng(3).normal(0,.05,8000)
    write_audio(source,signal,8000);write_audio(received,signal,8000);source.unlink()
    monkeypatch.setattr("sys.argv",["speechwm","--root",str(tmp_path),"detect","received.wav","--checkpoint","model.pt","--key","key.json","--frames","30"])
    main();record=json.loads(capsys.readouterr().out)
    audio,sr=read_audio(received);scores,feature,_=infer_audio(checkpoint,audio,sr,key)
    assert record["architecture"]==architecture and record["decision"] is None and record["status"]=="scores_only"
    assert record["groups"]==aggregate(scores,feature.times_s[feature.valid_mask],30)
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError):main()


def test_deep_inference_rejects_wrong_rate(tmp_path):
    from speechwm.models import infer_audio
    path=tmp_path/"model.pt";_,key=save_checkpoint(path)
    with pytest.raises(ValueError,match="8 kHz"):
        infer_audio(path,np.zeros(16000),16000,key)
