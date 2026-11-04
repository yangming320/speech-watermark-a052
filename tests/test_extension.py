import json
import numpy as np
import pytest
import torch
from speechwm.channels import opus_roundtrip
from speechwm.main_evaluation import Statistics
from speechwm.m6_experiment import calibrate,fixed_rates
from speechwm.m6_data import sample_indices,training_sets,prepare
from speechwm.models import load_checkpoint
from speechwm.utils import canonical_hash

@pytest.mark.parametrize('rate',[6000,8000,12000,16000])
def test_actual_four_cbr_rates_keep_samples_and_decode_deterministically(tmp_path,rate):
    t=np.arange(8007)/8000;x=.1*np.sin(2*np.pi*173*t)+np.random.default_rng(5).normal(0,.01,len(t))
    a,sr,metadata=opus_roundtrip(x,8000,tmp_path/'a.wav',bitrate_bps=rate)
    b,_,_=opus_roundtrip(x,8000,tmp_path/'b.wav',bitrate_bps=rate)
    assert sr==8000 and len(a)==len(x)
    np.testing.assert_array_equal(a,b)
    i=metadata['inspection'];assert i['packet_sizes_bytes']==[rate//400] and i['payload_bitrate_bps']==rate

@pytest.mark.parametrize('rate',[0,7000,24000])
def test_invalid_controlled_rates_fail_before_writing(tmp_path,rate):
    with pytest.raises(ValueError):opus_roundtrip(np.ones(8000),8000,tmp_path/'a.wav',bitrate_bps=rate)
    assert not list(tmp_path.iterdir())

def stats(split):
    s=Statistics()
    for cid,shift in [('R6',3),('R8',0),('R12',-1),('R16',2)]:
        for detector,seed in [('correlation',0),('single_A',2026),('mixed_A',2026)]:
            for i,(label,score) in enumerate([(0,0.),(0,1.),(1,2.),(1,3.)]):
                e={'condition':cid,'split':split,'kind':'main','label':label,'utterance_id':str(i%2),'speaker_id':str(i%2)}
                s.add(e,detector,seed,30,{'score':score+shift,'group_index':0})
    return s

def old_calibration():
    return {'settings':[{'condition':'C3','detector':d,'seed':seed,'frames':30,'threshold':.7} for d,seed in [('correlation',0),('model_A',2026)]]}

def test_cross_rate_calibration_inherits_8k_and_pools_mixed_only_on_validation():
    calibration=calibrate(stats('validation'),old_calibration(),{})
    assert {r['threshold'] for r in calibration['settings'] if r['detector']=='single_A'}=={.7}
    assert len({r['threshold'] for r in calibration['settings'] if r['detector']=='mixed_A'})==1
    single=[r for r in fixed_rates(stats('test'),calibration) if r['detector']=='single_A' and r['condition']=='R6'][0]
    assert single['threshold']==.7 and single['far']==1
    with pytest.raises(ValueError):calibrate(stats('test'),old_calibration(),{})

def entry(cid,uid,label,n):
    return {'id':f'{cid}-{uid}-{label}','condition':cid,'utterance_id':str(uid),'label':label,'x':np.arange(n*63).reshape(n,63).astype(np.float32),'times':np.arange(n)*.01,'correlation':np.arange(n).astype(float),'frame_indices':np.arange(n),'split':'train','speaker_id':str(uid),'parent_utterance_id':str(uid),'kind':'main'}

def test_training_budgets_preserve_per_label_total_equal_rate_and_combo_file_quota(tmp_path,monkeypatch):
    cs=['R6','R8','R12','R16','E06_D_SS','E06_D_SS_LPC','E06_O_SS','E06_O_SS_LPC']
    data={c:[entry(c,uid,label,43+i+uid+label) for uid in range(2) for label in [0,1]] for i,c in enumerate(cs)}
    monkeypatch.setattr('speechwm.deep_data.load_entries',lambda out,split,c:data[c])
    selected=training_sets(tmp_path);again=training_sets(tmp_path,persist=False)
    budget=json.loads((tmp_path/'training_budget.json').read_text())
    assert budget['MIXED']['frames']==sum(len(e['x']) for e in data['R8'])
    for label in [0,1]:
        assert sum(len(e['x']) for e in selected['MIXED'] if e['label']==label)==sum(len(e['x']) for e in data['R8'] if e['label']==label)
        counts=[budget['MIXED']['rate_label_frames'][c][str(label)] for c in cs[:4]];assert max(counts)-min(counts)<=1
    for uid in range(2):
        for label in [0,1]:assert len({len(next(e for e in selected[c] if e['utterance_id']==str(uid) and e['label']==label)['x']) for c in cs[4:]})==1
    for c in selected:
        for a,b in zip(selected[c],again[c]):np.testing.assert_array_equal(a['x'],b['x'])
    rows=[json.loads(l) for l in (tmp_path/'training_selection.jsonl').read_text().splitlines()]
    assert all(len(set(r['indices']))==len(r['indices']) for r in rows)

def test_lpc_checkpoint_is_explicitly_supported_but_other_embedding_rejected(tmp_path):
    from test_deep import save_checkpoint
    p=tmp_path/'model.pt';cp,key=save_checkpoint(p);cp['contract']['embedding']='SS_LPC';cp['contract_hash']=canonical_hash(cp['contract']);torch.save(cp,p)
    load_checkpoint(p,key)
    cp['contract']['embedding']='unknown';cp['contract_hash']=canonical_hash(cp['contract']);torch.save(cp,p)
    with pytest.raises(ValueError):load_checkpoint(p,key)

def test_extension_test_generation_requires_gate(tmp_path):
    with pytest.raises(ValueError,match='gate'):prepare(tmp_path,tmp_path,{},None,'test')

@pytest.mark.parametrize('section,field,value',[
    ('training','seeds',[2026]),('training','epochs',1),('extension','sampling_seed',123),
    ('extension','bitrates_kbps',[8]),('extension','offset_ms',[0]),('evaluation','aggregate_frames',[30])])
def test_fixed_extension_protocol_fails_before_touching_corpus_or_outputs(tmp_path,section,field,value):
    from pathlib import Path
    import yaml
    from speechwm.m6_experiment import run_extension
    config=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/extension.yaml').read_text());config[section][field]=value
    path=tmp_path/'bad.yaml';path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError,match='fixed protocol'):run_extension(tmp_path,path,'invalid')
    assert not (tmp_path/'outputs').exists()
