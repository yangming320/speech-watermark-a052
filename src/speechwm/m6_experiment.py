"""M6 paired cross-rate, equal-budget mixed-rate, LPC-combination and offset studies."""
from __future__ import annotations
import csv,json,shutil,time
from pathlib import Path
import numpy as np
import torch
import yaml
from .channels import codec_runtime
from .comparison import CSVStream
from .deep_data import load_entries
from .environment import probe_environment
from .evaluation import write_csv
from .features import aggregate
from .formal_data import validate_formal
from .keys import load_key
from .m6_data import conditions,prepare,prepare_offsets,training_sets
from .main_evaluation import Statistics,seed_summary
from .models import PARAMETERS,load_checkpoint,make_contract,predict_logits
from .training import fit_model
from .utils import canonical_hash,read_jsonl,sha256_file,write_json,write_jsonl


def train(root,out,config,key):
    m5=root/config['extension']['m5_run'];models={};records=[]
    for old in json.loads((m5/'training_summary.json').read_text())['models']:
        if old['condition']!='C3':continue
        name=f"SINGLE_{old['architecture']}_seed{old['seed']}";path=out/'checkpoints'/(name+'_epoch25.pt')
        shutil.copy2(m5/old['checkpoint_path'],path);model,cp=load_checkpoint(path,key)
        models.setdefault('SINGLE',{}).setdefault(old['seed'],{})[old['architecture']]=model
        records.append({**old,'name':name,'family':'SINGLE','checkpoint_path':str(path.relative_to(out)),
            'checkpoint_sha256':sha256_file(path),'copied_from_m5':True})
    sets=training_sets(out)
    for family,entries in sets.items():
        cid='R8' if family=='MIXED' else family;val=load_entries(out,'validation',cid)
        c=next(c for c in conditions(config) if c['id']==cid)
        for seed in config['training']['seeds']:
            for arch in ['A','B']:
                name=f'{family}_{arch}_seed{seed}';start=time.perf_counter();print(f'M6 training {name}, {sum(len(e["x"]) for e in entries)} frames',flush=True)
                tc={**config['training'],'seed':seed};result=fit_model(entries,val,arch,tc,train_split='train',validation_split='validation')
                contract=make_contract(key,config,c);contract.update(embedding=c['method'] if family!='MIXED' else 'SS',
                    condition=family,channel='mixed Opus 6/8/12/16 kbps' if family=='MIXED' else c['channel'],
                    scope='M6 exploratory fixed LibriSpeech subset; scores only on arbitrary uploads')
                checkpoint={'schema_version':1,'architecture':arch,'parameters':PARAMETERS[arch],'contract':contract,'contract_hash':canonical_hash(contract),
                    'binding':{'config_hash':canonical_hash(config),'input_manifest_sha256':sha256_file(out/'input_manifest.jsonl'),
                        'training_selection_sha256':sha256_file(out/'training_selection.jsonl'),'training_family':family,
                        'train_parent_ids':sorted({e['parent_utterance_id'] for e in entries}),
                        'validation_parent_ids':sorted({e['parent_utterance_id'] for e in val})},
                    'training_config':tc,'epoch':25,'selection':'preset final epoch','state_dict':result['final_state'],
                    'state_sha256':result['state_sha256'],'optimizer_state':result['optimizer_state'],'rng_state':result['rng_state'],'updates':result['updates']}
                path=out/'checkpoints'/(name+'_epoch25.pt');torch.save(checkpoint,path)
                write_csv(out/'training'/(name+'_history.csv'),result['history'])
                loaded,_=load_checkpoint(path,key);models.setdefault(family,{}).setdefault(seed,{})[arch]=loaded
                last=result['history'][-1]
                records.append({'name':name,'family':family,'architecture':arch,'seed':seed,'epoch':25,'parameters':PARAMETERS[arch],
                    'checkpoint_path':str(path.relative_to(out)),'checkpoint_sha256':sha256_file(path),'state_sha256':result['state_sha256'],
                    'initial_state_sha256':result['initial_state_sha256'],'updates':result['updates'],'fit_seconds':time.perf_counter()-start,
                    'best_epoch_diagnostic':result['best_epoch'],'first_train_bce':result['history'][0]['train_bce'],
                    'last_train_bce':last['train_bce'],'last_validation_bce':last['validation_bce'],'train_frames':last['train_frames'],
                    'positive_train_frames':last['positive_train_frames'],'validation_frames':last['validation_frames'],'copied_from_m5':False})
                write_json(out/'training_summary.json',{'models':records,'new_models':len(records)-6,'expected_new_models':30,'complete':len(records)==36})
                print(f'M6 trained {name}: {records[-1]["fit_seconds"]:.1f}s, validation BCE {last["validation_bce"]:.6f}',flush=True)
    return models,records


def predictions(entry,models):
    yield 'correlation',0,entry['correlation']
    families=['SINGLE','MIXED'] if entry['condition'].startswith(('R','OFFSET_')) else [entry['condition']]
    for family in families:
        prefix={'SINGLE':'single','MIXED':'mixed'}.get(family,'combo')
        for seed,architectures in sorted(models[family].items()):
            for arch,model in sorted(architectures.items()):yield f'{prefix}_{arch}',seed,predict_logits(model,arch,entry['x'])


def score(out,config,models,split,offsets=False):
    if split=='test' and not (out/'evaluation_gate.json').exists():raise ValueError('M6 test scoring requires frozen gate')
    tag=split+('_offsets' if offsets else '');stream=CSVStream(out/(tag+'_aggregate_scores.csv'));stats=Statistics();frames=[]
    cs=sorted({r['condition'] for r in read_jsonl(out/'feature_manifest.jsonl') if r['split']==split and r['condition'].startswith('OFFSET_')==offsets})
    for cid in cs:
        for kind in ['main','zero_control']:
            for e in load_entries(out,split,cid,kind):
                values=list(predictions(e,models));cache=out/'frame_predictions'/(e['id']+'.npz')
                np.savez_compressed(cache,**{f'{d}__{seed}':scores for d,seed,scores in values})
                frames.append({'id':e['id'],'path':str(cache.relative_to(out)),'sha256':sha256_file(cache),'frames':len(e['x'])})
                for n in config['evaluation']['aggregate_frames']:
                    for d,seed,scores in values:
                        for group in aggregate(scores,e['times'],n):
                            stream.add({'id':e['id'],'detector':d,'seed':seed,'frames':n,**group});stats.add(e,d,seed,n,group)
        print(f'M6 scored {tag}/{cid}',flush=True)
    stream.close();write_jsonl(out/(tag+'_frame_predictions.jsonl'),frames)
    return stats,{'aggregate_rows':stream.count,'received_files':len(frames)}


def pooled_mixed_validation(stats):
    result=Statistics()
    for key,b in stats.buckets.items():
        cid,split,kind,d,seed,n=key
        if not cid.startswith('R') or kind!='main' or not d.startswith('mixed_'):continue
        target=('MIXED',split,kind,d,seed,n)
        if target not in result.buckets:result.buckets[target]={'labels':[],'scores':[],'first_labels':[],'first_scores':[],'uids':[set(),set()],'speakers':set()}
        t=result.buckets[target]
        for field in ['labels','scores','first_labels','first_scores']:t[field].extend(b[field])
        for label in [0,1]:t['uids'][label].update(b['uids'][label])
        t['speakers'].update(b['speakers'])
    return result


def calibrate(stats,m5_calibration,hashes):
    if any(k[1]!='validation' for k in stats.buckets):raise ValueError('M6 calibration accepts validation only')
    old={(r['detector'],r['seed'],r['frames']):r for r in m5_calibration['settings'] if r['condition']=='C3'}
    pooled=pooled_mixed_validation(stats);pool={(r['detector'],r['seed'],r['frames']):r for r in pooled.metrics()}
    settings=[]
    for r in stats.metrics():
        if r['kind']!='main':continue
        cid,d,seed,n=[r[k] for k in ['condition','detector','seed','frames']]
        threshold=r['eer_threshold_interpolated'];origin=cid
        if cid.startswith('R'):
            if d.startswith('mixed_'):threshold=pool[(d,seed,n)]['eer_threshold_interpolated'];origin='pooled validation R6/R8/R12/R16'
            else:
                old_detector=d.replace('single_','model_');threshold=old[(old_detector,seed,n)]['threshold'];origin='M5/C3/8kbps validation'
        settings.append({'condition':cid,'detector':d,'seed':seed,'frames':n,'threshold':threshold,'source_split':'validation',
            'threshold_source_condition':origin,'per_condition_validation_threshold_diagnostic':r['eer_threshold_interpolated'],'validation_eer':r['eer']})
    return {'source_split':'validation','checkpoint_hashes':hashes,'settings':settings,
        'scope':'Fixed subset and declared channels only; arbitrary uploads scores_only','pooled_mixed_metrics':pooled.metrics()}


def fixed_rates(stats,calibration):
    lookup={(r['condition'],r['detector'],r['seed'],r['frames']):r for r in calibration['settings']};rows=[]
    for (cid,split,kind,d,seed,n),b in sorted(stats.buckets.items()):
        if kind!='main':continue
        base='R8' if cid.startswith('OFFSET_') else cid;r=lookup[(base,d,seed,n)];t=r['threshold']
        labels=np.asarray(b['labels']);scores=np.asarray(b['scores'])
        rows.append({'condition':cid,'split':split,'detector':d,'seed':seed,'frames':n,'threshold':t,'threshold_source':'validation',
            'threshold_source_condition':r['threshold_source_condition'],'far':float(np.mean(scores[labels==0]>=t)),
            'frr':float(np.mean(scores[labels==1]<t)),'negative_groups':int(sum(labels==0)),'positive_groups':int(sum(labels==1))})
    return rows


def run_extension(root:Path,config_path:Path,run_id=None):
    config=yaml.safe_load(config_path.read_text())
    if (config.get('experiment_kind')!='extension_v1' or config['training']['seeds']!=[2026,2027,2028]
        or config['training']['epochs']!=25 or config['training']['device']!='cpu' or config['training']['cpu_threads']!=1
        or config['training']['normalization']!='none' or config['evaluation']['aggregate_frames']!=[1,10,30]
        or config['extension']['bitrates_kbps']!=[6,8,12,16] or config['extension']['offset_ms']!=[0,2.5,5,10,15,20]
        or config['extension']['sampling_seed']!=20261018):
        raise ValueError('M6 v1 uses a fixed protocol; create a separately versioned experiment for changed settings')
    m5=root/config['extension']['m5_run'];assert json.loads((m5/'validation.json').read_text())['passed']
    assert json.loads((m5/'run_status.json').read_text())['status']=='complete'
    selected=json.loads((m5/'quality_selection.json').read_text());assert selected['ss_delta_db']==2 and selected['lpc_delta_db']==6
    audit=validate_formal(root,root/config['data']['dataset_directory'])
    run_id=run_id or time.strftime('%Y%m%d-%H%M%S');assert Path(run_id).name==run_id and run_id not in {'.','..'}
    out=root/config['output']['root']/config['experiment_id']/run_id;out.mkdir(parents=True,exist_ok=False)
    for folder in ['audio','features','embedded','references','frame_predictions','checkpoints','training','figures']:(out/folder).mkdir()
    (out/'config.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True));shutil.copy2(m5/'input_manifest.jsonl',out/'input_manifest.jsonl')
    for name in ['key.json','dataset_freeze.json','quality_selection.json']:shutil.copy2(m5/name,out/name)
    # Bind imported controls, quality, calibration and source configuration before extension scores.
    for name in ['calibration.json','training_summary.json','config.yaml','validation.json','quality_scores.jsonl','metrics.json']:
        shutil.copy2(m5/name,out/('m5_'+name))
    write_json(out/'m5_dependency.json',{'run':str(m5.relative_to(root)),'source_files':{n:sha256_file(m5/n) for n in ['config.yaml','input_manifest.jsonl','calibration.json','training_summary.json','validation.json']}})
    write_json(out/'environment.json',probe_environment());write_json(out/'codec_runtime.json',codec_runtime());write_json(out/'dataset_validation.json',audit)
    snapshot=out/'source_snapshot';shutil.copytree(Path(__file__).parent,snapshot,ignore=shutil.ignore_patterns('__pycache__'))
    write_json(out/'implementation_manifest.json',{'frozen_before_extension_data':True,'files':{str(p.relative_to(snapshot)):sha256_file(p) for p in sorted(snapshot.rglob('*')) if p.is_file()},'scope':'Exact source snapshot frozen before execution; Git release follows M5 archive completion'})
    protocol=root/'docs/M6评估协议.md'
    if not protocol.exists():protocol=root/'.cache/M6评估协议.md'
    shutil.copy2(protocol,out/'PROTOCOL.md')
    write_jsonl(out/'failures.jsonl',[]);key=load_key(out/'key.json');phases=[];start=time.perf_counter()
    def status(value):write_json(out/'run_status.json',{'status':value});print(value,flush=True)
    try:
        status('M6 preparing training and validation');phases.append(prepare(root,out,config,key,'train'));phases.append(prepare(root,out,config,key,'validation'))
        status('M6 training 30 new models, six fixed M5 controls');models,records=train(root,out,config,key)
        hashes={r['name']:r['checkpoint_sha256'] for r in records}
        status('M6 validation calibration');validation,vc=score(out,config,models,'validation')
        calibration=calibrate(validation,json.loads((out/'m5_calibration.json').read_text()),hashes);write_json(out/'calibration.json',calibration)
        metrics=validation.metrics();rates=fixed_rates(validation,calibration);del validation
        write_json(out/'evaluation_gate.json',{'config_hash':canonical_hash(config),'input_manifest_sha256':sha256_file(out/'input_manifest.jsonl'),
            'checkpoint_hashes':hashes,'training_selection_sha256':sha256_file(out/'training_selection.jsonl'),'calibration_sha256':sha256_file(out/'calibration.json'),
            'test_used_for_extension_training_or_selection':False,'scope':'Exploratory reuse of the M5 test speakers; no new untouched confirmation set'})
        status('M6 frozen exploratory test');phases.append(prepare(root,out,config,key,'test'));test,tc=score(out,config,models,'test')
        metrics+=test.metrics();rates+=fixed_rates(test,calibration);del test
        status('M6 known-start offset analysis');offset_count=prepare_offsets(root,out,config,key);offsets,oc=score(out,config,models,'test',offsets=True)
        metrics+=offsets.metrics();rates+=fixed_rates(offsets,calibration);del offsets
        write_csv(out/'metrics.csv',metrics);write_csv(out/'seed_summary.csv',seed_summary(metrics));write_csv(out/'fixed_threshold_rates.csv',rates)
        write_json(out/'metrics.json',{'experiment_kind':'extension_v1','experiment_id':config['experiment_id'],'run_id':run_id,'config_hash':canonical_hash(config),
            'scope':'E06-E09 exploratory paired LibriSpeech subset; not TIMIT or a new confirmation set','primary_epoch':25,'new_models':30,'fixed_models':6,
            'phase_counts':phases,'validation_counts':vc,'test_counts':tc,'offset_counts':oc,'offset_files':offset_count,'metrics':metrics,
            'metric_rows':len(metrics),'elapsed_s_before_audit':time.perf_counter()-start})
        status('M6 full artifact, score and representative training audit')
        from .m6_validation import validate_extension
        write_json(out/'validation.json',validate_extension(out,replay_seed=2026))
        from .m6_reporting import make_extension_report
        make_extension_report(out);status('complete');return out
    except Exception as exc:
        write_json(out/'run_status.json',{'status':'failed','error':repr(exc)});write_jsonl(out/'failures.jsonl',[{'stage':'fatal','error':repr(exc)}]);raise
