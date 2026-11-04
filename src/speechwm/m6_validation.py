"""Recompute M6 received features, all model predictions and all aggregated records."""
from __future__ import annotations
import csv,json,sys,subprocess
from pathlib import Path
import numpy as np
import yaml
from .audio import read_audio,resample_audio
from .channels import inspect_opus,codec_runtime
from .comparison import pcm_hash
from .deep_data import load_entries
from .features import aggregate,extract_features
from .formal_data import validate_formal
from .m6_data import training_sets,variants_for
from .m6_experiment import calibrate,fixed_rates,predictions
from .main_evaluation import Statistics,seed_summary
from .main_validation import check_csv_record
from .metrics import quality_scores
from .models import load_checkpoint
from .spectral import SpectralConfig
from .training import fit_model
from .utils import canonical_hash,read_jsonl,sha256_file


def audit_score_file(out,config,models,split,offsets=False):
    tag=split+('_offsets' if offsets else '');manifest={r['id']:r for r in read_jsonl(out/(tag+'_frame_predictions.jsonl'))}
    cs=sorted({r['condition'] for r in read_jsonl(out/'feature_manifest.jsonl') if r['split']==split and r['condition'].startswith('OFFSET_')==offsets})
    stats=Statistics();count=0;maximum=0.
    with (out/(tag+'_aggregate_scores.csv')).open() as stream:
        reader=csv.DictReader(stream)
        for cid in cs:
            for kind in ['main','zero_control']:
                for e in load_entries(out,split,cid,kind):
                    record=manifest[e['id']];path=out/record['path'];assert sha256_file(path)==record['sha256'] and record['frames']==len(e['x'])
                    values=list(predictions(e,models))
                    with np.load(path,allow_pickle=False) as cache:
                        assert set(cache.files)=={f'{d}__{seed}' for d,seed,_ in values}
                        for d,seed,scores in values:
                            error=float(np.max(abs(cache[f'{d}__{seed}']-scores)));assert error<=1e-6;maximum=max(maximum,error)
                    for n in config['evaluation']['aggregate_frames']:
                        for d,seed,scores in values:
                            for group in aggregate(scores,e['times'],n):
                                saved=next(reader);expected={'id':e['id'],'detector':d,'seed':seed,'frames':n,**group}
                                maximum=max(maximum,check_csv_record(saved,expected,1e-6));count+=1
                                stats.add(e,d,seed,n,{'score':float(saved['score']),'group_index':int(saved['group_index'])})
        assert next(reader,None) is None
    return stats,{'aggregate_rows':count,'received_files':len(manifest),'maximum_score_error':maximum}


def validate_extension(out:Path,replay_seed=2026):
    root=next(p for p in out.resolve().parents if (p/'pyproject.toml').exists())
    config=yaml.safe_load((out/'config.yaml').read_text());summary=json.loads((out/'metrics.json').read_text());gate=json.loads((out/'evaluation_gate.json').read_text())
    assert canonical_hash(config)==gate['config_hash']==summary['config_hash']
    assert sha256_file(out/'input_manifest.jsonl')==gate['input_manifest_sha256'] and not gate['test_used_for_extension_training_or_selection']
    data=validate_formal(root,root/config['data']['dataset_directory'])
    from .keys import load_key
    key=load_key(out/'key.json');cfg=SpectralConfig(**config['spectral']);source={r['utterance_id']:r for r in read_jsonl(out/'input_manifest.jsonl')}
    embeds=read_jsonl(out/'embedding_manifest.jsonl');byuid={}
    for r in embeds:byuid.setdefault(r['utterance_id'],[]).append(r)
    for uid,records in byuid.items():
        x,sr=read_audio(root/source[uid]['waveform_path']);variants,gain,_,_=variants_for(x,config,key)
        for r in records:
            assert r['common_gain']==gain and sha256_file(root/r['path'])==r['sha256']
            saved,rate=read_audio(root/r['path']);assert rate==16000 and pcm_hash(saved)==r['pcm_sha256']
            np.testing.assert_array_equal(saved,variants[(r['method'],r['delta_db'])].astype(np.float32).astype(np.float64))
            ref,_=read_audio(root/r['reference_path']);assert sha256_file(root/r['reference_path'])==r['reference_sha256']
            np.testing.assert_array_equal(ref,(x*gain).astype(np.float32).astype(np.float64))
    artifacts=read_jsonl(out/'audio_manifest.jsonl');index={r['id']:r for r in artifacts};features={r['id']:r for r in read_jsonl(out/'feature_manifest.jsonl')}
    channels={r['id']:r for r in read_jsonl(out/'channel_manifest.jsonl')};minimum=100000;opus=0;offsets=0;negatives={};zero={};reused=0
    overshoots=[];maximum_peak=0.;runtime=None
    for r in artifacts:
        for field in ['split','parent_utterance_id','speaker_id','gender']:assert r[field]==source[r['utterance_id']][field]
        path=root/r['waveform_path'];x,sr=read_audio(path);assert sr==8000 and len(x)==r['samples'] and np.isfinite(x).all()
        peak=float(abs(x).max());maximum_peak=max(maximum_peak,peak)
        assert sha256_file(path)==r['waveform_sha256'] and pcm_hash(x)==r['pcm_sha256']
        if peak>=1:
            # Floating-point Opus decoding can overshoot full scale. Preserve it,
            # and verify the actual decoder output rather than clip or rescale.
            assert r['bitrate_bps'] and not r['condition'].startswith('OFFSET_')
            if runtime is None:
                runtime=codec_runtime();assert runtime['executable_sha256']==json.loads((out/'codec_runtime.json').read_text())['executable_sha256']
            decoded=subprocess.run([runtime['executable'],'-hide_banner','-nostdin','-v','error','-i',str(path.with_suffix('.opus')),
                '-map','0:a:0','-ac','1','-ar','8000','-c:a','pcm_f32le','-f','f32le','pipe:1'],check=True,capture_output=True).stdout
            np.testing.assert_array_equal(np.frombuffer(decoded,dtype='<f4').astype(np.float64),x)
            overshoots.append({'id':r['id'],'split':r['split'],'peak':peak,'samples_at_or_above_one':int((abs(x)>=1).sum()),'original_stream_redecoded_exactly':True})
        if r['condition'].startswith('OFFSET_'):
            parent=index[r['parent_received_id']];original,_=read_audio(root/parent['waveform_path']);assert r['cropped_samples']==round(r['offset_ms']*8)
            np.testing.assert_array_equal(x,original[r['cropped_samples']:]);offsets+=1
        else:
            channel=channels[r['id']]
            if r['bitrate_bps']:
                compressed=path.with_suffix('.opus');inspection=inspect_opus(compressed)
                assert inspection==channel['inspection'] and sha256_file(compressed)==channel['compressed_sha256']
                assert inspection['packet_duration_ms']==[20] and inspection['packet_sizes_bytes']==[r['bitrate_bps']//400]
                assert inspection['payload_bitrate_bps']==r['bitrate_bps'] and inspection['container_pcm_duration_s']==3
                if r['condition']=='R8':assert sha256_file(path)==channel['m5_waveform_sha256'];reused+=1
                else:
                    codec_input,_=read_audio(path.with_suffix('.codec-input.wav'));embedded,_=read_audio(root/r['embedded_path'])
                    np.testing.assert_array_equal(codec_input,resample_audio(embedded,16000,8000).astype(np.float32).astype(np.float64))
                opus+=1
            else:
                embedded,_=read_audio(root/r['embedded_path']);np.testing.assert_array_equal(x,resample_audio(embedded,16000,8000).astype(np.float32).astype(np.float64))
        f=extract_features(x,sr,key,cfg,vad='webrtc',vad_aggressiveness=2);cache=out/features[r['id']]['features_path']
        assert sha256_file(cache)==features[r['id']]['features_sha256']
        with np.load(cache,allow_pickle=False) as z:
            for name,expected in [('features_f64',f.features),('features',f.features[f.valid_mask].astype(np.float32)),('times_s',f.times_s),('valid_mask',f.valid_mask)]:np.testing.assert_array_equal(z[name],expected)
        minimum=min(minimum,int(f.valid_mask.sum()))
        if not r['label']:
            pair=(r['utterance_id'],r['condition'],r['kind']);negatives[pair]=r['pcm_sha256']
        if r['kind']=='zero_control':zero.setdefault((r['utterance_id'],r['condition']),[]).append(r['pcm_sha256'])
    assert all(len(v)==2 and len(set(v))==1 for v in zero.values())
    training=json.loads((out/'training_summary.json').read_text());assert training['complete'] and training['new_models']==30 and len(training['models'])==36
    selection_hash=sha256_file(out/'training_selection.jsonl');budget_hash=sha256_file(out/'training_budget.json')
    sets=training_sets(out,persist=False);assert sha256_file(out/'training_selection.jsonl')==selection_hash==gate['training_selection_sha256'] and sha256_file(out/'training_budget.json')==budget_hash
    models={};replays=[];hashes={}
    for r in training['models']:
        path=out/r['checkpoint_path'];model,cp=load_checkpoint(path,key);assert sha256_file(path)==r['checkpoint_sha256']==gate['checkpoint_hashes'][r['name']]
        assert cp['state_sha256']==r['state_sha256'] and cp['epoch']==25;hashes[r['name']]=r['checkpoint_sha256']
        models.setdefault(r['family'],{}).setdefault(r['seed'],{})[r['architecture']]=model
        if r['copied_from_m5']:continue
        assert cp['binding']['config_hash']==summary['config_hash'] and cp['binding']['training_selection_sha256']==selection_hash
        history=list(csv.DictReader((out/'training'/(r['name']+'_history.csv')).open()));assert len(history)==25
        entries=sets[r['family']];cid='R8' if r['family']=='MIXED' else r['family'];val=load_entries(out,'validation',cid)
        assert cp['binding']['train_parent_ids']==sorted({e['parent_utterance_id'] for e in entries})
        assert cp['binding']['validation_parent_ids']==sorted({e['parent_utterance_id'] for e in val})
        if r['seed']==replay_seed:
            print('Replay '+r['name'],file=sys.stderr,flush=True)
            repeated=fit_model(entries,val,r['architecture'],cp['training_config'],train_split='train',validation_split='validation')
            assert repeated['state_sha256']==r['state_sha256'] and repeated['updates']==r['updates']
            for old,new in zip(history,repeated['history']):check_csv_record(old,new)
            replays.append(r['name'])
    budget=json.loads((out/'training_budget.json').read_text())
    assert len({budget[c]['frames'] for c in budget if c!='MIXED'})==1
    old_train=next(r for r in training['models'] if r['copied_from_m5']);assert budget['MIXED']['frames']==old_train['train_frames']
    assert all(r['updates']==old_train['updates'] for r in training['models'] if r['family']=='MIXED')
    calibration=json.loads((out/'calibration.json').read_text());assert sha256_file(out/'calibration.json')==gate['calibration_sha256']
    metrics=[];rates=[];audits={}
    for split,offset in [('validation',False),('test',False),('test',True)]:
        stats,proof=audit_score_file(out,config,models,split,offset);tag=split+('_offsets' if offset else '');audits[tag]=proof
        expected=summary['offset_counts' if offset else split+'_counts']
        assert proof['aggregate_rows']==expected['aggregate_rows'] and proof['received_files']==expected['received_files']
        metrics+=stats.metrics();rates+=fixed_rates(stats,calibration)
        if split=='validation':assert calibration==calibrate(stats,json.loads((out/'m5_calibration.json').read_text()),hashes)
    assert metrics==summary['metrics']
    m5_metrics=json.loads((out/'m5_metrics.json').read_text())['metrics']
    old={(r['detector'],r['seed'],r['frames']):r for r in m5_metrics if r['condition']=='C3' and r['split']=='test' and r['kind']=='main'}
    for r in metrics:
        if r['condition']=='R8' and r['split']=='test' and r['kind']=='main' and not r['detector'].startswith('mixed_'):
            base=old[(r['detector'].replace('single_','model_'),r['seed'],r['frames'])]
            for field in ['eer','eer_first_group_per_utterance','negative_groups','positive_groups','speakers']:assert r[field]==base[field]

    for name,records in [('metrics.csv',metrics),('seed_summary.csv',seed_summary(metrics)),('fixed_threshold_rates.csv',rates)]:
        with (out/name).open() as f:
            reader=csv.DictReader(f)
            for r in records:check_csv_record(next(reader),r)
            assert next(reader,None) is None
    null=[r for r in metrics if r['kind']=='zero_control'];assert len(null)==312 and all(r['eer']==.5 for r in null)
    # No-offset metrics must be identical to the original R8 condition.
    test_metrics={(r['condition'],r['detector'],r['seed'],r['frames']):r for r in metrics if r['split']=='test' and r['kind']=='main'}
    for r in metrics:
        if r['condition']=='OFFSET_0':
            other=test_metrics[('R8',r['detector'],r['seed'],r['frames'])]
            for field in ['eer','eer_first_group_per_utterance','negative_groups','positive_groups','speakers']:assert r[field]==other[field]
    strata=set();quality_max=0.
    from pesq import pesq
    for r in read_jsonl(out/'quality_scores.jsonl'):
        stratum=(r['split'],r['condition'],r['label'])
        if stratum in strata:continue
        a=index[r['id']];embedded=next(e for e in embeds if e['utterance_id']==r['utterance_id'])
        reference,_=read_audio(root/embedded['reference_path']);received,_=read_audio(root/a['waveform_path'])
        q=float(pesq(8000,resample_audio(reference,16000,8000),received,'nb'));error=abs(q-r['received_pesq_nb']);assert error<=1e-6
        quality_max=max(quality_max,error);strata.add(stratum)
    historical={}
    for stage in ['M1','M2','M3','M4','M5']:
        folder=root/'milestones'/stage;manifest=json.loads((folder/'snapshot_manifest.json').read_text())
        for r in manifest['files']:assert sha256_file(folder/r['path'])==r['sha256']
        historical[stage]=len(manifest['files'])
    failures=read_jsonl(out/'failures.jsonl');recovery=None
    if failures:
        # Keep the original failed audit record. Only this explicitly documented
        # peak-bound correction can resolve it, with unchanged primary artifacts.
        recovery=json.loads((out/'audit_recovery.json').read_text())
        assert failures==[{'stage':'fatal','error':'AssertionError()'}]
        assert recovery['resolved_check']=='floating_opus_decode_peak_bound'
        assert recovery['original_failures_sha256']==sha256_file(out/'failures.jsonl')
        assert sorted(recovery['overshoot_ids'])==sorted(r['id'] for r in overshoots)
        for name,digest in recovery['unchanged_primary_artifacts'].items():assert sha256_file(out/name)==digest
    return {'passed':True,'dataset_validation':data,'embedded_wavs_recomputed':len(embeds),'received_and_features_recomputed':len(artifacts),
        'opus_streams_verified':opus,'m5_r8_received_reused_exactly':reused,'offset_crops_verified':offsets,'zero_pcm_pairs':len(zero),
        'zero_eer_settings':len(null),'checkpoints_verified':36,'fully_retrained_settings':replays,'replay_seed':replay_seed,
        'score_audits':audits,'metric_rows_recomputed':len(metrics),'fixed_threshold_rate_rows':len(rates),'validation_thresholds':len(calibration['settings']),
        'received_pesq_strata_recomputed':len(strata),'pesq_maximum_error':quality_max,'minimum_valid_frames':minimum,
        'maximum_received_float_peak':maximum_peak,'floating_opus_overshoots_redecoded':overshoots,
        'resolved_initial_audit_failure':recovery is not None,'auditor_source_sha256':sha256_file(Path(__file__)),
        'historical_snapshot_files_verified':historical,'scope':f'All 36 model inference audited; full replay of {len(replays)} new settings, seed {replay_seed}; no other full training replay'}
