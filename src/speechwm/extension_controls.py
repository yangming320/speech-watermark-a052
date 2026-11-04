"""Prespecified paired-mask and speaker-cluster controls, separate from primary M6."""
from __future__ import annotations
import csv,gzip,json,shutil
from pathlib import Path
import numpy as np
from .evaluation import write_csv
from .features import aggregate
from .keys import load_key
from .main_evaluation import Statistics,seed_summary
from .models import load_checkpoint,predict_logits
from .utils import canonical_hash,read_jsonl,sha256_file,write_json,write_jsonl


def common_indices(masks):
    if not masks or len({len(m) for m in masks})!=1:raise ValueError('Aligned masks of equal length required')
    return np.flatnonzero(np.logical_and.reduce(masks))


def speaker_bootstrap(differences,replicates=1000,seed=20261018):
    x=np.asarray(differences,dtype=float)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():raise ValueError('At least two finite speaker-level differences required')
    draws=x[np.random.default_rng(seed).integers(0,len(x),size=(replicates,len(x)))].mean(axis=1)
    return {'mean_delta_error':float(x.mean()),'ci_low':float(np.quantile(draws,.025,method='linear')),
        'ci_high':float(np.quantile(draws,.975,method='linear')),'speakers':len(x),'replicates':replicates,'seed':seed},draws


def analyze_controls(root:Path,out:Path):
    proof=json.loads((out/'validation.json').read_text());assert proof['passed']
    trained=json.loads((out/'training_summary.json').read_text())['models']
    for mixed in [r for r in trained if r['family']=='MIXED']:
        single=next(r for r in trained if r['family']=='SINGLE' and r['architecture']==mixed['architecture'] and r['seed']==mixed['seed'])
        assert mixed['initial_state_sha256']==single['initial_state_sha256'] and mixed['updates']==single['updates'] and mixed['train_frames']==single['train_frames']
    destination=out/'controls';destination.mkdir(exist_ok=False)
    protocol=root/'docs/M6补充机制分析协议.md';shutil.copy2(protocol,destination/'PROTOCOL.md')
    write_json(destination/'binding.json',{'source_validation_sha256':sha256_file(out/'validation.json'),
        'source_gate_sha256':sha256_file(out/'evaluation_gate.json'),'source_training_summary_sha256':sha256_file(out/'training_summary.json'),'source_test_scores_sha256':sha256_file(out/'test_aggregate_scores.csv'),
        'protocol_sha256':sha256_file(protocol),'implementation_sha256':sha256_file(Path(__file__)),
        'primary_models_or_thresholds_changed':False})
    features=[r for r in read_jsonl(out/'feature_manifest.jsonl') if r['split']=='test' and r['kind']=='main' and r['condition'] in {'R6','R8','R12','R16'}]
    metadata={r['id']:r for r in features};pairs={}
    for r in features:pairs.setdefault(r['utterance_id'],{})[(r['condition'],r['label'])]=r
    key=load_key(out/'key.json');training=json.loads((out/'training_summary.json').read_text())['models'];models={}
    for r in training:
        if r['family'] in {'SINGLE','MIXED'}:models[(r['family'],r['seed'],r['architecture'])]=load_checkpoint(out/r['checkpoint_path'],key)[0]
    stats=Statistics();masks=[];checks=[];counter=0
    path=destination/'common_aggregate_scores.csv.gz'
    with gzip.open(path,'wt',newline='',compresslevel=6) as stream:
        fields=['id','detector','seed','frames','group_index','score'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for i,(uid,files) in enumerate(sorted(pairs.items())):
            assert len(files)==8
            caches={}
            for k,r in files.items():
                with np.load(out/r['features_path'],allow_pickle=False) as z:caches[k]={n:z[n] for n in z.files}
            ix=common_indices([c['valid_mask'] for c in caches.values()])
            masks.append({'utterance_id':uid,'speaker_id':next(iter(files.values()))['speaker_id'],'indices':ix.tolist(),
                'valid_frames':len(ix),'eligible_30':len(ix)>=30,'source_feature_hashes':{r['id']:r['features_sha256'] for r in files.values()}})
            for k,r in sorted(files.items()):
                cache=caches[k];original_indices=np.flatnonzero(cache['valid_mask']);positions=np.searchsorted(original_indices,ix)
                np.testing.assert_array_equal(original_indices[positions],ix)
                e={**r,'condition':r['condition']+'_common_mask'};times=cache['times_s'][ix]
                with np.load(out/'frame_predictions'/(r['id']+'.npz'),allow_pickle=False) as predictions:
                    for name in predictions.files:
                        d,seed_text=name.rsplit('__',1);seed=int(seed_text);scores=predictions[name][positions]
                        if i==0:
                            if d=='correlation':actual=cache['features_f64'][ix].mean(axis=1)
                            else:actual=predict_logits(models[('SINGLE' if d.startswith('single_') else 'MIXED',seed,d[-1])],d[-1],cache['features_f64'][ix].astype(np.float32))
                            error=float(np.max(abs(actual-scores))) if len(scores) else 0.;assert error<=3e-6
                            checks.append({'id':r['id'],'detector':d,'seed':seed,'maximum_error':error})
                        for n in [1,10,30]:
                            for group in aggregate(scores,times,n):
                                writer.writerow({'id':r['id'],'detector':d,'seed':seed,'frames':n,'group_index':group['group_index'],'score':group['score']});counter+=1
                                stats.add(e,d,seed,n,group)
            if (i+1)%40==0:print(f'Common-mask controls {i+1}/{len(pairs)}',flush=True)
    metrics=stats.metrics();write_csv(destination/'common_mask_metrics.csv',metrics);write_csv(destination/'common_mask_seed_summary.csv',seed_summary(metrics));write_jsonl(destination/'common_masks.jsonl',masks)
    # Reload the compressed records and independently reconstruct every metric.
    repeated=Statistics()
    with gzip.open(path,'rt') as stream:
        for r in csv.DictReader(stream):repeated.add({**metadata[r['id']],'condition':metadata[r['id']]['condition']+'_common_mask'},r['detector'],int(r['seed']),int(r['frames']),{'score':float(r['score']),'group_index':int(r['group_index'])})
    assert repeated.metrics()==metrics
    calibration=json.loads((out/'calibration.json').read_text());thresholds={(r['condition'],r['detector'],r['seed'],r['frames']):r for r in calibration['settings']}
    first={}
    with (out/'test_aggregate_scores.csv').open() as f:
        for r in csv.DictReader(f):
            if r['id'] not in metadata or int(r['frames'])!=30 or int(r['group_index'])!=0 or not r['detector'].startswith(('single_','mixed_')):continue
            e=metadata[r['id']];t=thresholds[(e['condition'],r['detector'],int(r['seed']),30)]['threshold'];prediction=int(float(r['score'])>=t)
            first[(e['condition'],e['utterance_id'],e['label'],r['detector'],int(r['seed']))]={'error':int(prediction!=e['label']),'score':float(r['score']),'threshold':t,'prediction':prediction,'id':r['id'],'speaker_id':e['speaker_id']}
    write_jsonl(destination/'first_group_errors.jsonl',[dict(zip(['condition','utterance_id','label','detector','seed'],k))|v for k,v in first.items()])
    intervals=[];speaker_rows=[];draw_rows=[];examples=[]
    audio={r['id']:r for r in read_jsonl(out/'audio_manifest.jsonl')}
    for cid in ['R6','R8','R12','R16']:
        for arch in ['A','B']:
            byspeaker={};eligible=[]
            for uid in sorted(pairs):
                keys=[(cid,uid,label,prefix+'_'+arch,seed) for label in [0,1] for prefix in ['single','mixed'] for seed in [2026,2027,2028]]
                if any(k not in first for k in keys):continue
                eligible.append(uid);speaker=first[keys[0]]['speaker_id']
                delta=np.mean([first[(cid,uid,label,'mixed_'+arch,seed)]['error']-first[(cid,uid,label,'single_'+arch,seed)]['error'] for label in [0,1] for seed in [2026,2027,2028]])
                byspeaker.setdefault(speaker,[]).append(float(delta))
                for label in [0,1]:
                    a=first[(cid,uid,label,'single_'+arch,2026)];b=first[(cid,uid,label,'mixed_'+arch,2026)]
                    categories=[]
                    if a['error']:categories.append('single_false_positive' if label==0 else 'single_false_negative')
                    if a['error'] and not b['error']:categories.append('improved')
                    if not a['error'] and b['error']:categories.append('regressed')
                    for category in categories:
                        if any(r['condition']==cid and r['architecture']==arch and r['category']==category for r in examples):continue
                        example={'condition':cid,'architecture':arch,'category':category,'utterance_id':uid,'label':label,'seed':2026,'single':a,'mixed':b}
                        target=destination/'cases'/(a['id']+'.wav');target.parent.mkdir(exist_ok=True);shutil.copy2(root/audio[a['id']]['waveform_path'],target)
                        example.update(audio_path=str(target.relative_to(destination)),audio_sha256=sha256_file(target));examples.append(example)
            differences=[]
            for speaker,values in sorted(byspeaker.items()):
                mean=float(np.mean(values));differences.append(mean);speaker_rows.append({'condition':cid,'architecture':arch,'speaker_id':speaker,'paired_utterances':len(values),'delta_error':mean})
            result,draws=speaker_bootstrap(differences);intervals.append({'condition':cid,'architecture':arch,**result,'paired_utterances':len(eligible),
                'metric':'mean over 3 seeds of first-30-frame fixed-threshold balanced error; mixed minus single; speaker-cluster bootstrap'})
            draw_rows.extend({'condition':cid,'architecture':arch,'replicate':i,'delta_error':float(v)} for i,v in enumerate(draws))
    write_csv(destination/'speaker_bootstrap.csv',intervals);write_csv(destination/'speaker_differences.csv',speaker_rows);write_csv(destination/'bootstrap_replicates.csv',draw_rows);write_jsonl(destination/'failure_examples.jsonl',examples)
    original=list(csv.DictReader((out/'seed_summary.csv').open()));common=seed_summary(metrics)
    def mean(rows,c,d):return 100*float(next(r for r in rows if r['condition']==c and r['detector']==d and int(r['frames'])==30)['eer_mean'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for d in ['correlation','single_B','mixed_B']:
        color={'correlation':'#277da1','single_B':'#f8961e','mixed_B':'#43aa8b'}[d]
        axes[0].plot([6,8,12,16],[mean(original,f'R{r}',d) for r in [6,8,12,16]],'o-',color=color,label=d+' original VAD')
        axes[0].plot([6,8,12,16],[mean(common,f'R{r}_common_mask',d) for r in [6,8,12,16]],'x--',color=color,label=d+' common VAD')
    axes[0].set(xlabel='Opus rate (kbps)',ylabel='30-frame EER (%)',title='Shared VAD across rates and labels');axes[0].legend(fontsize=7.5);axes[0].grid(alpha=.2)
    for arch,shift in [('A',-.15),('B',.15)]:
        rows=[r for r in intervals if r['architecture']==arch];m=np.asarray([r['mean_delta_error']*100 for r in rows]);lo=np.asarray([r['ci_low']*100 for r in rows]);hi=np.asarray([r['ci_high']*100 for r in rows])
        axes[1].errorbar(np.asarray([6,8,12,16])+shift,m,yerr=[np.maximum(0,m-lo),np.maximum(0,hi-m)],fmt='o',capsize=3,label='Model '+arch)
    axes[1].axhline(0,color='black',lw=.7);axes[1].set(xlabel='Opus rate (kbps)',ylabel='Mixed minus single error (percentage points)',title='First group: speaker bootstrap (1000 draws)');axes[1].legend();axes[1].grid(alpha=.2)
    fig.savefig(destination/'mechanism_controls.png',dpi=160);plt.close(fig)
    lines=['# M6 补充机制分析','','这两项分析在扩展测试前单独登记，不改变主模型、测试源、强度或阈值。共同帧控制使用成对标签和四通道信息，只用于分析，不是独立盲检测流程。','','## 共同有效帧控制','','| kb/s，30 帧 | 相关 原始 / 交集 EER % | 单 B 原始 / 交集 % | 混合 B 原始 / 交集 % |','| --- | --- | --- | --- |']
    for rate in [6,8,12,16]:lines.append('| '+str(rate)+' | '+' | '.join(f'{mean(original,f"R{rate}",d):.3f} / {mean(common,f"R{rate}_common_mask",d):.3f}' for d in ['correlation','single_B','mixed_B'])+' |')
    lines+=['',f"交集后仍有 30 帧的成对语句 {sum(r['eligible_30'] for r in masks)}/{len(masks)}；所有码率与标签使用相同原始索引。EER 变化提示 VAD 组成影响的大小，不能据此把所有差异都归因为 VAD；特征本身和固定阈值也可能变化。",'','## 首组固定阈值误差的说话人重采样','','| kb/s | 模型 | 混合−单码率误差 百分点 | 95% 百分位区间 | 配对语句 / 说话人 |','| --- | --- | ---: | --- | --- |']
    for r in intervals:lines.append(f"| {r['condition'][1:]} | {r['architecture']} | {100*r['mean_delta_error']:+.3f} | [{100*r['ci_low']:+.3f}, {100*r['ci_high']:+.3f}] | {r['paired_utterances']} / {r['speakers']} |")
    lines+=['','误差按每条语句首个 30 帧组、两标签与三个种子平均，再按说话人整体重采样 1000 次，固定种子 20261018。负数表示混合模型误差较低。该区间描述此固定子集的说话人重采样，不是主表全部帧组 EER 的区间，也不保证新语料总体覆盖。单码率与混合模型分别使用其预先规定的验证阈值，所以误差差异同时包含校准范围设计。跨零区间不作稳定改善结论。','','![共同帧与说话人控制](mechanism_controls.png)','','固定排序保留首个假阳性、假阴性、改进与退化样例，见 failure_examples.jsonl 与 cases/；类别没有出现时不填造案例。不根据这些样例改动已冻结模型。完整分数、交集索引、bootstrap 抽样、模型抽查与输入绑定见同目录。','']
    (destination/'M6补充机制分析.md').write_text('\n'.join(lines))
    result={'passed':True,'common_mask_metric_rows':len(metrics),'common_aggregate_rows_recomputed':counter,'common_pairs':len(masks),
        'eligible_30_pairs':sum(r['eligible_30'] for r in masks),'direct_inference_sample_checks':checks,'bootstrap_settings':len(intervals),'bootstrap_replicates_per_setting':1000,
        'failure_examples':len(examples),'primary_thresholds_unchanged':True,'scope':'Prespecified supplementary analysis; original run source and validation hashes bound; no new confirmation set'}
    write_json(destination/'validation.json',result)
    make_training_diagnostics(out)
    files=[p for p in destination.rglob('*') if p.is_file()];write_json(destination/'manifest.json',{'files':[{'path':str(p.relative_to(destination)),'sha256':sha256_file(p)} for p in sorted(files)]})
    return result


def validate_controls(out:Path):
    from .main_validation import check_csv_record
    d=out/'controls';manifest=json.loads((d/'manifest.json').read_text())
    for r in manifest['files']:
        assert not Path(r['path']).is_absolute() and '..' not in Path(r['path']).parts
        assert sha256_file(d/r['path'])==r['sha256']
    binding=json.loads((d/'binding.json').read_text())
    for file,key in [('validation.json','source_validation_sha256'),('evaluation_gate.json','source_gate_sha256'),('training_summary.json','source_training_summary_sha256'),('test_aggregate_scores.csv','source_test_scores_sha256')]:assert sha256_file(out/file)==binding[key]
    masks=read_jsonl(d/'common_masks.jsonl');features=[r for r in read_jsonl(out/'feature_manifest.jsonl') if r['split']=='test' and r['kind']=='main' and r['condition'] in {'R6','R8','R12','R16'}]
    byuid={};metadata={r['id']:r for r in features}
    for r in features:byuid.setdefault(r['utterance_id'],{})[(r['condition'],r['label'])]=r
    stats=Statistics();count=0
    with gzip.open(d/'common_aggregate_scores.csv.gz','rt') as stream:
        reader=csv.DictReader(stream)
        for record in masks:
            files=byuid[record['utterance_id']];caches={}
            for k,r in files.items():
                with np.load(out/r['features_path'],allow_pickle=False) as z:caches[k]={n:z[n] for n in z.files}
            ix=common_indices([z['valid_mask'] for z in caches.values()]);assert ix.tolist()==record['indices'] and len(ix)==record['valid_frames']
            assert record['source_feature_hashes']=={r['id']:r['features_sha256'] for r in files.values()}
            for k,r in sorted(files.items()):
                cache=caches[k];positions=np.searchsorted(np.flatnonzero(cache['valid_mask']),ix);times=cache['times_s'][ix]
                e={**r,'condition':r['condition']+'_common_mask'}
                with np.load(out/'frame_predictions'/(r['id']+'.npz'),allow_pickle=False) as predictions:
                    for name in predictions.files:
                        detector,seed_text=name.rsplit('__',1);seed=int(seed_text)
                        for n in [1,10,30]:
                            for group in aggregate(predictions[name][positions],times,n):
                                check_csv_record(next(reader),{'id':r['id'],'detector':detector,'seed':seed,'frames':n,'group_index':group['group_index'],'score':group['score']})
                                stats.add(e,detector,seed,n,group);count+=1
        assert next(reader,None) is None
    for name,values in [('common_mask_metrics.csv',stats.metrics()),('common_mask_seed_summary.csv',seed_summary(stats.metrics()))]:
        with (d/name).open() as f:
            reader=csv.DictReader(f)
            for row in values:check_csv_record(next(reader),row)
            assert next(reader,None) is None
    saved=read_jsonl(d/'first_group_errors.jsonl');lookup={(r['condition'],r['utterance_id'],r['label'],r['detector'],r['seed']):r for r in saved};seen=set()
    thresholds={(r['condition'],r['detector'],r['seed'],r['frames']):r['threshold'] for r in json.loads((out/'calibration.json').read_text())['settings']}
    with (out/'test_aggregate_scores.csv').open() as f:
        for row in csv.DictReader(f):
            if row['id'] not in metadata or int(row['frames'])!=30 or int(row['group_index'])!=0 or not row['detector'].startswith(('single_','mixed_')):continue
            e=metadata[row['id']];k=(e['condition'],e['utterance_id'],e['label'],row['detector'],int(row['seed']));old=lookup[k];score=float(row['score']);threshold=thresholds[(e['condition'],row['detector'],int(row['seed']),30)]
            assert old['score']==score and old['threshold']==threshold and old['prediction']==int(score>=threshold) and old['error']==int(int(score>=threshold)!=e['label']);seen.add(k)
    assert seen==set(lookup)
    intervals=list(csv.DictReader((d/'speaker_bootstrap.csv').open()));speakers=list(csv.DictReader((d/'speaker_differences.csv').open()));repeated={}
    for cid in ['R6','R8','R12','R16']:
        for arch in ['A','B']:
            byspeaker={};eligible=0
            for uid in sorted(byuid):
                keys=[(cid,uid,label,prefix+'_'+arch,seed) for label in [0,1] for prefix in ['single','mixed'] for seed in [2026,2027,2028]]
                if any(k not in lookup for k in keys):continue
                delta=float(np.mean([lookup[(cid,uid,label,'mixed_'+arch,seed)]['error']-lookup[(cid,uid,label,'single_'+arch,seed)]['error'] for label in [0,1] for seed in [2026,2027,2028]]));speaker=lookup[keys[0]]['speaker_id'];byspeaker.setdefault(speaker,[]).append(delta);eligible+=1
            actual=[]
            for speaker,values in sorted(byspeaker.items()):
                row=next(r for r in speakers if r['condition']==cid and r['architecture']==arch and r['speaker_id']==speaker)
                assert float(row['delta_error'])==float(np.mean(values)) and int(row['paired_utterances'])==len(values);actual.append(float(np.mean(values)))
            result,draws=speaker_bootstrap(actual);old=next(r for r in intervals if r['condition']==cid and r['architecture']==arch)
            assert int(old['paired_utterances'])==eligible
            for field in ['mean_delta_error','ci_low','ci_high','speakers','replicates','seed']:assert float(old[field])==result[field]
            repeated[(cid,arch)]=draws
    draws_count=0
    with (d/'bootstrap_replicates.csv').open() as f:
        for r in csv.DictReader(f):assert float(r['delta_error'])==repeated[(r['condition'],r['architecture'])][int(r['replicate'])];draws_count+=1
    assert draws_count==8000
    return {'passed':True,'files_hashed':len(manifest['files']),'common_aggregate_rows_recomputed':count,'common_metric_rows':len(stats.metrics()),
        'first_group_records_verified':len(saved),'speaker_bootstrap_settings':len(intervals),'bootstrap_draws_verified':draws_count,'scope':'All paired-mask groups and speaker bootstrap recomputed from audited primary caches/CSV'}


def make_training_diagnostics(out:Path):
    import matplotlib.pyplot as plt
    records=[r for r in json.loads((out/'training_summary.json').read_text())['models'] if not r['copied_from_m5']]
    families=sorted({r['family'] for r in records});fig,axes=plt.subplots(5,2,figsize=(11,14),layout='constrained')
    lines=['# M6 三十设置训练诊断','','所有主模型固定第 25 轮。下图展示预定种子 2026 的全部五个训练族与 A/B，没有按效果挑曲线；表保留全部三种子。最佳验证轮仅作诊断，未用于替换测试权重。混合模型训练期验证诊断为 R8，最终阈值才使用四码率 validation 池。','',
        '![预定种子训练曲线](training_diagnostics.png)','','| 训练族 | 模型 | 种子 | 首轮训练 BCE | 末轮训练 BCE | 末轮验证 BCE | 最佳验证轮 诊断 | 更新数 | 训练秒 |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row,family in enumerate(families):
        for col,arch in enumerate(['A','B']):
            model=next(r for r in records if r['family']==family and r['architecture']==arch and r['seed']==2026)
            history=list(csv.DictReader((out/'training'/(model['name']+'_history.csv')).open()));ax=axes[row,col]
            for field,label in [('train_bce','Train'),('validation_bce','Validation')]:ax.plot(range(1,26),[float(r[field]) for r in history],label=label)
            ax.set(title=model['name'],xlabel='Epoch',ylabel='BCE loss');ax.grid(alpha=.2);ax.legend(fontsize=8)
    for r in records:lines.append(f"| {r['family']} | {r['architecture']} | {r['seed']} | {r['first_train_bce']:.6f} | {r['last_train_bce']:.6f} | {r['last_validation_bce']:.6f} | {r['best_epoch_diagnostic']} | {r['updates']} | {r['fit_seconds']:.2f} |")
    fig.savefig(out/'controls/training_diagnostics.png',dpi=150);plt.close(fig)
    lines+=['','原始逐轮损失、验证 EER 和每轮 shuffle SHA 在 training/ 下，RNG/优化器状态与主权重绑定。不同训练族损失尺度和可分性不同，不把较低 BCE 直接解释为跨分布泛化；测试 EER 与固定阈值误差另表。','']
    (out/'controls/M6训练诊断.md').write_text('\n'.join(lines));return len(records)
