"""Frozen M6 received artifacts, exact M5 reuse, training budget selection."""
from __future__ import annotations
import hashlib,json
import shutil
from pathlib import Path
import numpy as np
from .audio import read_audio,resample_audio,write_audio
from .channels import apply_channel,opus_roundtrip
from .comparison import pcm_hash
from .features import extract_features
from .main_data import append_manifest,control_ids,variants_for
from .metrics import quality_scores
from .spectral import SpectralConfig
from .utils import canonical_hash,read_jsonl,sha256_file,write_json,write_jsonl


def conditions(config):
    return [{'id':f'R{rate}','method':'SS','delta_db':3,'bitrate_bps':rate*1000,'channel':f'opus8k_{rate}kbps'} for rate in config['extension']['bitrates_kbps']]+[
        {'id':f'E06_{channel}_{method}','method':method,'delta_db':2 if method=='SS' else 6,
         'bitrate_bps':8000 if channel=='O' else 0,'channel':'opus8k_8kbps' if channel=='O' else 'downsample_8k'}
        for channel in ['D','O'] for method in ['SS','SS_LPC']]


def record_features(root,out,config,key,base,destination):
    x,sr=read_audio(destination);cfg=SpectralConfig(**config['spectral'])
    f=extract_features(x,sr,key,cfg,vad='webrtc',vad_aggressiveness=2)
    cache=out/'features'/(base['id']+'.npz')
    np.savez_compressed(cache,features=f.features[f.valid_mask].astype(np.float32),features_f64=f.features,valid_mask=f.valid_mask,times_s=f.times_s)
    artifact={**base,'waveform_path':str(destination.relative_to(root)),'waveform_sha256':sha256_file(destination),
        'pcm_sha256':pcm_hash(x),'sample_rate':sr,'samples':len(x),'peak':float(abs(x).max())}
    feature={**base,'features_path':str(cache.relative_to(out)),'features_sha256':sha256_file(cache),
        'valid_frames':int(f.valid_mask.sum()),'waveform_sha256':artifact['waveform_sha256']}
    cover={**base,**f.metadata,'mean_abs_matched_feature':float(abs(f.features[f.valid_mask]).mean()),'supports_30_frames':int(f.valid_mask.sum())>=30}
    return artifact,feature,cover


def prepare(root,out,config,key,split):
    if split=='test' and not (out/'evaluation_gate.json').exists():raise ValueError('M6 test requires frozen evaluation gate')
    m5=root/config['extension']['m5_run'];source=read_jsonl(out/'input_manifest.jsonl');rows=[r for r in source if r['split']==split]
    old={r['id']:r for r in read_jsonl(m5/'audio_manifest.jsonl')}
    old_channels={r['id']:r for r in read_jsonl(m5/'channel_manifest.jsonl')}
    refs={r['utterance_id']:r for r in read_jsonl(m5/'reference_manifest.jsonl')}
    zero=control_ids(rows,10) if split!='train' else set()
    artifacts=[];features=[];channels=[];covers=[];qrows=[];embeds=[]
    for i,row in enumerate(rows):
        uid=row['utterance_id'];audio,sr=read_audio(root/row['waveform_path']);assert sr==16000
        variants,gain,peak,_=variants_for(audio,config,key);assert gain==refs[uid]['common_gain']
        reference_path=out/'references'/(uid+'.wav');shutil.copy2(root/refs[uid]['reference_path'],reference_path)
        reference,_=read_audio(reference_path);paths={}
        for method,d in [('SS',0),('SS',2),('SS',3),('SS_LPC',6)]:
            p=out/'embedded'/f'{uid}_{method}_d{d}.wav';original=m5/'embedded'/p.name
            if original.exists():shutil.copy2(original,p)
            else:write_audio(p,variants[(method,d)],16000)
            paths[(method,d)]=p
            embeds.append({'utterance_id':uid,'split':split,'method':method,'delta_db':d,'common_gain':gain,
                'path':str(p.relative_to(root)),'sha256':sha256_file(p),'pcm_sha256':pcm_hash(read_audio(p)[0]),
                'reference_path':str(reference_path.relative_to(root)),'reference_sha256':sha256_file(reference_path)})
        for c in conditions(config):
            for kind in ['main']+(['zero_control'] if uid in zero and c['id'].startswith('R') else []):
                for label in [0,1]:
                    delta=c['delta_db'] if label and kind=='main' else 0
                    emb=paths[(c['method'],delta)] if delta else paths[('SS',0)]
                    stem=f"{uid}_{c['id']}_{kind}_label{label}";destination=out/'audio'/(stem+'.wav')
                    base={k:row[k] for k in ['utterance_id','parent_utterance_id','speaker_id','gender','split']}
                    base.update(id=stem,condition=c['id'],channel=c['channel'],kind=kind,label=label,method=c['method'],
                        delta_db=delta,nominal_delta_db=c['delta_db'],bitrate_bps=c['bitrate_bps'],
                        embedded_path=str(emb.relative_to(root)),embedded_sha256=sha256_file(emb))
                    if c['id']=='R8':
                        old_id=f'{uid}_C3_{kind}_label{label}';r=old[old_id];p=root/r['waveform_path']
                        for suffix in ['.wav','.opus','.codec-input.wav']:shutil.copy2(p.with_suffix(suffix),destination.with_suffix(suffix))
                        metadata={**old_channels[old_id],'reused_from_m5_id':old_id,'m5_waveform_sha256':r['waveform_sha256']}
                        assert sha256_file(destination)==r['waveform_sha256']
                        metadata.update(compressed_path=str(destination.with_suffix('.opus')),reused_original_commands=True)
                    else:
                        x,_=read_audio(emb)
                        if c['bitrate_bps']:
                            _,_,metadata=opus_roundtrip(resample_audio(x,16000,8000),8000,destination,bitrate_bps=c['bitrate_bps'])
                        else:_,_,metadata=apply_channel(x,16000,'downsample_8k',destination,reference,uid,20261017)
                    artifact,feature,cover=record_features(root,out,config,key,base,destination)
                    artifacts.append(artifact);features.append(feature);covers.append(cover);channels.append({**metadata,**base})
                    if split!='train' and kind=='main':
                        received,_=read_audio(destination);ref8=resample_audio(reference,16000,8000)
                        q=quality_scores(ref8,received,8000,compute_pesq=False)
                        # NB directly at the received 8 kHz; WB is deliberately not inferred.
                        from pesq import pesq
                        nb=float(pesq(8000,ref8,received,'nb'))
                        qrows.append({**base,'received_pesq_nb':nb,'received_snr_db':q['embedding_snr_db'],
                            'scope':'total channel and embedding distortion, reference available','quality_failures':[]})
        if (i+1)%20==0 or i+1==len(rows):print(f'M6 prepare {split}: {i+1}/{len(rows)}',flush=True)
    for name,data in [('audio_manifest.jsonl',artifacts),('feature_manifest.jsonl',features),('channel_manifest.jsonl',channels),('coverage.jsonl',covers),('quality_scores.jsonl',qrows),('embedding_manifest.jsonl',embeds)]:append_manifest(out,name,data)
    return {'split':split,'sources':len(rows),'received':len(artifacts),'quality_rows':len(qrows),'minimum_frames':min(r['valid_frames'] for r in features)}


def sample_indices(entry,count,salt):
    if count>len(entry['x']) or count<1:raise ValueError('Invalid no-replacement frame quota')
    seed=int.from_bytes(hashlib.sha256((salt+'|'+entry['id']).encode()).digest()[:8],'little')
    return np.sort(np.random.default_rng(seed).choice(len(entry['x']),count,replace=False))


def subset(entry,indices):
    return {**entry,**{field:entry[field][indices] for field in ['x','times','correlation','frame_indices']}}


def training_sets(out,*,persist=True):
    from .deep_data import load_entries
    data={c:load_entries(out,'train',c) for c in ['R6','R8','R12','R16','E06_D_SS','E06_D_SS_LPC','E06_O_SS','E06_O_SS_LPC']}
    records=[];sets={};combo=[c for c in data if c.startswith('E06')]
    byfile={c:{(e['utterance_id'],e['label']):e for e in data[c]} for c in combo}
    quotas={k:min(len(byfile[c][k]['x']) for c in combo) for k in byfile[combo[0]]}
    for c in combo:
        sets[c]=[]
        for e in data[c]:
            ix=sample_indices(e,quotas[(e['utterance_id'],e['label'])],'M6-E06-20261018');sets[c].append(subset(e,ix))
            records.append({'training_family':c,'id':e['id'],'label':e['label'],'indices':ix.tolist()})
    mixed=[];rates=['R6','R8','R12','R16']
    for label in [0,1]:
        budget=sum(len(e['x']) for e in data['R8'] if e['label']==label)
        for ri,c in enumerate(rates):
            entries=[e for e in data[c] if e['label']==label];total=sum(len(e['x']) for e in entries);quota=budget//4+(ri<budget%4)
            if quota>total:raise ValueError('Insufficient mixed-rate frame pool')
            selected=np.sort(np.random.default_rng(20261018+label*100+ri).choice(total,quota,replace=False));start=0
            for e in entries:
                ix=selected[(selected>=start)&(selected<start+len(e['x']))]-start;start+=len(e['x'])
                if len(ix):mixed.append(subset(e,ix));records.append({'training_family':'MIXED','id':e['id'],'label':label,'indices':ix.tolist()})
    sets['MIXED']=mixed
    assert sum(len(e['x']) for e in mixed)==sum(len(e['x']) for e in data['R8'])
    if persist:write_jsonl(out/'training_selection.jsonl',records)
    counts={c:{'frames':sum(len(e['x']) for e in es),'label_frames':{str(l):sum(len(e['x']) for e in es if e['label']==l) for l in [0,1]},'files':len(es)} for c,es in sets.items()}
    counts['MIXED']['rate_label_frames']={c:{str(l):sum(len(e['x']) for e in mixed if e['condition']==c and e['label']==l) for l in [0,1]} for c in rates}
    if persist:write_json(out/'training_budget.json',counts)
    else:
        assert records==read_jsonl(out/'training_selection.jsonl')
        assert counts==json.loads((out/'training_budget.json').read_text())
    return sets


def prepare_offsets(root,out,config,key):
    artifacts=[];features=[];covers=[]
    source=[r for r in read_jsonl(out/'audio_manifest.jsonl') if r['split']=='test' and r['condition']=='R8' and r['kind']=='main']
    for offset in config['extension']['offset_ms']:
        for r in source:
            x,sr=read_audio(root/r['waveform_path']);cut=round(offset*sr/1000)
            stem=f"{r['utterance_id']}_OFFSET_{offset:g}_main_label{r['label']}";destination=out/'audio'/(stem+'.wav')
            if cut:write_audio(destination,x[cut:],sr)
            else:shutil.copy2(root/r['waveform_path'],destination)
            base={k:v for k,v in r.items() if k not in ['waveform_path','waveform_sha256','pcm_sha256','sample_rate','samples','peak']}
            base.update(id=stem,condition=f'OFFSET_{offset:g}',offset_ms=offset,cropped_samples=cut,parent_received_id=r['id'])
            a,f,c=record_features(root,out,config,key,base,destination);artifacts.append(a);features.append(f);covers.append(c)
        print(f'M6 offset {offset:g} ms prepared',flush=True)
    for name,rows in [('audio_manifest.jsonl',artifacts),('feature_manifest.jsonl',features),('coverage.jsonl',covers)]:append_manifest(out,name,rows)
    return len(artifacts)
