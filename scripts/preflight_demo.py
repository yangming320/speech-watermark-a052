"""Verify a packaged demo without a training corpus; current platform only."""
import argparse,base64,json,platform,time,io,resource,sys,os
from pathlib import Path
import numpy as np
from speechwm.demo import DemoService
from speechwm.models import infer_audio
from speechwm.audio import read_audio
from speechwm.features import aggregate
from speechwm.utils import sha256_file,write_json

def check(root):
    start=time.perf_counter();service=DemoService(root,root/'deployment/models.json')
    fixture=root/'deployment/samples/clean.wav';data=base64.b64encode(fixture.read_bytes()).decode()
    traces=[]
    for detector in ['correlation',*service.models]:
        q={'mode':'chain','audio_base64':data,'method':'ss','delta':3,'channel':'opus8','detector':detector,'frames':30,'allow_resample':True}
        family=detector
        if 'LPC' in family:q.update(method='lpc',delta=6)
        elif 'SS2' in family:q.update(delta=2)
        if '_D_' in family or '_C1_' in family:q.update(channel='downsample')
        if '_C1_' in family:q.update(delta=1)
        first=service.process(q)
        # A new service has no previous source or in-memory job context.
        second=DemoService(root,root/'deployment/models.json').process({'mode':'detect','audio_base64':first['audio']['detector_input']['base64'],
            'detector':detector,'frames':30})
        assert first['groups']==second['groups'] and first['decision'] is None and 'quality' not in second
        assert first['groups'] and first['coverage']['valid_frames']>=30
        traces.append({'detector':detector,'groups':len(first['groups']),'elapsed_chain_s':first['elapsed_s'],'elapsed_detect_s':second['elapsed_s'],
            'first_score':first['groups'][0]['score'],'same_after_received_only_replay':True})
    golden=json.loads((root/'deployment/golden_scores.json').read_text());golden_checks=[]
    for case in golden['cases']:
        received=root/case['audio_path'];assert sha256_file(received)==case['audio_sha256']
        assert service.manifest['models'][case['detector']]['sha256']==case['checkpoint_sha256']
        actual=service.process({'mode':'detect','audio_base64':base64.b64encode(received.read_bytes()).decode(),'detector':case['detector'],'frames':case['frames']})
        assert len(actual['groups'])==len(case['groups'])>0;maximum=0.
        for current,expected in zip(actual['groups'],case['groups']):
            for key,value in expected.items():
                error=abs(current[key]-value);assert error<=(golden['score_tolerance'] if key=='score' else 0.)
                if key=='score':maximum=max(maximum,error)
        golden_checks.append({'detector':case['detector'],'groups':len(actual['groups']),'maximum_score_error':maximum})
    for channel in ['none','downsample','noise20','noise15','opus6','opus8','opus12','opus16']:
        r=service.process({'mode':'chain','audio_base64':data,'method':'ss','delta':3,'channel':channel,'detector':'correlation','frames':30})
        assert r['groups'] and r['decision'] is None
        if channel.startswith('opus'):assert r['channel']['inspection']['payload_bitrate_bps']==int(channel[4:])*1000
    import soundfile as sf
    clip,sr=read_audio(fixture);long_clip=np.tile(clip,10);buffer=io.BytesIO();sf.write(buffer,long_clip,sr,format='WAV',subtype='FLOAT')
    thirty=service.process({'mode':'chain','audio_base64':base64.b64encode(buffer.getvalue()).decode(),'method':'ss','delta':3,'channel':'opus8','detector':'M5_C3_A','frames':30})
    assert thirty['input']['duration_s']==30 and thirty['groups']
    return {'passed':True,'platform':platform.platform(),'machine':platform.machine(),'logical_cpu_count':os.cpu_count(),
        'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
        'thirty_second_chain':{'elapsed_s':thirty['elapsed_s'],'real_time_factor':thirty['elapsed_s']/30,'valid_frames':thirty['coverage']['valid_frames'],'scope':'10 copies of public 3-second sample concatenated for latency boundary check only, not a new speech experiment'},'models':len(service.models),'settings':traces,'channels':8,'fixed_received_golden_checks':golden_checks,
        'elapsed_s':time.perf_counter()-start,'manifest_sha256':sha256_file(root/'deployment/models.json'),
        'scope':'Current platform, packaged weights/key/sample only; no training corpus, no cloud external-access claim'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--output',type=Path,required=True);a=p.parse_args();target=a.output
    if target.exists():raise FileExistsError(target)
    result=check(a.root.resolve());write_json(target,result);print(json.dumps({'passed':True,'models':result['models'],'channels':result['channels']}))
if __name__=='__main__':main()
