"""Loopback HTTP chain and received-only replay checks for the packaged demo."""
import argparse,base64,io,json,time,urllib.error,urllib.parse,urllib.request
from pathlib import Path
import numpy as np
import soundfile as sf
from speechwm.utils import canonical_hash,sha256_file,write_json

def verify(root,url,token,out):
    if urllib.parse.urlparse(url).hostname not in {'127.0.0.1','localhost','::1'}:raise ValueError('This verification script is restricted to loopback')
    out.mkdir(parents=True,exist_ok=False);records=[]
    info=json.loads(urllib.request.urlopen(url+'/api/info',timeout=10).read())
    assert info['status']=='ready' and info['manifest_hash']==canonical_hash(json.loads((root/'deployment/models.json').read_text()))
    audio=base64.b64encode((root/'deployment/samples/clean.wav').read_bytes()).decode()
    def post(q,expected=200,provided_token=token):
        req=urllib.request.Request(url+'/api/process',data=json.dumps(q).encode(),headers={'Content-Type':'application/json','X-SpeechWM-Token':provided_token})
        start=time.perf_counter()
        try:
            with urllib.request.urlopen(req,timeout=60) as response:status=response.status;result=json.loads(response.read())
        except urllib.error.HTTPError as exc:status=exc.code;result=json.loads(exc.read())
        assert status==expected,(status,result);return result,time.perf_counter()-start
    # Each new request supplies only its own waveform and options; no cookie/context state.
    detectors=[n for n in info['models'] if n in {'M5_C3_A','M6_MIXED_B','M6_E06_O_LPC_A','M6_E06_D_SS2_B'}]
    for name in detectors:
        q={'mode':'chain','audio_base64':audio,'method':'lpc' if 'LPC' in name else 'ss','delta':6 if 'LPC' in name else 2 if 'SS2' in name else 3,
            'channel':'downsample' if '_D_' in name else 'opus8','detector':name,'frames':30}
        first,seconds=post(q);assert first['groups'] and first['decision'] is None
        second,replay_seconds=post({'mode':'detect','audio_base64':first['audio']['detector_input']['base64'],'detector':name,'frames':30})
        assert first['groups']==second['groups'] and 'quality' not in second and second['decision'] is None
        folder=out/name;folder.mkdir();compact={k:v for k,v in first.items() if k!='audio'};write_json(folder/'chain.json',compact);write_json(folder/'received_only.json',{k:v for k,v in second.items() if k!='audio'})
        for key,item in first['audio'].items():(folder/(key+'.wav')).write_bytes(base64.b64decode(item['base64']))
        records.append({'detector':name,'groups':len(first['groups']),'chain_roundtrip_s':seconds,'received_only_roundtrip_s':replay_seconds,'received_only_scores_identical':True})
    invalid,_=post({'mode':'detect','audio_base64':audio,'detector':'missing'},expected=400);assert 'error' in invalid
    _,_=post({'audio_base64':audio},expected=403,provided_token='invalid')
    zero=io.BytesIO();sf.write(zero,np.zeros(8000),8000,format='WAV');invalid,_=post({'audio_base64':base64.b64encode(zero.getvalue()).decode(),'mode':'detect'},expected=400);assert '静音' in invalid['error']
    summary={'passed':True,'settings':records,'models_available':len(info['models']),'rejected_missing_model':True,'rejected_silence':True,'rejected_invalid_request_token':True,
        'manifest_sha256':sha256_file(root/'deployment/models.json'),'scope':'Loopback HTTP, current machine; no cloud external-access claim'}
    write_json(out/'verification.json',summary);return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--ready-file',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();ready=json.loads(a.ready_file.read_text());r=verify(a.root.resolve(),ready['url'],ready['api_token'],a.output)
    print(json.dumps({'passed':True,'settings':len(r['settings']),'models_available':r['models_available']}))
if __name__=='__main__':main()
