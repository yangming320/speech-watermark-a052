"""Local, single-job WAV demonstration using the same watermark, channel and inference APIs."""
from __future__ import annotations
import argparse,base64,hashlib,io,json,os,secrets,tempfile,threading,time
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import numpy as np
import soundfile as sf
from .audio import prepare_audio,read_audio,resample_audio,write_audio
from .channels import apply_channel,codec_runtime,opus_roundtrip
from .embedding import embed,embed_lpc
from .features import aggregate,extract_features,score_frames
from .keys import load_key
from .metrics import quality_scores
from .models import load_checkpoint,predict_logits
from .utils import canonical_hash,sha256_file,write_json

MAX_BYTES=12*1024*1024


class DemoService:
    def __init__(self,root,manifest):
        self.root=Path(root).resolve();self.manifest=json.loads(Path(manifest).read_text());self.lock=threading.Lock()
        self.key=load_key(self.root/self.manifest['key']['path'])
        if sha256_file(self.root/self.manifest['key']['path'])!=self.manifest['key']['sha256']:raise ValueError('Key hash mismatch')
        self.models={}
        for name,record in self.manifest['models'].items():
            path=self.root/record['path']
            if sha256_file(path)!=record['sha256']:raise ValueError('Checkpoint hash mismatch: '+name)
            model,checkpoint=load_checkpoint(path,self.key);self.models[name]=(model,checkpoint)
        self.codec=codec_runtime()

    def info(self):
        return {'status':'ready','maximum_duration_s':30,'single_job':True,'models':{n:{'architecture':c['architecture'],'epoch':c['epoch'],
            'condition':c['contract']['condition'],'embedding':c['contract']['embedding'],'delta_db':c['contract']['delta_db'],'channel':c['contract']['channel'],'display_name':self.manifest['models'][n].get('display_name',n),'sha256':self.manifest['models'][n]['sha256']} for n,(_,c) in self.models.items()},
            'decision':None,'scope':'任意上传音频仅显示检测分数。','manifest_hash':canonical_hash(self.manifest)}

    def process(self,request):
        if not self.lock.acquire(blocking=False):raise RuntimeError('已有任务正在处理，请稍后再试')
        start=time.perf_counter()
        try:
            raw=base64.b64decode(request.get('audio_base64',''),validate=True)
            if not raw or len(raw)>MAX_BYTES:raise ValueError('WAV 文件为空或超过 12 MiB')
            with io.BytesIO(raw) as stream:
                info=sf.info(stream)
                if info.format not in {'WAV','WAVEX'}:raise ValueError('当前演示仅支持 WAV')
                if info.duration>30 or info.duration<.1:raise ValueError('音频时长必须在 0.1～30 秒之间')
                if info.samplerate<8000 or info.samplerate>96000:raise ValueError('采样率须在 8～96 kHz')
                stream.seek(0);audio,sr=sf.read(stream,dtype='float64',always_2d=True)
            if not np.isfinite(audio).all():raise ValueError('音频包含非有限数值')
            audio=audio.mean(axis=1)
            if abs(audio).max()<1e-10:raise ValueError('音频为静音或幅度过低')
            mode=request.get('mode','chain');detector=request.get('detector','correlation');frames=int(request.get('frames',30))
            if mode not in {'chain','detect'} or detector not in {'correlation',*self.models}:raise ValueError('未知处理模式或检测器')
            if frames not in [1,10,30]:raise ValueError('请选择 1、10 或 30 个有效帧')
            result={'input':{'sample_rate':sr,'duration_s':len(audio)/sr,'channels':info.channels},'mode':mode,'detector':detector,'aggregation_frames':frames,'audio':{}}
            with tempfile.TemporaryDirectory(prefix='speechwm-demo-') as folder:
                folder=Path(folder)
                def output(name,x,rate):
                    p=folder/(name+'.wav');write_audio(p,x,rate,subtype='PCM_16')
                    result['audio'][name]={'base64':base64.b64encode(p.read_bytes()).decode(),'sample_rate':rate,'duration_s':len(x)/rate,
                        'note':'试听和下载为 PCM16 WAV；检测使用已保存的浮点处理音频'}
                if mode=='chain':
                    audio,prep=prepare_audio(audio,sr);sr=16000;reference=audio.copy()
                    method=request.get('method','ss');delta=float(request.get('delta',3));channel=request.get('channel','opus8')
                    if method not in {'ss','lpc'} or delta not in [0,1,2,3,4,6]:raise ValueError('不支持的嵌入预设')
                    from dataclasses import asdict
                    from .spectral import SpectralConfig
                    from .lpc import LPCConfig
                    from .main_data import variants_for
                    embedding_config={'spectral':asdict(SpectralConfig()),'lpc':asdict(LPCConfig()),
                        'evaluation':{'quality_deltas_db':[0,1,2,3,4,6,8]},'audio':{'joint_peak_limit':.9}}
                    variants,gain,maximum_peak,profile=variants_for(reference,embedding_config,self.key)
                    embedded=variants[('SS_LPC' if method=='lpc' else 'SS',delta)];reference*=gain
                    metadata={'embedder':'SS_LPC' if method=='lpc' else 'SS','delta_db':delta,'sample_rate':16000,
                        'joint_gain_grid_db':[0,1,2,3,4,6,8],'maximum_grid_peak':maximum_peak,
                        'lpc':profile.metadata if method=='lpc' else None}
                    p=folder/'embedded-float.wav';write_audio(p,embedded,sr);embedded,_=read_audio(p)
                    result.update(preparation=prep,embedding={**metadata,'joint_gain':gain,'method':method,'delta_db':delta},
                        quality=quality_scores(reference,embedded,16000))
                    output('embedded',embedded,sr)
                    if channel=='none':received,rate,channel_info=embedded,16000,{'name':'none'}
                    elif channel=='downsample':received,rate,channel_info=apply_channel(embedded,16000,'downsample_8k',folder/'received-float.wav',reference,'demo',20261018)
                    elif channel in {'noise20','noise15'}:received,rate,channel_info=apply_channel(embedded,16000,channel+'_downsample_8k',folder/'received-float.wav',reference,'demo',20261018)
                    elif channel in {'opus6','opus8','opus12','opus16'}:
                        received,rate,channel_info=opus_roundtrip(resample_audio(embedded,16000,8000),8000,folder/'received-float.wav',bitrate_bps=int(channel[4:])*1000)
                    else:raise ValueError('未知通道')
                    result['channel']={k:v for k,v in channel_info.items() if k not in ['encode_command','decode_command','compressed_path','executable']}
                    output('received',received,rate);audio,sr=received,rate
                else:
                    # No reference, embedding, quality calculation or previous request state is used here.
                    if detector!='correlation' and sr!=8000:
                        if not request.get('allow_resample',False):raise ValueError('深度模型要求 8 kHz，勾选明确的降采样选项后重试')
                        audio=resample_audio(audio,sr,8000);sr=8000;result['detector_preparation']='explicit resample to 8000 Hz'
                if detector!='correlation':
                    if sr!=8000:
                        if not request.get('allow_resample',False):raise ValueError('当前接收采样率与深度模型不兼容，请选 8 kHz 通道或明确允许重采样')
                        audio=resample_audio(audio,sr,8000);sr=8000;result['detector_preparation']='explicit resample to 8000 Hz'
                    audio=audio.astype(np.float32).astype(np.float64)
                    model,checkpoint=self.models[detector];contract=checkpoint['contract']
                    from .spectral import SpectralConfig
                    f=extract_features(audio,sr,self.key,SpectralConfig(**contract['spectral']),vad=contract['vad']['mode'],vad_aggressiveness=contract['vad']['aggressiveness'])
                    scores=predict_logits(model,checkpoint['architecture'],f.features[f.valid_mask].astype(np.float32));result['model_contract']=contract
                else:
                    if sr not in [8000,16000]:audio=resample_audio(audio,sr,16000);sr=16000
                    audio=audio.astype(np.float32).astype(np.float64)
                    f=extract_features(audio,sr,self.key);scores=score_frames(f.features)[f.valid_mask]
                from .spectral import analyze,SpectralConfig
                spectrum,_=analyze(audio,sr,SpectralConfig());indices,hz=SpectralConfig().indices(sr)
                result['spectrum_trace']={'frequency_hz':hz.tolist(),'mean_magnitude_db':(20*np.log10(np.maximum(abs(spectrum[indices]).mean(axis=1),1e-12))).tolist()}
                if mode=='chain' and request.get('method')=='lpc':
                    result['lpc_trace']={'frequency_hz':hz.tolist(),'mean_weight':profile.weights.mean(axis=1).tolist(),
                        'mean_centered_envelope_db':profile.centered_band_db.mean(axis=1).tolist()}
                groups=aggregate(scores,f.times_s[f.valid_mask],frames)
                result.update(sample_rate=sr,coverage=f.metadata,groups=groups,status='scores_only' if groups else 'insufficient_voiced_frames',decision=None,
                    reason='任意上传缺少已验证的通道与数据分布假设，当前只显示分数；线性 logit 不是概率。',
                    frame_trace={'time_s':f.times_s[f.valid_mask].tolist(),'score':scores.tolist()},elapsed_s=time.perf_counter()-start)
                # Export exact float input separately, so received-only replay compares the same waveform.
                p=folder/'detector-input.wav';write_audio(p,audio,sr)
                result['audio']['detector_input']={'base64':base64.b64encode(p.read_bytes()).decode(),'sample_rate':sr,'duration_s':len(audio)/sr,'note':'检测实际使用的浮点 WAV，可独立上传复核'}
                return result
        finally:self.lock.release()


def serve(root,manifest,host='127.0.0.1',port=7860,ready_file=None):
    service=DemoService(root,manifest);token=secrets.token_urlsafe(32);html=(Path(__file__).parent/'web/index.html').read_text().replace('__API_TOKEN__',token)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,format,*args):pass
        def reply(self,status,data,content_type='application/json; charset=utf-8'):
            body=data.encode() if isinstance(data,str) else json.dumps(data,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status);self.send_header('Content-Type',content_type);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.path=='/':return self.reply(200,html,'text/html; charset=utf-8')
            if self.path=='/api/info':return self.reply(200,service.info())
            self.reply(404,{'error':'Not found'})
        def do_POST(self):
            if self.path!='/api/process':return self.reply(404,{'error':'Not found'})
            if self.headers.get('X-SpeechWM-Token')!=token:return self.reply(403,{'error':'请从当前页面发起操作'})
            if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.reply(415,{'error':'Expected JSON'})
            try:
                length=int(self.headers.get('Content-Length',0))
                if length<1 or length>MAX_BYTES*4//3+16384:return self.reply(413,{'error':'请求过大或为空'})
                result=service.process(json.loads(self.rfile.read(length)));self.reply(200,result)
            except (ValueError,RuntimeError,KeyError,OSError) as exc:self.reply(400,{'error':str(exc)})
            except Exception as exc:self.reply(500,{'error':'处理失败：'+str(exc)})
    server=ThreadingHTTPServer((host,port),Handler);address=f'http://{host}:{server.server_port}'
    if ready_file:write_json(Path(ready_file),{'url':address,'pid':os.getpid(),'manifest_hash':canonical_hash(service.manifest),'api_token':token})
    print('SpeechWM local demo: '+address,flush=True)
    try:server.serve_forever()
    finally:server.server_close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--manifest',type=Path,default=Path('deployment/models.json'))
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=7860);p.add_argument('--ready-file',type=Path)
    a=p.parse_args();serve(a.root,a.root/a.manifest,a.host,a.port,a.ready_file)
if __name__=='__main__':main()
