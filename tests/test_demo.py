import base64,io,json
import numpy as np
import pytest
import soundfile as sf
from speechwm.demo import DemoService
from speechwm.keys import save_key
from speechwm.utils import sha256_file

@pytest.fixture
def service(tmp_path):
    save_key(tmp_path/'key.json')
    (tmp_path/'models.json').write_text(json.dumps({'key':{'path':'key.json','sha256':sha256_file(tmp_path/'key.json')},'models':{}}))
    return DemoService(tmp_path,tmp_path/'models.json')

def payload(duration=1,sr=8000,silent=False,channels=1):
    x=.1*np.sin(2*np.pi*170*np.arange(round(duration*sr))/sr)+np.random.default_rng(6).normal(0,.01,round(duration*sr))
    if silent:x*=0
    if channels==2:x=np.column_stack([x,x*.7])
    s=io.BytesIO();sf.write(s,x,sr,format='WAV',subtype='FLOAT')
    return {'audio_base64':base64.b64encode(s.getvalue()).decode(),'mode':'detect','detector':'correlation','frames':30}

@pytest.mark.parametrize('problem',['silence','short','long','format','frames','detector'])
def test_invalid_uploads_are_rejected_without_leaking_busy_state(service,problem):
    q=payload()
    if problem=='silence':q=payload(silent=True)
    if problem=='short':q=payload(.05)
    if problem=='long':q=payload(30.01)
    if problem=='format':q['audio_base64']=base64.b64encode(b'not audio').decode()
    if problem=='frames':q['frames']=7
    if problem=='detector':q['detector']='missing'
    with pytest.raises((ValueError,RuntimeError)):service.process(q)
    assert service.lock.acquire(False);service.lock.release()

def test_independent_detect_needs_no_original_and_keeps_decision_null(service):
    result=service.process(payload(channels=2))
    assert result['decision'] is None and 'quality' not in result and result['input']['channels']==2
    assert result['status'] in {'scores_only','insufficient_voiced_frames'}
    again=service.process({**payload(),'audio_base64':result['audio']['detector_input']['base64']})
    assert result['groups']==again['groups']

def test_single_job_limit_and_unsupported_embed_presets(service):
    service.lock.acquire()
    with pytest.raises(RuntimeError,match='正在处理'):service.process(payload())
    service.lock.release()
    with pytest.raises(ValueError):service.process({**payload(),'mode':'chain','delta':9})

def test_chain_includes_reference_quality_and_replayable_received_audio(service):
    q={**payload(2,16000),'mode':'chain','method':'lpc','delta':6,'channel':'opus6'}
    result=service.process(q)
    assert result['quality']['pesq_nb'] is not None and result['channel']['inspection']['payload_bitrate_bps']==6000
    assert len(result['spectrum_trace']['frequency_hz'])==len(result['lpc_trace']['mean_weight'])==len(result['lpc_trace']['mean_centered_envelope_db'])==63
    again=service.process({**payload(),'audio_base64':result['audio']['detector_input']['base64']})
    assert again['groups']==result['groups'] and 'quality' not in again

def test_short_non_silent_chain_preserves_pesq_failure_instead_of_fake_zero(service):
    result=service.process({**payload(.15,16000),'mode':'chain','method':'ss','delta':1,'channel':'downsample'})
    assert result['quality']['pesq_nb'] is None and result['quality']['quality_failures']
    assert not result['groups'] and result['status']=='insufficient_voiced_frames'
