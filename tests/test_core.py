import json

import numpy as np
import pytest

from speechwm.audio import read_audio, resample_audio, write_audio
from speechwm.embedding import embed, modify_spectrum
from speechwm.features import aggregate, extract_features
from speechwm.keys import load_key, make_key, save_key
from speechwm.metrics import equal_error_rate
from speechwm.spectral import SpectralConfig, analyze, cepstral_filter, synthesize


@pytest.mark.parametrize("sr",[8000,16000])
def test_stft_roundtrip_including_boundaries(sr):
    cfg = SpectralConfig()
    x = np.random.default_rng(4).normal(0,.05,12347)
    x[0],x[-1] = .4,-.4
    spectrum,_ = analyze(x,sr,cfg)
    y = synthesize(spectrum,sr,len(x),cfg)
    assert len(y)==len(x)
    assert np.max(np.abs(y-x)) < 1e-12


def test_same_physical_watermark_band_after_resampling():
    cfg=SpectralConfig()
    _,f16=cfg.indices(16000)
    _,f8=cfg.indices(8000)
    np.testing.assert_array_equal(f16, np.arange(500,3650,50))
    np.testing.assert_array_equal(f8, f16)


def test_embedding_db_change_phase_and_unchanged_outside_band():
    cfg=SpectralConfig()
    rng=np.random.default_rng(1)
    spectrum=rng.normal(size=(161,7))+1j*rng.normal(size=(161,7))
    key=make_key()
    result=modify_spectrum(spectrum,16000,key,3,cfg)
    indices,_=cfg.indices(16000)
    observed=20*np.log10(np.abs(result[indices])/np.abs(spectrum[indices]))
    np.testing.assert_allclose(observed,np.broadcast_to(3*key[:,None],observed.shape),atol=1e-12)
    np.testing.assert_allclose(np.angle(result/spectrum),0,atol=1e-12)
    outside=np.setdiff1d(np.arange(161),indices)
    np.testing.assert_array_equal(result[outside],spectrum[outside])


def test_zero_strength_roundtrip_on_disk(tmp_path):
    x=np.random.default_rng(8).normal(0,.05,16001)
    y,_=embed(x,16000,make_key(),0)
    path=tmp_path/"roundtrip.wav"
    write_audio(path,y,16000)
    read,sr=read_audio(path)
    assert sr==16000
    assert np.max(np.abs(read-x)) < 2e-8


def test_key_hash_detects_corruption(tmp_path):
    path=tmp_path/"key.json"
    save_key(path)
    np.testing.assert_array_equal(load_key(path),make_key())
    record=json.loads(path.read_text())
    record["values"][0]*=-1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError,match="hash mismatch"):
        load_key(path)


@pytest.mark.parametrize("labels,scores,expected",[
    ([0,0,1,1],[0,1,2,3],0),
    ([0,1,0,1],[1,1,1,1],.5),
    ([0,0,1,1],[0,1,1,2],.25),
    ([0,0,1,1],[3,2,1,0],1),
])
def test_eer_ties_separation_and_score_direction(labels,scores,expected):
    assert equal_error_rate(np.array(labels),np.array(scores))["eer"]==pytest.approx(expected)


def test_aggregation_does_not_repeat_short_remainder():
    scores=np.arange(10,dtype=float)
    groups=aggregate(scores,scores*.01,3)
    assert len(groups)==3
    assert [g["score"] for g in groups]==[1,4,7]
    assert aggregate(scores[:2],scores[:2],3)==[]


def test_resampling_keeps_physical_tone_frequency():
    x=.1*np.sin(2*np.pi*2000*np.arange(16000)/16000)
    y=resample_audio(x,16000,8000)
    assert len(y)==8000
    frequency=np.fft.rfftfreq(len(y),1/8000)[np.argmax(np.abs(np.fft.rfft(y)))]
    assert frequency==2000


def test_cepstral_filter_removes_constant_log_level():
    cfg=SpectralConfig()
    filtered=cepstral_filter(np.full((161,4),-40.),16000,cfg)
    np.testing.assert_allclose(filtered,0,atol=1e-12)


def test_silence_has_no_voiced_frames_and_shape_is_stable():
    result=extract_features(np.zeros(16000),16000,make_key())
    assert result.features.shape[1]==63
    assert not result.valid_mask.any()
    assert np.isfinite(result.features).all()

