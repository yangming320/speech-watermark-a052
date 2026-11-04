import json
import subprocess
import sys

import numpy as np
import pytest
from scipy.linalg import toeplitz
from scipy.signal import lfilter

from speechwm.audio import read_audio, write_audio
from speechwm.embedding import embed, embed_lpc, modify_spectrum
from speechwm.keys import make_key, save_key
from speechwm.lpc import LPCConfig, analyze_lpc, estimate_frame, gaussian_weights
from speechwm.spectral import SpectralConfig, analyze
from speechwm.comparison import choose_quality_pair


def test_quality_selection_uses_actual_global_points_and_tolerance():
    rows=[{"method":m,"delta_db":d,"pesq_nb":q} for m,d,q in
          [("SS",1,4.5),("SS",3,4.416),("SS_LPC",4,4.47),("SS_LPC",6,4.411)]]
    p=choose_quality_pair(rows,4.4,.02)
    assert (p["ss_delta_db"],p["lpc_delta_db"])==(3,6)
    assert p["absolute_mean_difference"]==pytest.approx(.005)
    assert not p["interpolated"] and not p["per_utterance_strength_selection"]


def test_quality_selection_refuses_to_extrapolate_or_fake_a_match():
    rows=[{"method":"SS","delta_db":3,"pesq_nb":4.4},
          {"method":"SS_LPC","delta_db":6,"pesq_nb":4.45}]
    with pytest.raises(ValueError,match="within tolerance"):choose_quality_pair(rows,4.4,.02)


def test_gaussian_formula_monotonic_extremes_and_fixed_scale():
    x=np.array([-1e6,-13.5,0,13.5,1e6])
    w=gaussian_weights(x)
    assert w[0]==1 and w[-1]==0 and w[2]==pytest.approx(.5**.35)
    assert np.all(np.diff(w)<=0) and np.all((w>=0)&(w<=1))
    np.testing.assert_array_equal(gaussian_weights(x,0),np.ones(5))


@pytest.mark.parametrize("config",[LPCConfig(order=0),LPCConfig(order=320),LPCConfig(alpha=-1),
                                  LPCConfig(gaussian_sigma_db=0),LPCConfig(ridge_relative=0),
                                  LPCConfig(centering="global")])
def test_bad_lpc_config_rejected(config):
    with pytest.raises(ValueError):analyze_lpc(np.ones(16000),16000,config=config)


def test_ar_poles_sign_envelope_and_yule_walker_solution():
    r=.9;theta=2*np.pi*1000/16000
    truth=np.array([1,-2*r*np.cos(theta),r*r])
    frame=lfilter([1],truth,np.random.default_rng(44).normal(0,.1,80000))[1000:]
    cfg=LPCConfig(order=2)
    a,error,status=estimate_frame(frame,cfg)
    assert status=="ok" and error>0
    np.testing.assert_allclose(a,truth,atol=.012)
    corr=np.correlate(frame[:1000],frame[:1000],mode="full")[999:1002]
    aa,_,status=estimate_frame(frame[:1000],cfg)
    dense=toeplitz(corr[:2]/corr[0])+np.eye(2)*cfg.ridge_relative
    np.testing.assert_allclose(dense @ (-aa[1:]),corr[1:]/corr[0],atol=1e-12)
    assert np.max(np.abs(np.roots(a)))<1
    freq=np.fft.rfftfreq(320,1/16000)
    peak=freq[np.argmax(1/np.abs(np.fft.rfft(a,n=320)))]
    assert abs(peak-1000)<=100


def test_frame_alignment_centering_and_gain_invariance():
    x=np.random.default_rng(2).normal(0,.04,16013)
    profile=analyze_lpc(x,16000)
    scaled=analyze_lpc(x*.5,16000)
    spec,times=analyze(x,16000,SpectralConfig())
    assert profile.weights.shape==(63,spec.shape[1])
    np.testing.assert_array_equal(np.rint(profile.times_s*16000),np.rint(times*16000))
    np.testing.assert_allclose(profile.centered_band_db.mean(axis=0),0,atol=1e-12)
    np.testing.assert_allclose(profile.weights,scaled.weights,atol=1e-12)
    assert not profile.metadata["numerical_failure_frames"]
    for i in np.flatnonzero(profile.status=="ok"):
        order=np.argsort(profile.centered_band_db[:,i])
        assert np.all(np.diff(profile.weights[order,i])<=0)


def test_silence_zero_weights_and_small_denominator_is_recorded():
    p=analyze_lpc(np.zeros(16000),16000)
    assert not p.weights.any() and set(p.status)=={"low_energy"}
    p=analyze_lpc(np.random.default_rng(0).normal(0,.04,16000),16000,
                  config=LPCConfig(denominator_power_floor=100))
    assert p.metadata["numerical_failure_frames"] and not p.weights.any()
    y,_=embed_lpc(np.zeros(16000),16000,make_key(),4)
    np.testing.assert_array_equal(y,0)


def test_weighted_spectrum_matches_formula_and_ss_ablation():
    cfg=SpectralConfig();key=make_key();rng=np.random.default_rng(1)
    s=rng.normal(size=(161,8))+1j*rng.normal(size=(161,8))
    weights=np.broadcast_to(np.linspace(0,1,63)[:,None],(63,8)).copy()
    actual=modify_spectrum(s,16000,key,3,cfg,weights)
    indices,_=cfg.indices(16000)
    np.testing.assert_allclose(20*np.log10(abs(actual[indices]/s[indices])),3*key[:,None]*weights,atol=1e-12)
    np.testing.assert_allclose(np.angle(actual/s),0,atol=1e-12)
    outside=np.setdiff1d(np.arange(161),indices)
    np.testing.assert_array_equal(actual[outside],s[outside])
    np.testing.assert_array_equal(modify_spectrum(s,16000,key,3,cfg,np.ones((63,8))),modify_spectrum(s,16000,key,3,cfg))
    with pytest.raises(ValueError):modify_spectrum(s,16000,key,3,cfg,np.full((63,8),1.01))


def test_lpc_zero_strength_same_as_original_ss_and_actual_reload(tmp_path):
    x=np.random.default_rng(9).normal(0,.04,16013)
    a,_=embed(x,16000,make_key(),0)
    b,_=embed_lpc(x,16000,make_key(),0)
    np.testing.assert_array_equal(a,b)
    path=tmp_path/"lpc.wav"
    y,metadata=embed_lpc(x,16000,make_key(),3)
    write_audio(path,y,16000)
    saved,_=read_audio(path)
    np.testing.assert_allclose(saved,y,atol=2e-8)
    assert metadata["embedder"]=="SS_LPC" and not metadata["lpc"]["numerical_failure_frames"]


def test_lpc_nan_short_audio_rejected():
    with pytest.raises(ValueError):analyze_lpc(np.full(16000,np.nan),16000)
    with pytest.raises(ValueError):analyze_lpc(np.ones(12),16000)
    with pytest.raises(ValueError):gaussian_weights(np.array([np.nan]))


def test_cli_lpc_embedding_then_independent_detect(tmp_path):
    t=np.arange(32000)/16000;x=.1*np.sin(2*np.pi*180*t)+.04*np.sin(2*np.pi*900*t)
    source=tmp_path/"source.wav";received=tmp_path/"lpc.wav";key=tmp_path/"key.json"
    write_audio(source,x,16000);save_key(key)
    def cli(*args):return subprocess.run([sys.executable,"-m","speechwm.cli","--root",str(tmp_path),*map(str,args)],check=True,capture_output=True,text=True)
    result=json.loads(cli("embed",source,received,"--key",key,"--delta","3","--method","lpc").stdout)
    assert result["embedder"]=="SS_LPC"
    source.unlink()
    result=json.loads(cli("detect",received,"--key",key).stdout)
    assert result["sample_rate"]==16000 and result["decision"] is None
