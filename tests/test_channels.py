from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.signal import butter, correlate, correlation_lags, sosfilt

from speechwm.audio import read_audio, resample_audio, write_audio
from speechwm.channels import apply_channel, codec_runtime, inspect_opus, opus_roundtrip, paired_noise, snr_db
from speechwm.embedding import embed
from speechwm.keys import save_key, load_key


def signal(sr=16000, samples=None):
    n=samples or sr*2
    t=np.arange(n)/sr
    return .12*np.sin(2*np.pi*173*t)+.06*np.sin(2*np.pi*937*t)+np.random.default_rng(5).normal(0,.01,n)


@pytest.mark.parametrize("snr",[15,20])
def test_exact_noise_power_reference_and_determinism(snr):
    x=signal()
    n,a=paired_noise(x,"source",20261017,snr)
    n2,b=paired_noise(x,"source",20261017,snr)
    np.testing.assert_array_equal(n,n2)
    assert a==b and abs(snr_db(x,n)-snr)<1e-10
    assert a["reference_voiced_samples"]>0
    assert a["actual_reference_snr_post_db"]>snr+1
    other,_=paired_noise(x,"other-source",20261017,snr)
    assert not np.array_equal(n,other)


def test_noise_shared_across_snr_and_scales_with_common_gain():
    x=signal()
    a,ma=paired_noise(x,"a",4,20)
    b,mb=paired_noise(x,"a",4,15)
    c,mc=paired_noise(x*.7,"a",4,20)
    assert ma["standard_noise_sha256"]==mb["standard_noise_sha256"]==mc["standard_noise_sha256"]
    np.testing.assert_allclose(b,a*10**(.25),atol=1e-16)
    np.testing.assert_allclose(c,a*.7,atol=1e-16)


@pytest.mark.parametrize("x,snr",[(np.zeros(16000),20),(np.ones(16000),20),(signal(),np.nan)])
def test_invalid_noise_reference_fails(x,snr):
    with pytest.raises(ValueError):
        paired_noise(x,"a",0,snr)


def test_noise_added_before_antialias_and_both_labels_share_it(tmp_path):
    x=signal(samples=32013)
    positive=x+.004*np.sin(2*np.pi*800*np.arange(len(x))/16000)
    a,sr,ma=apply_channel(x,16000,"noise15_downsample_8k",tmp_path/"a.wav",x,"same",7)
    b,_,mb=apply_channel(positive,16000,"noise15_downsample_8k",tmp_path/"b.wav",x,"same",7)
    noise,_=paired_noise(x,"same",7,15)
    np.testing.assert_allclose(a,resample_audio(x+noise,16000,8000),atol=2e-8)
    np.testing.assert_allclose(b-a,resample_audio(positive-x,16000,8000),atol=3e-8)
    assert ma["noise_sha256"]==mb["noise_sha256"]
    assert sr==8000 and len(a)==16007
    with pytest.raises(FileExistsError):
        apply_channel(x,16000,"noise15_downsample_8k",tmp_path/"a.wav",x,"same",7)


def test_downsample_rejects_alias_not_header_only(tmp_path):
    t=np.arange(16000)/16000
    high=.1*np.sin(2*np.pi*6000*t)
    out,sr,_=apply_channel(high,16000,"downsample_8k",tmp_path/"a.wav",high,"tone",0)
    assert sr==8000 and len(out)==8000
    assert np.sqrt(np.mean(out[100:-100]**2))<1e-4


@pytest.mark.parametrize("channel",["clean","downsample_8k","noise20_downsample_8k",
                                    "noise15_downsample_8k","opus8k_8kbps"])
def test_zero_strength_both_labels_have_identical_decoded_pcm(tmp_path,channel):
    x=signal()
    a,sra,_=apply_channel(x,16000,channel,tmp_path/"label0.wav",x,"same",2)
    b,srb,_=apply_channel(x,16000,channel,tmp_path/"label1.wav",x,"same",2)
    assert sra==srb
    np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize("samples",[8000,8007,16001,16960])
def test_real_opus_rate_length_payload_and_repeatability(tmp_path,samples):
    x=signal(8000,samples)
    a,sr,meta=opus_roundtrip(x,8000,tmp_path/"a.wav")
    b,_,again=opus_roundtrip(x,8000,tmp_path/"b.wav")
    assert sr==8000 and len(a)==samples
    np.testing.assert_array_equal(a,b)
    i=inspect_opus(tmp_path/"a.opus")
    assert i==meta["inspection"]
    assert i["payload_sha256"]==again["inspection"]["payload_sha256"]
    assert i["packet_duration_ms"]==[20] and i["packet_sizes_bytes"]==[20]
    assert i["payload_bitrate_bps"]==8000
    assert i["bandwidth_counts"].keys()=={"NB"}
    assert i["container_pcm_duration_s"]==samples/8000
    assert i["pre_skip_ms"]==6.5
    assert not np.allclose(a,x)  # actual lossy codec, not a renamed WAV
    assert meta["encode_returncode"]==meta["decode_returncode"]==0
    with pytest.raises(FileExistsError):
        opus_roundtrip(x,8000,tmp_path/"a.wav")


def test_opus_timing_diagnostic_without_using_lag_to_align(tmp_path):
    x=np.zeros(24000)
    # Tonal signals have periodic correlation ambiguities after lossy phase changes.
    x[4000:20000]=sosfilt(butter(3,2800,fs=8000,output="sos"),
        np.random.default_rng(42).normal(0,.1,16000))
    a,_,_=opus_roundtrip(x,8000,tmp_path/"latency.wav")
    corr=correlate(a,x,method="fft")
    lags=correlation_lags(len(a),len(x))
    lag=int(lags[np.argmax(corr)])
    assert abs(lag)<=1  # diagnostic only; production never correlates or shifts


def test_opus_silence_keeps_packet_cadence(tmp_path):
    y,_,meta=opus_roundtrip(np.zeros(8000),8000,tmp_path/"silence.wav")
    assert len(y)==8000
    assert meta["inspection"]["packet_sizes_bytes"]==[20]
    assert meta["inspection"]["audio_packets"]==51


@pytest.mark.parametrize("mutation",["truncate","crc"])
def test_corrupt_opus_fails_validation(tmp_path,mutation):
    opus_roundtrip(signal(8000,8007),8000,tmp_path/"a.wav")
    data=bytearray((tmp_path/"a.opus").read_bytes())
    if mutation=="truncate":
        data=data[:-5]
    else:
        data[-10]^=1
    (tmp_path/"bad.opus").write_bytes(data)
    with pytest.raises(ValueError):
        inspect_opus(tmp_path/"bad.opus")


def test_codec_is_packaged_and_wrong_rate_fails(tmp_path):
    runtime=codec_runtime()
    assert "imageio_ffmpeg/binaries" in runtime["executable"]
    assert len(runtime["executable_sha256"])==64
    with pytest.raises(ValueError):
        opus_roundtrip(signal(),16000,tmp_path/"a.wav")
    with pytest.raises(ValueError):
        apply_channel(signal(),16000,"fake",tmp_path/"a.wav",signal(),"a",0)


def test_cli_received_opus_detection_after_removing_source(tmp_path):
    x=signal()
    source=tmp_path/"source.wav"
    write_audio(source,x,16000)
    key_path=tmp_path/"key.json"
    save_key(key_path,20261017)
    marked,_=embed(x,16000,load_key(key_path),3)
    write_audio(tmp_path/"marked.wav",marked,16000)
    env={**os.environ,"PYTHONPATH":str(__import__('pathlib').Path(__file__).resolve().parents[1]/"src")}
    def cli(*args):
        return subprocess.run([sys.executable,"-m","speechwm.cli","--root",str(tmp_path),*map(str,args)],
                              check=True,capture_output=True,text=True,env=env)
    cli("channel",tmp_path/"marked.wav",tmp_path/"received.wav","--name","opus8k_8kbps")
    source.unlink()
    (tmp_path/"marked.wav").unlink()
    (tmp_path/"received.codec-input.wav").unlink()
    result=json.loads(cli("detect",tmp_path/"received.wav","--key",key_path).stdout)
    assert result["sample_rate"]==8000 and result["groups"]
    assert result["decision"] is None
