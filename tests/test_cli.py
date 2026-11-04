import json
import subprocess
import sys

import numpy as np

from speechwm.audio import write_audio
from speechwm.keys import save_key


def run_cli(root,*arguments):
    return subprocess.run([sys.executable,"-m","speechwm.cli","--root",str(root),*arguments],
                          check=True,capture_output=True,text=True)


def test_independent_detection_after_original_removed(tmp_path):
    sr=16000
    t=np.arange(sr*2)/sr
    # An artificial signal is used only to check file/interface independence.
    signal=.15*np.sin(2*np.pi*180*t)+.04*np.sin(2*np.pi*900*t)
    source=tmp_path/"original.wav"
    received=tmp_path/"received.wav"
    key=tmp_path/"key.json"
    write_audio(source,signal,sr)
    save_key(key)
    run_cli(tmp_path,"embed",str(source),str(received),"--key",str(key),"--delta","1")
    source.unlink()
    record=json.loads(run_cli(tmp_path,"detect",str(received),"--key",str(key)).stdout)
    assert record["decision"] is None
    assert record["sample_rate"]==16000
    assert record["status"] in {"scores_only","insufficient_voiced_frames"}


def test_embed_refuses_to_overwrite_source(tmp_path):
    path=tmp_path/"source.wav"
    write_audio(path,np.random.default_rng(4).normal(0,.03,16000),16000)
    before=path.read_bytes()
    key=tmp_path/"key.json"
    save_key(key)
    result=subprocess.run([sys.executable,"-m","speechwm.cli","--root",str(tmp_path),
                           "embed",str(path),str(path),"--key",str(key)],capture_output=True)
    assert result.returncode!=0
    assert path.read_bytes()==before

