"""Paper A/B architectures and strict, reference-free checkpoint inference."""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .features import extract_features
from .keys import key_hash
from .spectral import SpectralConfig
from .utils import canonical_hash

FEATURE_VERSION="matched-log-spectrum-63-v1"
PARAMETERS={"A":6209,"B":15713}


class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(63,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))

    def forward(self,x):
        if x.ndim!=2 or x.shape[1]!=63:raise ValueError("A expects [N,63]")
        return self.layers(x)


class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv1d(1,16,3,stride=1,padding=0)
        self.layers=nn.Sequential(nn.Flatten(),nn.Linear(976,16),nn.ReLU(),nn.Linear(16,1))

    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(1,63):raise ValueError("B expects [N,1,63]")
        return self.layers(self.conv(x))  # No extra activation after convolution.


def build_model(architecture):
    if architecture not in PARAMETERS:raise ValueError("Unknown model architecture")
    model={"A":ModelA,"B":ModelB}[architecture]()
    assert sum(p.numel() for p in model.parameters())==PARAMETERS[architecture]
    return model


def model_input(x,architecture):
    return x.unsqueeze(1) if architecture=="B" else x


def state_digest(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        value=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def make_contract(key,config,condition):
    cfg=SpectralConfig(**config["spectral"])
    return {"schema_version":1,"feature_version":FEATURE_VERSION,"key_sha256":key_hash(key),
        "spectral":asdict(cfg),"frequency_hz":cfg.indices(8000)[1].tolist(),"sample_rate":8000,
        "vad":config["evaluation"]["vad"],"normalization":"none","model_dtype":"float32",
        "embedding":"SS","condition":condition["id"],"delta_db":condition["delta_db"],
        "channel":condition["channel"],"aggregation":"mean of linear logits on fixed valid frames",
        "aggregate_frames":config["evaluation"]["aggregate_frames"],
        "scope":"development smoke model; no production decision threshold"}


def load_checkpoint(path: Path,key=None):
    checkpoint=torch.load(path,map_location="cpu",weights_only=True)
    torch.set_num_threads(checkpoint["training_config"]["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    contract=checkpoint["contract"]
    if checkpoint["schema_version"]!=1 or contract["schema_version"]!=1:raise ValueError("Unsupported checkpoint schema")
    if contract["embedding"] not in {"SS","SS_LPC"} or contract["model_dtype"]!="float32":raise ValueError("Unsupported deep embedding/dtype contract")
    if canonical_hash(contract)!=checkpoint["contract_hash"]:raise ValueError("Checkpoint contract hash mismatch")
    if contract["feature_version"]!=FEATURE_VERSION or contract["normalization"]!="none":
        raise ValueError("Unsupported feature contract")
    cfg=SpectralConfig(**contract["spectral"])
    if contract["sample_rate"]!=8000 or contract["frequency_hz"]!=cfg.indices(8000)[1].tolist():
        raise ValueError("Checkpoint frequency/rate mismatch")
    if key is not None and key_hash(key)!=contract["key_sha256"]:raise ValueError("Checkpoint watermark key mismatch")
    if state_digest(checkpoint["state_dict"])!=checkpoint["state_sha256"]:raise ValueError("Checkpoint tensor hash mismatch")
    model=build_model(checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"],strict=True);model.eval()
    return model,checkpoint


def predict_logits(model,architecture,x,batch_size=1024):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=63 or not np.isfinite(x).all():raise ValueError("Expected finite [N,63] features")
    model.eval();outputs=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch_size):
            tensor=torch.from_numpy(np.ascontiguousarray(x[start:start+batch_size]))
            outputs.append(model(model_input(tensor,architecture)).squeeze(1).numpy().astype(np.float64))
    values=np.concatenate(outputs) if outputs else np.empty(0)
    if not np.isfinite(values).all():raise ValueError("Non-finite model logits")
    return values


def infer_audio(path,audio,sr,key):
    model,checkpoint=load_checkpoint(path,key)
    contract=checkpoint["contract"]
    if sr!=contract["sample_rate"]:raise ValueError("Deep checkpoint requires received 8 kHz WAV")
    f=extract_features(audio,sr,key,SpectralConfig(**contract["spectral"]),
        vad=contract["vad"]["mode"],vad_aggressiveness=contract["vad"]["aggressiveness"])
    scores=predict_logits(model,checkpoint["architecture"],f.features[f.valid_mask])
    return scores,f,checkpoint
