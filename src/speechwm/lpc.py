"""Framewise autocorrelation LPC and the paper's Gaussian survival weighting."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import get_window
from scipy.special import ndtr

from .audio import validate_audio
from .spectral import SpectralConfig


@dataclass(frozen=True)
class LPCConfig:
    order: int = 16
    alpha: float = .35
    gaussian_sigma_db: float = 13.5
    ridge_relative: float = 1e-6
    energy_floor: float = 1e-12
    denominator_power_floor: float = 1e-12
    centering: str = "watermark_band_per_frame"

    def validate(self, window: int) -> None:
        if not isinstance(self.order,int) or not 1 <= self.order < window:
            raise ValueError("LPC order must be an integer between 1 and frame length-1")
        values=[self.alpha,self.gaussian_sigma_db,self.ridge_relative,
                self.energy_floor,self.denominator_power_floor]
        if not np.isfinite(values).all() or self.alpha < 0 or any(v <= 0 for v in values[1:]):
            raise ValueError("Invalid LPC weight or stability parameters")
        if self.centering != "watermark_band_per_frame":
            raise ValueError("Only the explicitly fixed per-frame band centering is implemented")


@dataclass
class LPCResult:
    weights: np.ndarray                 # [63, T]
    envelope_db: np.ndarray             # [N/2+1, T], 10log10(error_power/|A|^2)
    centered_band_db: np.ndarray        # [63, T]
    coefficients: np.ndarray            # [T, order+1], A(z)=1+sum(a_j z^-j)
    times_s: np.ndarray
    frame_power: np.ndarray
    status: np.ndarray
    metadata: dict


def gaussian_weights(centered_db: np.ndarray, alpha: float = .35, sigma_db: float = 13.5) -> np.ndarray:
    x=np.asarray(centered_db,dtype=float)
    if not np.isfinite(x).all() or not np.isfinite([alpha,sigma_db]).all() or alpha<0 or sigma_db<=0:
        raise ValueError("Invalid Gaussian weight inputs")
    # ndtr(-x/sigma) avoids cancellation in 1-ndtr(x/sigma) in the upper tail.
    return ndtr(-x/sigma_db)**alpha


def estimate_frame(frame: np.ndarray, config: LPCConfig) -> tuple[np.ndarray,float,str]:
    """Windowed, mean-removed frame -> LPC denominator and residual variance."""
    frame=np.asarray(frame,dtype=float)
    config.validate(len(frame))
    if frame.ndim!=1 or not np.isfinite(frame).all():
        raise ValueError("Expected finite one-dimensional LPC frame")
    a=np.r_[1.,np.zeros(config.order)]
    r=np.array([np.dot(frame[k:],frame[:len(frame)-k]) for k in range(config.order+1)])
    if r[0]/len(frame) <= config.energy_floor:
        return a,0.,"low_energy"
    normalized=r/r[0]
    first_column=normalized[:config.order].copy()
    first_column[0]+=config.ridge_relative
    try:
        a[1:]=-solve_toeplitz(first_column,normalized[1:])
        error=float((r[0]+np.dot(a[1:],r[1:]))/len(frame))
        if not np.isfinite(a).all() or error<=0 or not np.isfinite(error):
            return np.r_[1.,np.zeros(config.order)],0.,"invalid_solution"
        if np.max(np.abs(np.roots(a))) >= 1:
            return np.r_[1.,np.zeros(config.order)],0.,"unstable_poles"
        return a,error,"ok"
    except (np.linalg.LinAlgError,ValueError,FloatingPointError):
        return np.r_[1.,np.zeros(config.order)],0.,"solver_failure"


def analyze_lpc(audio: np.ndarray, sr: int, spectral: SpectralConfig | None = None,
                config: LPCConfig | None = None) -> LPCResult:
    cfg=spectral or SpectralConfig()
    lp=config or LPCConfig()
    audio=validate_audio(audio,sr)
    window,hop=cfg.lengths(sr)
    lp.validate(window)
    if len(audio)<window:
        raise ValueError("Audio is shorter than an LPC analysis frame")
    indices,_=cfg.indices(sr)
    padded=np.pad(audio,(window//2,window//2+(-len(audio))%hop))
    frames=np.lib.stride_tricks.sliding_window_view(padded,window)[::hop].copy()
    frames-=frames.mean(axis=1,keepdims=True)
    frames*=get_window("hann",window,fftbins=True)
    count=len(frames)
    coeff=np.zeros((count,lp.order+1));coeff[:,0]=1
    envelope=np.zeros((window//2+1,count))
    centered=np.zeros((len(indices),count));weights=np.zeros_like(centered)
    powers=np.mean(frames**2,axis=1)
    statuses=[]
    for i,frame in enumerate(frames):
        a,error,status=estimate_frame(frame,lp)
        if status=="ok":
            denom=np.abs(np.fft.rfft(a,n=window))**2
            if not np.isfinite(denom).all() or denom.min()<=lp.denominator_power_floor:
                status="small_denominator"
            else:
                envelope[:,i]=10*np.log10(error/denom)
                centered[:,i]=envelope[indices,i]-envelope[indices,i].mean()
                weights[:,i]=gaussian_weights(centered[:,i],lp.alpha,lp.gaussian_sigma_db)
                coeff[i]=a
        statuses.append(status)
    status_array=np.array(statuses,dtype="U24")
    ok=status_array=="ok"
    metadata={"config":asdict(lp),"frames":count,"valid_lpc_frames":int(ok.sum()),
              "low_energy_frames":np.flatnonzero(status_array=="low_energy").tolist(),
              "numerical_failure_frames":np.flatnonzero(~ok & (status_array!="low_energy")).tolist(),
              "status_counts":{s:statuses.count(s) for s in sorted(set(statuses))},
              "lpc_input":"same centered/padded frames as STFT; remove frame mean then periodic Hann; no pre-emphasis",
              "envelope":"10log10(residual_variance/abs(rfft(A))^2)",
              "zero_weight_policy":"low-energy or invalid frames receive zero weights, recorded by frame index"}
    return LPCResult(weights,envelope,centered,coeff,np.arange(count)*hop/sr,powers,status_array,metadata)
