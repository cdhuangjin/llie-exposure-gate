"""Archived global/local exposure gate; CLI explicitly selects global mode."""
import cv2
import numpy as np

def fuse(raw, enhanced, threshold=.18, temperature=.04, sigma=7., global_map=False):
    """Return fused float32 RGB and alpha; alpha=1 keeps enhancement."""
    raw=np.asarray(raw,dtype=np.float32);enhanced=np.asarray(enhanced,dtype=np.float32)
    if raw.shape!=enhanced.shape or raw.ndim!=3 or raw.shape[2]!=3:
        raise ValueError('Expected matching HxWx3 RGB arrays; no automatic resize')
    if not np.isfinite(raw).all() or not np.isfinite(enhanced).all():
        raise ValueError('Images must be finite')
    if raw.min()<0 or raw.max()>1:
        raise ValueError('Raw RGB must be scaled to [0,1]')
    if not(0<=threshold<=1 and 0<temperature<=1 and 0<sigma<10000):
        raise ValueError('Invalid gate parameters')
    luminance=raw @ np.array([.2126,.7152,.0722],dtype=np.float32)
    mu=np.full_like(luminance,luminance.mean()) if global_map else cv2.GaussianBlur(
        luminance,(0,0),sigmaX=sigma,sigmaY=sigma,borderType=cv2.BORDER_REFLECT_101)
    alpha=np.float32(1)/(np.float32(1)+np.exp(np.clip((mu-threshold)/temperature,-60,60)))
    enhanced=np.clip(enhanced,0,1)
    fused=alpha[...,None]*enhanced+(np.float32(1)-alpha[...,None])*raw
    return np.clip(fused,0,1).astype(np.float32),alpha.astype(np.float32)
