import numpy as np
# import cv2
from dataclasses import dataclass
from scipy.signal import stft
try:
    import cv2
except ImportError:
    cv2 = None
except OSError as e:
    # Esto captura específicamente el error de libGL.so.1 faltante
    import types
    print("⚠️ OpenCV desactivado (no se encontró libGL). Algunas funciones de espectrogramas se omitirán.")
    cv2 = types.SimpleNamespace(
        imread=lambda *a, **kw: None,
        resize=lambda *a, **kw: None,
        imwrite=lambda *a, **kw: None,
        cvtColor=lambda *a, **kw: None
    )


@dataclass
class SpecCfg:
    nperseg: int = 128
    noverlap: int = 64
    nfft: int | None = None
    window: str = "hann"
    fmax: float | None = 60.0
    eps: float = 1e-6
    out_size: int = 224
    normalize: str = "zscore_then_minmax"   # "zscore_then_minmax" | "minmax"
    clip_percentiles: tuple[float, float] = (1.0, 99.0)

def _normalize_img(img: np.ndarray, eps: float, mode: str, clip_percentiles=(0.0, 100.0)) -> np.ndarray:
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = clip_percentiles
    if (lo, hi) != (0.0, 100.0):
        finite = np.isfinite(img)
        if finite.any():
            vlo, vhi = np.percentile(img[finite], [lo, hi])
            if vhi > vlo:
                img = np.clip(img, vlo, vhi)
    if mode == "zscore_then_minmax":
        std = img.std()
        img = (img - img.mean()) / (std + eps)
        img = img - img.min()
        img = img / (img.max() + eps)
    elif mode == "minmax":
        img = img - img.min()
        img = img / (img.max() + eps)
    else:
        raise ValueError("normalize debe ser 'zscore_then_minmax' o 'minmax'")
    return img

def _safe_nfft(nperseg: int, nfft: int | None) -> int:
    if nfft is None:
        return nperseg
    if nfft < nperseg:
        return int(2 ** int(np.ceil(np.log2(nperseg))))
    return int(nfft)

def _tta_spectrogram(img_uint8: np.ndarray) -> np.ndarray:
    """TTA leve: brillo + contraste + ruido gaussian."""
    alpha = np.random.uniform(0.95, 1.05)
    beta = np.random.uniform(-5, 5)
    img = cv2.convertScaleAbs(img_uint8, alpha=alpha, beta=beta)
    if np.random.rand() < 0.4:
        noise = np.random.normal(0, 3.0, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return img


def signal_to_spec_img(sig_1d: np.ndarray, fs: float, cfg: SpecCfg) -> np.ndarray:
    sig_1d = np.asarray(sig_1d, dtype=np.float32)
    if not np.isfinite(sig_1d).all():
        sig_1d = np.nan_to_num(sig_1d, nan=0.0, posinf=0.0, neginf=0.0)

    nfft_eff = _safe_nfft(cfg.nperseg, cfg.nfft)
    nover = int(min(cfg.noverlap, cfg.nperseg - 1))

    f, t, Z = stft(sig_1d, fs=fs, window=cfg.window,
                   nperseg=cfg.nperseg, noverlap=nover,
                   nfft=nfft_eff, padded=False, boundary=None)

    mag = np.abs(Z)
    mag = np.nan_to_num(mag, nan=0.0, posinf=0.0, neginf=0.0)

    if cfg.fmax is not None and f.size > 0:
        mask = (f <= cfg.fmax)
        if not np.any(mask):
            mask = np.ones_like(f, dtype=bool)
        mag = mag[mask, :]

    img = np.log1p(mag + cfg.eps)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = _normalize_img(img, cfg.eps, cfg.normalize, cfg.clip_percentiles)
    img = (img * 255.0).astype(np.uint8)

    in_h, in_w = img.shape
    inter = cv2.INTER_AREA if cfg.out_size < min(in_h, in_w) else cv2.INTER_CUBIC
    img = cv2.resize(img, (cfg.out_size, cfg.out_size), interpolation=inter)
    return img
