# utils_ecg.py
# Utilidades livianas para carga WFDB, ventanas, picos R, AAMI y validaciones.

from __future__ import annotations

import io
import os
import zipfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from typing import List, Tuple

import numpy as np
import wfdb
from scipy.signal import resample_poly, find_peaks

# ---------------------------------------------------------------------
# Config base
# ---------------------------------------------------------------------
DEFAULT_FS_TARGET = 360  # Hz

# ---------------------------------------------------------------------
# Carga de registros WFDB locales (MITDB/SVDB) con resample opcional
# ---------------------------------------------------------------------
def load_record(record_id: str, data_dir: str, fs_target: int = DEFAULT_FS_TARGET) -> tuple[np.ndarray, int]:
    """
    Carga un registro WFDB desde data_dir/record_id.{dat,hea}, devuelve (sig[N,C], fs).
    Si fs != fs_target, remuestrea a fs_target.
    """
    rec_path = str(Path(data_dir) / record_id)
    rec = wfdb.rdrecord(rec_path)
    sig = np.asarray(rec.p_signal, dtype=np.float32)
    fs_src = int(rec.fs)
    if sig.ndim == 1:
        sig = sig[:, None]
    if fs_src != fs_target:
        sig = resample_poly(sig, fs_target, fs_src, axis=0).astype(np.float32)
        fs_src = fs_target
    return sig, fs_src

# ---------------------------------------------------------------------
# Ventaneo centrado en picos R
# ---------------------------------------------------------------------
@dataclass
class WindowCfg:
    win_sec: float = 2.5
    zscore_per_window: bool = True
    pad_mode: str = "reflect"

def _slice_with_padding(x: np.ndarray, start: int, length: int, pad_mode: str = "reflect") -> np.ndarray:
    n = len(x)
    end = start + length
    core = x[max(start, 0):min(end, n)]
    if start < 0:
        core = np.pad(core, (abs(start), 0), mode=pad_mode)
    if end > n:
        core = np.pad(core, (0, end - n), mode=pad_mode)
    return core[:length]

def extract_windows_from_rpeaks(signal: np.ndarray, fs: int, rpeaks: np.ndarray, cfg: WindowCfg | None = None) -> np.ndarray:
    """
    Extrae ventanas centradas en cada R-peak.
    Devuelve array shape (n_windows, win_len).
    """
    if cfg is None:
        cfg = WindowCfg()
    win_len = int(cfg.win_sec * fs)
    half = win_len // 2
    windows: List[np.ndarray] = []
    for r in rpeaks:
        start = int(r) - half
        xw = _slice_with_padding(signal, start, win_len, cfg.pad_mode)
        if cfg.zscore_per_window:
            mu, sd = np.mean(xw), np.std(xw) + 1e-6
            xw = (xw - mu) / sd
        windows.append(xw.astype(np.float32, copy=False))
    return np.asarray(windows, dtype=np.float32)

def get_rpeaks(signal: np.ndarray, fs: int) -> np.ndarray:
    """
    Detección simple de picos R con find_peaks (heurística rápida).
    """
    peaks, _ = find_peaks(signal, distance=int(0.2 * fs), height=np.percentile(signal, 95))
    return peaks.astype(np.int32, copy=False)

# ---------------------------------------------------------------------
# Mapeo MIT-BIH/SVDB a clases AAMI
# ---------------------------------------------------------------------
def _symbol_to_aami(sym: str) -> str:
    sym = (sym or "").strip()
    set_N = {'N', 'L', 'R', 'e', 'j', 'B'}
    set_S = {'A', 'a', 'J', 'S'}
    set_V = {'V', 'E'}
    set_F = {'F'}
    set_Q = {'/', 'f', 'Q', '?'}
    if sym in set_N: return 'N'
    if sym in set_S: return 'S'
    if sym in set_V: return 'V'
    if sym in set_F: return 'F'
    if sym in set_Q: return 'Q'
    return 'Q'

def get_aami_distribution(record_id: str, data_dir: str) -> tuple[Counter, str | None]:
    """
    Lee anotaciones WFDB 'atr' y devuelve conteos por clase AAMI.
    """
    rec_path = str(Path(data_dir) / record_id)
    try:
        ann = wfdb.rdann(rec_path, 'atr')
    except Exception as e:
        return Counter(), f"No se pudieron leer anotaciones 'atr' para {record_id}: {e}"
    if not hasattr(ann, "symbol") or ann.symbol is None or len(ann.symbol) == 0:
        return Counter(), f"El archivo de anotaciones no contiene símbolos en {record_id}."
    counts = Counter(_symbol_to_aami(sym) for sym in ann.symbol)
    return counts, None

# ---------------------------------------------------------------------
# Carga desde ZIP WFDB estilo MIT-BIH
# ---------------------------------------------------------------------
def load_record_from_wfdb_zip(zip_file: io.BytesIO | bytes | str) -> tuple[np.ndarray, int, str, bool]:
    """
    Acepta un ZIP con .hea/.dat (opcional .atr), lo extrae a /tmp y carga.
    Retorna: (sig[N,1], fs, base_path_temp, has_ann)
    """
    tmpdir = tempfile.mkdtemp(prefix="wfdb_")
    # Soportar ruta, bytes o BytesIO
    if isinstance(zip_file, (bytes, bytearray)):
        zf = zipfile.ZipFile(io.BytesIO(zip_file))
    elif isinstance(zip_file, io.BytesIO):
        zf = zipfile.ZipFile(zip_file)
    else:
        zf = zipfile.ZipFile(zip_file)
    with zf:
        zf.extractall(tmpdir)

    hea_files = [f for f in os.listdir(tmpdir) if f.lower().endswith(".hea")]
    if not hea_files:
        raise ValueError("ZIP no contiene archivo .hea de WFDB.")
    hea_path = os.path.join(tmpdir, hea_files[0])
    base = hea_path[:-4]  # sin .hea

    rec = wfdb.rdrecord(base)
    sig = np.asarray(rec.p_signal, dtype=np.float32)
    if sig.ndim == 1:
        sig = sig[:, None]
    fs = int(rec.fs)

    has_ann = False
    try:
        _ = wfdb.rdann(base, 'atr')
        has_ann = True
    except Exception:
        has_ann = False

    return sig, fs, base, has_ann

def get_aami_distribution_from_base(base_path: str) -> tuple[Counter, str | None]:
    try:
        ann = wfdb.rdann(base_path, 'atr')
    except Exception as e:
        return Counter(), f"No se pudieron leer anotaciones 'atr': {e}"
    if not hasattr(ann, "symbol") or ann.symbol is None or len(ann.symbol) == 0:
        return Counter(), "El archivo de anotaciones no contiene símbolos."
    counts = Counter(_symbol_to_aami(sym) for sym in ann.symbol)
    return counts, None

# ---------------------------------------------------------------------
# Validadores de señal
# ---------------------------------------------------------------------
def _interp_nans_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=True)
    n = x.size
    if n == 0:
        return x
    mask = np.isnan(x)
    if not mask.any():
        return x
    if mask.all():
        return np.zeros_like(x)
    idx = np.arange(n, dtype=np.float32)
    x[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return x

def _resample_to_target(sig: np.ndarray, fs_src: int, fs_target: int) -> np.ndarray:
    if fs_src == fs_target:
        return sig.astype(np.float32, copy=False)
    return resample_poly(sig, fs_target, fs_src, axis=0).astype(np.float32)

def validate_and_prepare_signal(
    sig: np.ndarray,
    fs: int,
    fs_target: int = DEFAULT_FS_TARGET,
    min_seconds: float = 5.0
) -> tuple[np.ndarray, int, list[str]]:
    """
    - Verifica Fs [50..2000]
    - Interpola NaNs
    - Chequea duración mínima
    - Descarta señal constante
    - Remuestrea a fs_target
    Devuelve: (sig_ok[N,1], fs_target, warnings)
    """
    issues: List[str] = []

    if sig is None:
        raise ValueError("No se recibió señal.")
    sig = np.asarray(sig)
    if sig.ndim == 1:
        sig = sig[:, None]
    if sig.ndim != 2 or sig.shape[1] < 1:
        raise ValueError(f"Se esperaba una matriz [N, C] con al menos 1 canal; llegó {sig.shape}.")

    if sig.shape[1] > 1:
        vars_ch = sig.var(axis=0)
        ch = int(np.argmax(vars_ch))
        sig = sig[:, [ch]]

    if not (isinstance(fs, (int, np.integer)) or (isinstance(fs, float) and float(fs).is_integer())):
        raise ValueError(f"Frecuencia de muestreo inválida: {fs}")
    fs = int(fs)
    if fs < 50 or fs > 2000:
        raise ValueError(f"Fs fuera de rango esperado [50..2000] Hz: {fs}")

    x = sig[:, 0].astype(np.float32)
    bad = ~np.isfinite(x)
    if bad.any():
        issues.append(f"Se interpolaron {int(bad.sum())} muestras no finitas (NaN/Inf).")
        x = _interp_nans_1d(x)
    sig = x[:, None]

    dur = len(sig) / fs if fs > 0 else 0.0
    if dur < min_seconds:
        raise ValueError(f"La señal es demasiado corta ({dur:.2f}s). Mínimo requerido: {min_seconds:.1f}s.")

    if float(np.std(sig)) < 1e-6:
        raise ValueError("La señal parece constante (varianza ~ 0).")

    sig = _resample_to_target(sig, fs_src=fs, fs_target=fs_target)
    fs = fs_target
    return sig.astype(np.float32, copy=False), fs, issues

# ---------------------------------------------------------------------
# Validador de estructura de ZIP WFDB
# ---------------------------------------------------------------------
def validate_wfdb_zip_structure(zip_bytes: bytes) -> None:
    """
    Verifica que el ZIP contenga al menos un par .hea/.dat con mismo basename.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        heas = [n for n in names if n.lower().endswith(".hea")]
        dats = [n for n in names if n.lower().endswith(".dat")]
        if not heas or not dats:
            raise ValueError("El ZIP debe contener al menos .hea y .dat.")
        bases_hea = {Path(h).stem.lower() for h in heas}
        bases_dat = {Path(d).stem.lower() for d in dats}
        if bases_hea.isdisjoint(bases_dat):
            raise ValueError("No se encontró coincidencia basename entre .hea y .dat en el ZIP.")

# ---------------------------------------------------------------------
# Export público
# ---------------------------------------------------------------------
__all__ = [
    "DEFAULT_FS_TARGET",
    "load_record",
    "WindowCfg",
    "_slice_with_padding",
    "extract_windows_from_rpeaks",
    "get_rpeaks",
    "_symbol_to_aami",
    "get_aami_distribution",
    "load_record_from_wfdb_zip",
    "get_aami_distribution_from_base",
    "_interp_nans_1d",
    "_resample_to_target",
    "validate_and_prepare_signal",
    "validate_wfdb_zip_structure",
]
