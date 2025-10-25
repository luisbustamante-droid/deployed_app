import torch
torch.set_default_device("cpu")
torch.set_num_threads(2)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import streamlit as st
from collections import Counter
import zipfile, tempfile, io, os

# <<< NUEVO: imports mínimos para meta y (opcional) splits
import json
import pandas as pd

# <<< NUEVO: imports para inferencia
import torch
import torch.nn.functional as F
from utils.utils_models import build_model

from utils.utils_spectrograms import SpecCfg, signal_to_spec_img
from utils.utils_ecg import (
    DEFAULT_FS_TARGET, load_record, WindowCfg,
    extract_windows_from_rpeaks, get_rpeaks,
    get_aami_distribution, load_record_from_wfdb_zip,
    get_aami_distribution_from_base, validate_and_prepare_signal,
    validate_wfdb_zip_structure
)

# >>> añadido: import global para Plotly
import plotly.graph_objects as go

# >>> añadido: imports para visual clínica estilo ejemplo
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# ==== GRAD-CAM helpers (mínimos) ====
import torch
import torch.nn.functional as F
import numpy as np
import cv2

def _get_cam_layer(model, arch: str):
    """Devuelve la capa CONV adecuada para Grad-CAM según arquitectura."""
    arch = arch.lower()
    if "mobilenet" in arch:
        return model.features[-1]
    if "resnet" in arch:
        return model.layer4
    if "efficientnet" in arch:
        # Torchvision EfficientNetV2_B0: último stage es un Sequential; tomamos su último submódulo conv.
        last_block = model.features[-1]
        if isinstance(last_block, torch.nn.Sequential) and len(list(last_block.children())) > 0:
            return list(last_block.children())[-1]
        return last_block
    # Fallback: última conv del modelo
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m
    return model  # último recurso

def _compute_gradcam(model, arch: str, x_1x3hw: torch.Tensor, target_idx: int) -> np.ndarray:
    """
    Calcula Grad-CAM para x (1,3,H,W) y clase target_idx.
    Devuelve CAM mejorado [0..1] como np.ndarray (H,W).
    Aplica gamma, blur y CLAHE para mejorar interpretabilidad visual.
    """
    layer = _get_cam_layer(model, arch)
    feats, grads = [], []

    def _fhook(_, __, output):
        out = output[0] if isinstance(output, (tuple, list)) else output
        feats.append(out.detach().contiguous().clone())

    def _bhook(_, grad_input, grad_output):
        g = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
        grads.append(g.detach().contiguous().clone())

    for m in model.modules():
        if isinstance(m, torch.nn.SiLU):
            m.inplace = False

    h1 = layer.register_forward_hook(_fhook)
    h2 = layer.register_full_backward_hook(_bhook)

    model.zero_grad(set_to_none=True)
    x_1x3hw.requires_grad_(True)

    with torch.enable_grad():
        logits = model(x_1x3hw)
        if target_idx is None:
            target_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[0, target_idx]
        score.backward(retain_graph=True)

    h1.remove()
    h2.remove()

    if not feats or not grads:
        raise RuntimeError("No se capturaron activaciones/gradientes para Grad-CAM.")

    A = feats[-1]
    G = grads[-1]
    weights = torch.mean(G, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * A, dim=1, keepdim=False)
    cam = torch.relu(cam)[0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam_np = cam.detach().cpu().numpy()
    cam_np = cv2.resize(cam_np, (x_1x3hw.shape[-1], x_1x3hw.shape[-2]), interpolation=cv2.INTER_CUBIC)

    # ==== NUEVO: realce visual (versión robusta) ====
    # 1. Limpieza numérica
    cam_np = np.nan_to_num(cam_np, nan=0.0, posinf=0.0, neginf=0.0)
    cam_np = np.clip(cam_np, 0, 1)

    # 2. Potenciar regiones débiles (gamma < 1)
    gamma = 0.6
    try:
        cam_np = np.power(cam_np, gamma)
    except Exception:
        cam_np = np.zeros_like(cam_np)

    # 3. Suavizado gaussiano
    cam_np = cv2.GaussianBlur(cam_np, (9, 9), sigmaX=3)

    # 4. Realce local de contraste (CLAHE)
    cam_uint8 = np.uint8(255 * np.clip(cam_np, 0, 1))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cam_eq = clahe.apply(cam_uint8)
    cam_np = cam_eq.astype(np.float32) / 255.0

    # 5. Normalización final y limpieza extra
    cam_np = np.nan_to_num(cam_np, nan=0.0)
    cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
    cam_np = np.clip(cam_np, 0, 1)

    return cam_np

def _cam_time_profile(cam_hw: np.ndarray) -> np.ndarray:
    """Proyección temporal usando energía cuadrática (resalta atención fuera del centro)."""
    # Evita promediar linealmente: usa potencia para resaltar diferencias
    power_map = np.power(np.maximum(cam_hw, 0), 1.5)
    time_imp = power_map.mean(axis=0)
    time_imp /= (np.percentile(time_imp, 99) + 1e-8)
    time_imp = np.clip(time_imp, 0, 1)
    return time_imp


# ===============================================================
# CONFIGURACIÓN DE RUTAS Y PARÁMETROS
# ===============================================================
DATA_DIR_MITDB = "./assets/mit-bih-arrhythmia-database-1.0.0"
DATA_DIR_SVDB  = "./assets/mit-bih-supraventricular-arrhythmia-database-1.0.0"
FS_TARGET = 360

# <<< NUEVO: pequeñas utilidades locales (no invaden tu flujo)
def _load_model_meta(path="assets/model_meta.json"):
    """
    Carga metadatos del modelo si existen.
    """
    if "CLASS_NAMES" not in st.session_state:
        st.session_state.CLASS_NAMES = ["N","S","V","F","Q"]
    if "SPEC_CFG" not in st.session_state:
        st.session_state.SPEC_CFG = SpecCfg(
            nperseg=128, noverlap=64, nfft=None, window="hann",
            fmax=60.0, out_size=224, normalize="zscore_then_minmax",
            clip_percentiles=(1,99)
        )
    if "WIN_SEC" not in st.session_state:
        st.session_state.WIN_SEC = 2.5
    if "FS_TARGET" not in st.session_state:
        st.session_state.FS_TARGET = FS_TARGET
    if "MODEL_ARCH" not in st.session_state:
        st.session_state.MODEL_ARCH = "efficientnetv2_b0"
    if "NUM_CLASSES" not in st.session_state:
        st.session_state.NUM_CLASSES = len(st.session_state.CLASS_NAMES)
    if "WEIGHTS_PATH" not in st.session_state:
        st.session_state.WEIGHTS_PATH = "assets/efficientnet_v2_b0_best_ema.pt"

    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        cls = meta.get("class_names")
        if isinstance(cls, (list, tuple)) and len(cls) > 0:
            st.session_state.CLASS_NAMES = list(map(str, cls))

        scfg = meta.get("spec_cfg", {})
        if isinstance(scfg, dict):
            st.session_state.SPEC_CFG = SpecCfg(
                nperseg=scfg.get("nperseg", 128),
                noverlap=scfg.get("noverlap", 64),
                nfft=scfg.get("nfft", None),
                window=scfg.get("window", "hann"),
                fmax=scfg.get("fmax", 60.0),
                out_size=scfg.get("out_size", 224),
                normalize=scfg.get("normalize", "zscore_then_minmax"),
                clip_percentiles=tuple(scfg.get("clip_percentiles", (1,99))),
            )

        if "win_sec" in meta:
            st.session_state.WIN_SEC = float(meta["win_sec"])
        if "fs_target" in meta:
            st.session_state.FS_TARGET = int(meta["fs_target"])

        if "arch" in meta:
            st.session_state.MODEL_ARCH = str(meta["arch"])
        if "num_classes" in meta:
            st.session_state.NUM_CLASSES = int(meta["num_classes"])
        else:
            st.session_state.NUM_CLASSES = len(st.session_state.CLASS_NAMES)
        if "weights_path" in meta:
            st.session_state.WEIGHTS_PATH = str(meta["weights_path"])

    except FileNotFoundError:
        pass
    except Exception as e:
        st.warning(f"model_meta.json no pudo cargarse: {e}")
    # Forzar EfficientNetV2_B0 por defecto
    st.session_state.MODEL_ARCH = "efficientnet_v2_b0"


def _try_load_test_records_from_splits(pkl_path="assets/df_windows_mitdb_svdb_splits_final.pkl"):
    try:
        df = pd.read_pickle(pkl_path)
        if "db_record_id" not in df.columns or "split" not in df.columns:
            return None
        test_ids = (
            df.loc[df["split"]=="test", "db_record_id"]
              .astype(str).unique().tolist()
        )
        test_records = []
        for rid in test_ids:
            parts = rid.split("_")
            if len(parts) >= 2:
                prefix = parts[0].upper()
                num = parts[1]
                if prefix in ("MITDB","SVDB"):
                    test_records.append(f"{prefix}_{num}")
        if test_records:
            return sorted(list(set(test_records)))
    except FileNotFoundError:
        return None
    except Exception as e:
        st.warning(f"No se pudo leer splits de prueba: {e}")
    return None

# <<< NUEVO: helpers de inferencia
def _ensure_model_loaded():
    device = torch.device("cpu")
    st.session_state.DEVICE = device

    if "MODEL" in st.session_state and st.session_state.MODEL is not None:
        return

    arch = st.session_state.MODEL_ARCH
    num_classes = int(st.session_state.NUM_CLASSES)
    weights_path = st.session_state.WEIGHTS_PATH

    if not os.path.exists(weights_path):
        st.error(f"No se encontró el archivo de pesos: {weights_path}")
        st.stop()

    # Construcción
    model = build_model(arch=arch, num_classes=num_classes, pretrained=False)

    # Carga robusta
    try:
        sd = torch.load(weights_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        new_sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        if missing:
            st.warning(f"Capas faltantes al cargar pesos: {len(missing)}")
        if unexpected:
            st.warning(f"Capas inesperadas en pesos: {len(unexpected)}")
    except Exception as e:
        st.error(f"No se pudieron cargar los pesos: {e}")
        st.stop()

    model.eval().to(device)
    st.session_state.MODEL = model


def _windows_to_batch_spectrograms(windows: np.ndarray, fs: int, spec_cfg: SpecCfg) -> torch.Tensor:
    """
    Convierte una lista de ventanas 1D en un batch 4D [N,3,H,W].
    Aplica siempre un leve Test-Time Augmentation (brillo, contraste y ruido)
    para igualar el dominio de inferencia al entrenamiento.
    """
    if windows.ndim == 2:
        X = windows
    elif windows.ndim == 3 and windows.shape[1] == 1:
        X = windows[:, 0, :]
    else:
        raise ValueError(f"Forma de ventanas no soportada: {windows.shape}")

    imgs = []
    for i in range(X.shape[0]):
        # --- Espectrograma base ---
        img = signal_to_spec_img(X[i], fs=fs, cfg=spec_cfg)

        # --- TTA leve (obligatorio) ---
        try:
            from utils.utils_spectrograms import _tta_spectrogram
            img = _tta_spectrogram(img)
        except Exception:
            # fallback silencioso si no existe o falla
            alpha = np.random.uniform(0.95, 1.05)
            beta = np.random.uniform(-5, 5)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            if np.random.rand() < 0.4:
                noise = np.random.normal(0, 3.0, img.shape)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # --- Normalización y tensor ---
        imgf = (img.astype(np.float32) / 255.0)[None, ...]  # [1,H,W]
        img3 = np.repeat(imgf, 3, axis=0)                   # [3,H,W]
        imgs.append(img3)

    batch = np.stack(imgs, axis=0)  # [N,3,H,W]
    return torch.from_numpy(batch)

def _predict_windows(model, batch_3chw: torch.Tensor) -> np.ndarray:
    device = st.session_state.DEVICE
    probs_all = []
    with torch.no_grad():
        B = 64
        for i in range(0, batch_3chw.shape[0], B):
            x = batch_3chw[i:i+B].to(device, dtype=torch.float32)
            logits = model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            probs_all.append(probs)
    return np.concatenate(probs_all, axis=0)

# ===============================================================
# CLASE PRINCIPAL STREAMLIT
# ===============================================================
class InformeClinico:
    def __init__(self):
        pass

    def render(self):
        st.markdown("""
        <link rel="stylesheet"
              href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
        """, unsafe_allow_html=True)

        st.title("Informe Clínico")
        st.markdown("<hr style='margin-top:0.5rem; margin-bottom:1.5rem; border:1px solid #ccc;'>",
                    unsafe_allow_html=True)

        _load_model_meta()

        TEST_RECORDS = [
            'MITDB_108', 'MITDB_113', 'MITDB_117', 'MITDB_123', 'MITDB_124',
            'SVDB_821', 'SVDB_822', 'SVDB_851', 'SVDB_852', 'SVDB_854',
            'SVDB_855', 'SVDB_859', 'SVDB_860', 'SVDB_863', 'SVDB_866',
            'SVDB_868', 'SVDB_869', 'SVDB_870', 'SVDB_879', 'SVDB_881',
            'SVDB_884', 'SVDB_885', 'SVDB_891', 'SVDB_892'
        ]

        test_records_from_pkl = _try_load_test_records_from_splits()
        if test_records_from_pkl:
            TEST_RECORDS = test_records_from_pkl

        st.markdown("""
        <style>
        div[data-testid="stFileUploader"] section[data-testid="stFileUploaderDropzone"] {
            padding: 0.75rem 0.5rem;
        }
        div[data-testid="stFileUploader"] { margin-top: 0.25rem; }
        </style>
        """, unsafe_allow_html=True)

        cols_top = st.columns(2, gap="medium")
        with cols_top[0]:
            st.caption("Seleccione un registro de prueba:")
            record_id = st.selectbox(
                label="Seleccione un registro de prueba:",
                options=TEST_RECORDS,
                index=0,
                label_visibility="collapsed"
            )
        with cols_top[1]:
            st.caption("…o suba un archivo (ZIP WFDB estilo MIT-BIH)")
            uploaded_file = st.file_uploader(
                label="…o suba un archivo (ZIP WFDB estilo MIT-BIH)",
                type=["zip"],
                help="Sube un ZIP con .hea/.dat (y opcional .atr) del mismo basename.",
                label_visibility="collapsed"
            )

        use_upload = uploaded_file is not None

        if use_upload:
            try:
                st.info("Paquete WFDB (ZIP) detectado. Intentando leer record y anotaciones…")
                zip_bytes = uploaded_file.getvalue()
                try:
                    validate_wfdb_zip_structure(zip_bytes)
                except Exception as e:
                    st.error(f"ZIP WFDB inválido: {e}")
                    st.stop()

                sig, fs, base_temp, has_ann = load_record_from_wfdb_zip(io.BytesIO(zip_bytes))

                try:
                    sig, fs, _warns = validate_and_prepare_signal(sig, fs, fs_target=int(st.session_state.get("FS_TARGET", FS_TARGET)))
                    for w in _warns:
                        st.warning(w)
                except Exception as e:
                    st.error(f"Señal WFDB incompatible: {e}")
                    st.stop()

                if has_ann:
                    aami_counts, aami_err = get_aami_distribution_from_base(base_temp)
                    aami_available = True
                else:
                    aami_counts, aami_err = Counter(), "No se encontraron anotaciones 'atr' en el ZIP."
                    aami_available = False

            except Exception as e:
                st.error(f"No se pudo cargar el archivo subido: {e}")
                st.stop()

        else:
            if not record_id:
                st.info("Seleccione un registro o suba un archivo para comenzar.")
                st.stop()

            if record_id.startswith("MITDB_"):
                data_dir = DATA_DIR_MITDB
                real_id = record_id.replace("MITDB_", "")
            elif record_id.startswith("SVDB_"):
                data_dir = DATA_DIR_SVDB
                real_id = record_id.replace("SVDB_", "")
            else:
                st.error("Formato de registro no reconocido.")
                st.stop()

            sig, fs = load_record(real_id, data_dir, fs_target=FS_TARGET)
            try:
                sig, fs, _warns = validate_and_prepare_signal(sig, fs, fs_target=int(st.session_state.get("FS_TARGET", FS_TARGET)))
                for w in _warns:
                    st.warning(w)
            except Exception as e:
                st.error(f"Registro incompatible: {real_id}. Detalle: {e}")
                st.stop()

            aami_counts, aami_err = get_aami_distribution(real_id, data_dir)
            aami_available = (sum(aami_counts.values()) > 0)

        rpeaks = get_rpeaks(sig[:, 0], fs)
        # === Opción B: usar ventanas más largas (5 s) solo para Grad-CAM / inferencia ===
        base_win = float(st.session_state.get("WIN_SEC", 2.5))
        cfg = WindowCfg(win_sec=base_win * 2.0, zscore_per_window=True)  # 5 s si el original era 2.5 s
        windows = extract_windows_from_rpeaks(sig[:, 0], fs, rpeaks, cfg)

        col1,col2 = st.columns(2)
        with col1:
            st.success(f"Fs = {fs} Hz | Detectados {len(rpeaks)} R-peaks.")
        with col2:
            st.success(f"Extraídas {len(windows)} ventanas de {cfg.win_sec:.2f} segundos.")

        sel_idx = st.slider("Seleccionar ventana para visualizar:", 0, max(len(windows) - 1, 0), 0)
        if len(windows) > 0:
            w = windows[sel_idx]
            fig_win = go.Figure()
            fig_win.add_trace(go.Scatter(
                x=np.linspace(-cfg.win_sec / 2, cfg.win_sec / 2, len(w)),
                y=w,
                mode='lines',
                name=f"Ventana {sel_idx}"
            ))
            fig_win.update_layout(
                title=f"Ventana {sel_idx} centrada en R-peak {rpeaks[sel_idx] if len(rpeaks) > sel_idx else '—'} (z-score)",
                xaxis_title="Tiempo relativo (s)",
                yaxis_title="Amplitud normalizada",
                height=300,
                margin=dict(l=40, r=40, t=60, b=40),
                showlegend=False
            )
            st.plotly_chart(fig_win, use_container_width=True)
        else:
            st.info("No se detectaron R-peaks en la señal.")

        st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
        colp1, colp2 = st.columns([1,3])
        with colp1:
            predict_clicked = st.button("Predecir", type="primary", use_container_width=True)

        if predict_clicked:
            if len(windows) == 0:
                st.warning("No hay ventanas para inferir.")
                st.stop()
            try:
                progress_text = st.empty()
                progress_bar = st.progress(0)

                try:
                    progress_text.text("Cargando modelo…")
                    _ensure_model_loaded()
                    progress_bar.progress(20)

                    progress_text.text("Generando espectrogramas de las ventanas…")
                    batch = _windows_to_batch_spectrograms(
                        windows=np.asarray(windows, dtype=np.float32),
                        fs=fs,
                        spec_cfg=st.session_state.SPEC_CFG
                    )
                    progress_bar.progress(50)

                    progress_text.text("Ejecutando inferencia por lotes…")
                    probs = _predict_windows(st.session_state.MODEL, batch)  # [N,C]
                    progress_bar.progress(80)

                    cls_names = st.session_state.CLASS_NAMES
                    C = probs.shape[1]
                    if C != len(cls_names):
                        st.warning(
                            f"Aviso: el modelo tiene {C} salidas y CLASS_NAMES={len(cls_names)}. Revisa 'class_names' en model_meta.json.")

                    mean_probs = probs.mean(axis=0)
                    pred_idx = int(np.argmax(mean_probs))
                    pred_label = cls_names[pred_idx] if pred_idx < len(cls_names) else f"cls{pred_idx}"
                    pred_conf = float(mean_probs[pred_idx])

                    progress_text.text("Finalizando resultados…")
                    progress_bar.progress(100)

                finally:
                    progress_text.empty()
                    progress_bar.empty()

                st.success(f"Predicción del registro: **{pred_label}** ({pred_conf*100:.1f}%)")

                # ==== Mensaje clínico según clase AAMI predicha ====
                if pred_label == "N":
                    st.info(
                        "**Clase N (Normal):** el trazado corresponde a un ritmo sinusal o a latidos con morfología normal. "
                        "No se observan alteraciones aparentes en la morfología del complejo QRS ni en el intervalo de la ventana analizada. "
                        "*Este resultado no reemplaza la evaluación médica. Todo trazado debe ser interpretado por un profesional de la salud.*"
                    )

                elif pred_label == "S":
                    st.warning(
                        "**Clase S (Supraventricular):** el modelo identifica patrones compatibles con actividad auricular o supraventricular prematura, "
                        "posiblemente relacionados con contracciones auriculares ectópicas o ritmos supraventriculares transitorios. "
                        "*Se recomienda valoración clínica especializada, especialmente si estos hallazgos son frecuentes o sintomáticos.*"
                    )

                elif pred_label == "V":
                    st.error(
                        "**Clase V (Ventricular):** se detecta actividad ventricular prematura o morfología sugestiva de contracciones ventriculares ectópicas. "
                        "Estos hallazgos pueden ser clínicamente relevantes, sobre todo si son recurrentes, multifocales o se presentan en salvas. "
                        "*Es imprescindible la revisión inmediata por un cardiólogo para descartar patología ventricular significativa.*"
                    )

                elif pred_label == "F":
                    st.info(
                        "**Clase F (Fusión):** el modelo identifica un latido de fusión, es decir, una contracción resultante de la activación simultánea "
                        "de un impulso normal y otro ectópico. Por sí sola, esta clase no suele tener implicaciones clínicas importantes, "
                        "pero puede aparecer en el contexto de otras arritmias. "
                        " *Se aconseja interpretar este hallazgo junto con el resto del registro y bajo supervisión médica.*"
                    )
                    st.stop()  # evita mostrar el bloque de detalle

                elif pred_label == "Q":
                    st.info(
                        "**Clase Q (No clasificada / Artefacto):** la señal presenta morfología atípica o ruido significativo, "
                        "lo que impide una clasificación confiable. Esto puede deberse a artefactos de movimiento, interferencia eléctrica "
                        "o mala calidad de registro. "
                        "*El resultado no tiene valor diagnóstico y debe ser revisado por un profesional con acceso al trazado original.*"
                    )
                    st.stop()  # evita mostrar el bloque de detalle

            except FileNotFoundError:
                st.error(f"No se encontró el archivo de pesos: {st.session_state.WEIGHTS_PATH}.")
            except Exception as e:
                st.error(f"Error durante la inferencia: {e}")
        # ====== Vista detallada: espectrograma + señal cruda de la ventana más “evidente” ======
        try:
            cls_names = st.session_state.CLASS_NAMES
            pred_class_idx = pred_idx

            win_pred = probs.argmax(axis=1)
            conf_for_pred = probs[:, pred_class_idx]
            mask = (win_pred == pred_class_idx)
            if mask.any():
                best_idx = int(np.argmax(conf_for_pred * mask))
            else:
                best_idx = int(np.argmax(conf_for_pred))

            w_best = windows[best_idx]
            t_best = np.linspace(-cfg.win_sec / 2, cfg.win_sec / 2, len(w_best))

            st.markdown("### Detalle de la detección principal")

            # dos columnas: izquierda señal cruda, derecha Grad-CAM
            col_signal, col_gradcam = st.columns([1, 1], gap="large")

            with col_signal:
                st.caption(f"Señal cruda — Ventana {best_idx} (centrada en R-peak)")
                fig_best = go.Figure()
                fig_best.add_trace(go.Scatter(
                    x=t_best, y=w_best, mode='lines', name='ventana'
                ))
                fig_best.add_vline(x=0.0, line_dash="dash", opacity=0.6)
                fig_best.update_layout(
                    height=300,
                    margin=dict(l=40, r=20, t=30, b=40),
                    xaxis_title="Tiempo relativo (s)",
                    yaxis_title="Amplitud (z-score)",
                    showlegend=False
                )
                st.plotly_chart(fig_best, use_container_width=True)

            with col_gradcam:
                st.caption("Grad-CAM sobre la misma ventana")
                try:
                    # --- Prepara tensor para Grad-CAM ---
                    img_cam = signal_to_spec_img(
                        w_best.astype(np.float32), fs=int(fs), cfg=st.session_state.SPEC_CFG
                    )  # uint8 [H,W]
                    x_cam = (img_cam.astype(np.float32) / 255.0)[None, None, ...]
                    x_cam = np.repeat(x_cam, 3, axis=1)
                    x_cam = torch.from_numpy(x_cam).to(st.session_state.DEVICE, dtype=torch.float32)

                    _ensure_model_loaded()
                    cam = _compute_gradcam(
                        st.session_state.MODEL,
                        st.session_state.MODEL_ARCH,
                        x_cam,
                        target_idx=int(pred_class_idx)
                    )  # [H,W] 0..1

                    # --- Proyección temporal sobre la señal ---
                    time_imp = _cam_time_profile(cam)
                    t = t_best
                    # Mantener correspondencia temporal: espectrograma → señal
                    cam_time = np.linspace(-cfg.win_sec / 2, cfg.win_sec / 2, len(time_imp))
                    time_imp_interp = np.interp(t, cam_time, time_imp)  # interpola respetando tiempos reales
                    time_imp = (time_imp_interp - time_imp_interp.min()) / (time_imp_interp.max() + 1e-8)

                    rec_id = record_id if not use_upload else "ZIP"

                    # usar misma predicción mostrada arriba
                    det_label = pred_label
                    confidence = pred_conf
                    if aami_available and len(aami_counts) > 0:
                        real_label = max(aami_counts.items(), key=lambda kv: kv[1])[0]
                    else:
                        real_label = "—"

                    # --- Señal y Grad-CAM estilo clínico ---
                    sig_plot = w_best / (np.max(np.abs(w_best)) + 1e-8)
                    det_time = 0.0
                    t_max = t[np.argmax(time_imp)]
                    cmap = cm.get_cmap("turbo")

                    plt.rcParams.update({
                        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                        "xtick.labelsize": 9, "ytick.labelsize": 9,
                        "figure.facecolor": "white", "axes.facecolor": "white"
                    })

                    fig, ax1 = plt.subplots(figsize=(8, 3.2))
                    # Fondo coloreado continuo según atención temporal
                    for i in range(len(t) - 1):
                        ax1.axvspan(t[i], t[i + 1], color=cmap(time_imp[i]), alpha=0.45)
                    # Línea base y señal principal
                    ax1.axhline(0, color="gray", lw=0.4, ls="--", alpha=0.4)
                    ax1.plot(t, sig_plot, color="black", lw=1.3, zorder=5)

                    # Líneas clínicas
                    ax1.axvline(det_time, color="red", ls="--", lw=1.2, label="Latido detectado")
                    ax1.axvline(t_max, color="dodgerblue", ls=":", lw=1.2, label="Máx. atención Grad-CAM")

                    # Título y etiquetas
                    ax1.set_title(
                        f"Grad-CAM sobre señal ECG ({cfg.win_sec:.1f} s) — {rec_id}\n"
                        f"Pred: {det_label} ({confidence * 100:.1f} %)"
                    )
                    ax1.set_xlabel("Tiempo [s]")
                    ax1.set_ylabel("Amplitud normalizada")
                    ax1.grid(alpha=0.25)
                    ax1.legend(loc="upper right", fontsize=8, frameon=False)

                    # Curva Grad-CAM (eje secundario)
                    ax2 = ax1.twinx()
                    ax2.plot(t, time_imp, color="royalblue", lw=0.9, alpha=0.6)
                    ax2.fill_between(t, 0, time_imp, color="royalblue", alpha=0.15)
                    ax2.set_ylim(0, 1.05)
                    ax2.set_ylabel("Intensidad Grad-CAM", color="royalblue", fontsize=9)

                    # Texto descriptivo al pie
                    ax1.text(
                        0.02, -0.25,
                        "La región coloreada indica el intervalo temporal con mayor contribución del modelo al diagnóstico.",
                        transform=ax1.transAxes, fontsize=8, color="dimgray"
                    )

                    plt.tight_layout()
                    st.pyplot(fig, clear_figure=True)

                except Exception as e:
                    st.warning(f"No se pudo generar Grad-CAM: {e}")


        except Exception as e:
            st.warning(f"No se pudo renderizar el detalle de la detección principal: {e}")
