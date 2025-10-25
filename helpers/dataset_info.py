import streamlit as st
import wfdb
import numpy as np
import plotly.graph_objects as go
import os

def render_dataset_section():
    st.header("Explorador del dataset y señal ECG")

    # ===============================================================
    # DESCRIPCIÓN DE LOS DATASETS
    # ===============================================================
    st.markdown("""
    Este proyecto emplea **dos bases de datos complementarias** de PhysioNet, ambas anotadas clínicamente
    y ampliamente utilizadas en la investigación sobre detección de arritmias cardíacas.

    ### MIT-BIH Arrhythmia Database
    - **Origen:** Beth Israel Hospital (Boston) y MIT Laboratory for Computational Physiology.  
    - **Contenido:** 48 registros de ~30 minutos de duración cada uno.  
    - **Frecuencia de muestreo:** 360 Hz  
    - **Canales:** MLII y V1/V5.  
    - **Etiquetado:** Anotaciones validadas por cardiólogos siguiendo la norma **AAMI EC57**.  
    - **Referencia:** [MIT-BIH Arrhythmia Database – PhysioNet](https://physionet.org/content/mitdb/1.0.0/)

    ### MIT-BIH Supraventricular Arrhythmia Database (SVDB)
    - **Origen:** Beth Israel Hospital (Boston).  
    - **Contenido:** 78 registros de pacientes con predominancia de arritmias **supraventriculares**.  
    - **Frecuencia de muestreo:** 128 Hz  
    - **Canales:** MLII y V1/V2.  
    - **Propósito:** Mejorar la representatividad de las clases **S** en el esquema AAMI (supraventriculares).  
    - **Referencia:** [MIT-BIH Supraventricular Arrhythmia Database – PhysioNet](https://physionet.org/content/svdb/1.0.0/)
    """)

    st.divider()
    st.subheader("Visualización interactiva de señal ECG")

    # ===============================================================
    # CONFIGURACIÓN DE RUTAS BASE
    # ===============================================================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(current_dir, "..", "assets")

    datasets = {
        "MIT-BIH Arrhythmia Database (mitdb)": "mit-bih-arrhythmia-database-1.0.0",
        "MIT-BIH Supraventricular Arrhythmia Database (svdb)": "mit-bih-supraventricular-arrhythmia-database-1.0.0"
    }

    # ===============================================================
    # SELECCIÓN DE DATASET Y REGISTRO
    # ===============================================================
    col1, col2 = st.columns([1, 1])
    with col1:
        dataset_name = st.selectbox(
            "Selecciona la base de datos",
            list(datasets.keys()),
            key="dataset_selector"
        )

    dataset_dir = os.path.normpath(os.path.join(base_path, datasets[dataset_name]))
    if not os.path.exists(dataset_dir):
        st.error(f"No se encontró la carpeta del dataset:\n`{dataset_dir}`")
        st.info("Verifica que los datasets estén dentro de la carpeta `assets/` del proyecto.")
        return

    available_records = sorted([rec.replace(".dat", "") for rec in os.listdir(dataset_dir) if rec.endswith(".dat")])
    if not available_records:
        st.warning("⚠️ No se encontraron archivos de señal (.dat) en la carpeta seleccionada.")
        return

    with col2:
        record_id = st.selectbox(
            "Selecciona un registro",
            available_records,
            key="record_selector"
        )

    record_path = os.path.join(dataset_dir, record_id)

    # ===============================================================
    # LECTURA DE LA SEÑAL
    # ===============================================================
    try:
        record = wfdb.rdrecord(record_path)
    except Exception as e:
        st.error(f"No se pudo leer el registro `{record_id}`:\n{e}")
        return

    sig = record.p_signal[:, 0]
    fs = record.fs

    # ===============================================================
    # CONTROLES DE VISUALIZACIÓN
    # ===============================================================
    dur = len(sig) / fs
    st.markdown("**Controles de visualización:**")

    colA, colB = st.columns([1, 1])
    with colA:
        seconds = st.slider("Duración de la ventana (s)", min_value=2, max_value=10, value=5, key="window_duration")
    with colB:
        start_sec = st.slider("Inicio de la ventana (s)", 0.0, max(0.0, dur - seconds), 0.0, step=0.1, key="window_start")

    # ===============================================================
    # CÁLCULO Y VISUALIZACIÓN
    # ===============================================================
    start_idx = int(start_sec * fs)
    end_idx = start_idx + int(seconds * fs)
    t = np.arange(start_idx, end_idx) / fs
    y = sig[start_idx:end_idx]

    # === AQUÍ SE CAMBIA EL COLOR DE LA LÍNEA A AZUL CLÍNICO ===
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=y,
        mode="lines",
        line=dict(color="royalblue", width=2),  # ← azul clínico
        name="ECG"
    ))
    fig.update_layout(
        title=f"{dataset_name} — Registro {record_id} (Canal {record.sig_name[0]})",
        xaxis_title="Tiempo (s)",
        yaxis_title="Amplitud (mV)",
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Visualización de la señal cruda. Cada registro contiene anotaciones de latido "
        "que se emplean para generar espectrogramas ECG 2D en el pipeline del modelo."
    )
