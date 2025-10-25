import streamlit as st
import wfdb
import pandas as pd
import numpy as np
import plotly.express as px
import os

class InformeTecnico:
    def __init__(self):
        pass

    def render(self):
        # ===============================================================
        # CARGA DE FONT AWESOME
        # ===============================================================
        st.markdown("""
        <link rel="stylesheet"
              href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
        """, unsafe_allow_html=True)

        # ===============================================================
        # ENCABEZADO PRINCIPAL
        # ===============================================================
        st.title("Informe Técnico")
        st.subheader("Clasificación de Arritmias mediante Espectrogramas de ECG — Modelo EfficientNetV2-B0")
        st.markdown(
            "<hr style='margin-top:0.5rem; margin-bottom:1.5rem; border:1px solid #ccc;'>",
            unsafe_allow_html=True
        )

        # ===============================================================
        # SECCIÓN 1 – OBJETIVO
        # ===============================================================
        st.markdown("""
        <h3 style='margin-bottom:0.3rem'>
            <i class="fa-solid fa-bullseye" style="color:#4b9cd3"></i> Objetivo
        </h3>
        """, unsafe_allow_html=True)

        st.markdown("""
        Este informe documenta el desarrollo y la validación del modelo **EfficientNetV2-B0**, 
        aplicado a la clasificación automática de arritmias cardíacas a partir de espectrogramas de señales ECG.  
        Se evaluó su rendimiento sobre datasets MIT-BIH y SVDB con métricas reproducibles y enfoque biomédico.
        """)

        # ===============================================================
        # SECCIÓN 2 – CONJUNTOS DE DATOS
        # ===============================================================
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <h3 style='margin-bottom:0.3rem'>
            <i class="fa-solid fa-database" style="color:#4b9cd3"></i> Conjuntos de Datos Utilizados
        </h3>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **MIT-BIH Arrhythmia Database (mitdb)**  
            - 48 registros de 30 min  
            - Frecuencia: 360 Hz  
            - Clases: N, S, V, F, Q  
            - Fuente principal para entrenamiento  
            """)
        with col2:
            st.markdown("""
            **MIT-BIH Supraventricular Arrhythmia Database (svdb)**  
            - 78 registros con eventos supraventriculares  
            - Frecuencia: 128 Hz  
            - Enriquecimiento de la clase **S**  
            """)

        st.info(
            "Ambas bases fueron armonizadas en un formato unificado de beats individuales, "
            "convertidos a espectrogramas STFT de 224×224 px normalizados."
        )

        # ===============================================================
        # SECCIÓN 3 – PIPELINE TÉCNICO
        # ===============================================================
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <h3 style='margin-bottom:0.3rem'>
            <i class="fa-solid fa-cogs" style="color:#4b9cd3"></i> Descripción Técnica del Pipeline
        </h3>
        """, unsafe_allow_html=True)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("""
            1. **Preprocesamiento**  
               - Lectura de registros WFDB  
               - Ventanas centradas en R-peaks (2.5 s)  
               - Generación de espectrogramas STFT  
            2. **Balanceo de clases**  
               - **SMOTE 1D** solo en clase F (post-split, train)  
               - Caché: `spec_cache_224/train/AUG`  
            """)
        with colB:
            st.markdown("""
            3. **Entrenamiento**  
               - Arquitectura: **EfficientNetV2-B0**  
               - Optimizador: *AdamW* + *Cosine Annealing*  
               - Pérdida: *ClassBalanced Focal Loss*  
            4. **Evaluación**  
               - Métricas: *Accuracy*, *Macro-F1*, *Recall*, *Precision*  
               - *Val/Test* sin SMOTE ni augment sintético  
            """)

        # ===============================================================
        # SECCIÓN 4 – INFORMACIÓN DEL MODELO
        # ===============================================================
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <h3 style='margin-bottom:0.3rem'>
            <i class="fa-solid fa-microchip" style="color:#4b9cd3"></i> Información del Modelo
        </h3>
        """, unsafe_allow_html=True)

        st.markdown("#### Arquitectura final: EfficientNetV2-B0")
        st.markdown("""
        EfficientNetV2-B0 combina convoluciones con bloques MBConv y *squeeze-and-excitation*, 
        optimizando la relación entre precisión y eficiencia computacional.  
        Su diseño compacto permite una implementación viable en entornos clínicos.
        """)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Resolución de entrada", "224 × 224 px")
        with col2:
            st.metric("Batch Size", "64")
        with col3:
            st.metric("Dropout", "0.20")

        st.markdown("""
        **Configuración de entrenamiento:**  
        - *Épocas:* 15 (con Early Stopping en la 13)  
        - *Learning Rate Inicial:* 3×10⁻⁴  
        - *Weight Decay:* 1×10⁻⁴  
        - *Scheduler:* Cosine Annealing  
        """)

        # ---------------------------------------------------------------
        # MÉTRICAS DE RENDIMIENTO
        # ---------------------------------------------------------------
        st.markdown("""
        <h4 style='margin-top:1rem;'>
            <i class="fa-solid fa-chart-line" style="color:#4b9cd3"></i> Resultados de Evaluación
        </h4>
        """, unsafe_allow_html=True)

        data = {
            "Conjunto": ["Validación", "Test"],
            "Accuracy": [0.9348, 0.8327],
            "Macro-F1": [0.7013, 0.6495],
            "F1 (N)": [0.94, 0.83],
            "F1 (S)": [0.74, 0.80],
            "F1 (V)": [0.94, 0.85],
            "F1 (F)": [0.25, 0.29],
            "F1 (Q)": [0.98, 0.90],
        }
        df_metrics = pd.DataFrame(data)
        st.dataframe(df_metrics, width='stretch', hide_index=True)
        st.caption("*El modelo EfficientNetV2-B0 logra el mejor equilibrio global entre precisión y generalización.*")

        # ---------------------------------------------------------------
        # LIMITACIONES Y ADVERTENCIAS
        # ---------------------------------------------------------------
        with st.expander("Ver Limitaciones y Advertencias"):
            st.markdown("""
            - **Desbalance residual:** la clase F continúa siendo escasa pese al uso de SMOTE.  
            - **Generalización clínica:** entrenado exclusivamente con MIT-BIH/SVDB.  
            - **Canal MLII dominante:** posible sesgo frente a derivaciones distintas.  
            - **Sensibilidad al ruido:** artefactos de movimiento o basales afectan la morfología espectral.  
            - **Interpretabilidad:** las *Grad-CAM* son aproximaciones visuales; **no representan diagnóstico médico.**  
            """)

        # ===============================================================
        # SECCIÓN 5 – METADATOS Y DISTRIBUCIÓN DE LATIDOS
        # ===============================================================
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
        <h3 style='margin-bottom:0.3rem'>
            <i class="fa-solid fa-wave-square" style="color:#4b9cd3"></i> Metadatos y Distribución de Latidos
        </h3>
        """, unsafe_allow_html=True)

        datasets = {
            "MIT-BIH Arrhythmia Database (mitdb)": "mit-bih-arrhythmia-database-1.0.0",
            "MIT-BIH Supraventricular Arrhythmia Database (svdb)": "mit-bih-supraventricular-arrhythmia-database-1.0.0",
        }

        col1, col2 = st.columns([1, 1])
        with col1:
            dataset_name = st.selectbox(
                "Selecciona la base de datos",
                list(datasets.keys()),
                key="dataset_selector_metadata"
            )

        dataset_dir = os.path.join("./assets", datasets[dataset_name])

        if not os.path.exists(dataset_dir):
            st.error(f"No se encontró la carpeta del dataset:\n`{dataset_dir}`")
            st.info("Verifica que los datasets estén dentro de la carpeta `assets/`.")
            return

        available_records = sorted(
            [rec.replace(".dat", "") for rec in os.listdir(dataset_dir) if rec.endswith(".dat")]
        )
        if not available_records:
            st.warning("No se encontraron registros en la carpeta seleccionada.")
            return

        with col2:
            record_id = st.selectbox(
                "Selecciona un registro",
                available_records,
                key="record_selector_metadata"
            )

        record_path = os.path.join(dataset_dir, record_id)

        try:
            record = wfdb.rdrecord(record_path)
            fs = record.fs
            sig_len = len(record.p_signal)
            dur = sig_len / fs
            sig_names = ", ".join(record.sig_name)
            base_time = getattr(record, "base_time", None)
            base_date = getattr(record, "base_date", None)

            # Anotaciones
            annot_path = os.path.join(dataset_dir, record_id)
            df_counts = None
            try:
                annotation = wfdb.rdann(annot_path, "atr")
                symbols = np.array(annotation.symbol)
                mapping = {
                    "N": "N","L": "N","R": "N","e": "N","j": "N",
                    "A": "S","a": "S","J": "S","S": "S",
                    "V": "V","E": "V",
                    "F": "F",
                    "Q": "Q","/": "Q","f": "Q","?": "Q"
                }
                aami_classes = [mapping.get(s, "Q") for s in symbols]
                df_counts = pd.DataFrame(pd.Series(aami_classes).value_counts()).reset_index()
                df_counts.columns = ["Clase AAMI", "Cantidad"]
            except Exception:
                st.warning("No se encontraron anotaciones (.atr) para este registro.")

            # Metadatos y gráfico
            col_meta, col_plot = st.columns([1, 1])
            with col_meta:
                df_info = pd.DataFrame({
                    "Atributo": [
                        "Dataset","Registro","Frecuencia (Hz)",
                        "Muestras","Duración (s)","Canales",
                        "Fecha base","Hora base"
                    ],
                    "Valor": [
                        dataset_name,str(record_id),str(fs),
                        f"{sig_len:,}",f"{dur:.2f}",str(sig_names),
                        str(base_date) if base_date else "No especificada",
                        str(base_time) if base_time else "No especificada"
                    ]
                })
                st.table(df_info)

            with col_plot:
                if df_counts is not None and not df_counts.empty:
                    fig = px.bar(
                        df_counts,
                        x="Clase AAMI",
                        y="Cantidad",
                        text="Cantidad",
                        color="Clase AAMI",
                        color_discrete_map={
                            "N": "#4b9cd3",
                            "S": "#9e77d8",
                            "V": "#e07b91",
                            "F": "#f2b134",
                            "Q": "#6c757d"
                        },
                        title=f"Distribución de latidos – {record_id}"
                    )
                    fig.update_traces(textposition="outside")
                    fig.update_layout(
                        showlegend=False, height=350,
                        margin=dict(l=20, r=20, t=40, b=20),
                        xaxis_title="Clase AAMI",
                        yaxis_title="Frecuencia de latidos"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No hay datos de anotaciones para graficar la distribución.")

        except Exception as e:
            st.error(f"No se pudo cargar el registro {record_id}. Detalle: {e}")
