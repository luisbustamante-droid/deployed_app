import streamlit as st
from helpers.dataset_info import render_dataset_section

class Home:
    def __init__(self):
        pass

    def render(self):
        # ===============================================================
        # ENCABEZADO PRINCIPAL
        # ===============================================================
        st.title("Clasificación explicable de arritmias cardíacas mediante espectrogramas de ECG")

        st.markdown("""
        Este sistema forma parte de un proyecto de investigación orientado a la 
        **clasificación automática de arritmias cardíacas** utilizando **redes neuronales convolucionales (CNNs)**
        entrenadas sobre **espectrogramas derivados de señales ECG** de las bases **MIT-BIH** y **SVDB**.

        A través de esta interfaz podrás:
        - **Visualizar resultados técnicos** del entrenamiento (métricas, curvas, Grad-CAMs, etc.).
        - **Explorar interpretaciones clínicas** sobre las activaciones de la red.
        - **Comparar arquitecturas CNN** (ResNet, MobileNet, EfficientNet) y su rendimiento por clase AAMI.
        """)

        st.divider()

        # ===============================================================
        # SECCIONES DISPONIBLES
        # ===============================================================
        st.markdown("### Secciones principales")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Inicio**  \nResumen general y objetivos del proyecto.")
        with col2:
            st.markdown("**Informe técnico**  \nEntrenamiento, métricas y resultados experimentales.")
        with col3:
            st.markdown("**Informe clínico**  \nVisualizaciones interpretables (Grad-CAM) y análisis fisiológico.")

        st.divider()

        # ===============================================================
        # SECCIÓN DE DATASETS
        # ===============================================================
        render_dataset_section()
