import streamlit as st
from streamlit_option_menu import option_menu
from informe_tecnico import InformeTecnico
from informe_clinico import InformeClinico
from home import Home

def show_pagina_inicial():
    with st.sidebar:
        # ============================
        # MENÚ PRINCIPAL
        # ============================
        selected = option_menu(
            menu_title="Menú",
            options=["Inicio", "Informe Tecnico", "Informe Clinico"],
            icons=["house-heart-fill", "cpu-fill", "heart-pulse-fill"],
            menu_icon="menu-button-wide",
            default_index=0
        )


        # ============================
        # BOTÓN DE CERRAR SESIÓN
        # ============================
        st.markdown("---")  # línea divisoria
        logout = st.button("Cerrar sesión", use_container_width=True)
        if logout:
            st.session_state.logged_in = False
            st.session_state.user = None
            st.rerun()

    # ============================
    # CONTENIDO PRINCIPAL
    # ============================
    if selected == "Inicio":
        pagina = Home()
        pagina.render()
    elif selected == "Informe Tecnico":
        pagina = InformeTecnico()
        pagina.render()
    elif selected == "Informe Clinico":
        pagina = InformeClinico()
        pagina.render()
    else:
        st.write("Opción no válida")
