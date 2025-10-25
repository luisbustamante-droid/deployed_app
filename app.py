import logging, sys
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
print("=== Iniciando app.py ===")

import streamlit as st
try:
    from pagina_inicial import show_pagina_inicial
    print("Import de pagina_inicial OK")
except Exception as e:
    import traceback
    traceback.print_exc()
    import streamlit as st
    st.error(f"Error al importar pagina_inicial: {e}")
    raise e

import os
os.environ["STREAMLIT_WATCHDOG_TYPE"] = "polling"
os.environ["STREAMLIT_WATCHDOG"] = "false"


# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="Arrhythmia Spectrograms", layout="wide")

# ==============================
# CSS limpio (sin card)
# ==============================
st.markdown(r"""
<style>
h1 {
    margin-top: 1rem !important;  /* aumenta o reduce según tu gusto */
}
.block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 0.5rem !important;
}
.stApp { background-color: #ffffff; font-family: 'Segoe UI', sans-serif; }

/* Elimina borde, fondo y sombra de la caja */
.login-title { font-size: 1.6rem; font-weight: 600; color: #333; margin: 0.4rem 0; }
.subtitle { font-size: 0.95rem; color: #666; margin-bottom: 1.2rem; }

/* Inputs */
label, .stTextInput>div>div>input { color: #333 !important; }

/* Botón */
div.stButton > button {
    background-color: #7C1C27; color: #fff; font-weight: 500;
    border-radius: 8px; border: none; padding: 0.6rem 0; width: 100%;
    transition: all 0.25s ease;
}
div.stButton > button:hover { background-color: #5e121b; }

.stSuccess, .stError { font-size: 0.9rem; }
.element-container { padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ==============================
# USUARIOS (demo)
# ==============================
USERS = {"user": "password",}
def login(username, password): return USERS.get(username) == password

# ==============================
# SESIÓN
# ==============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user = None

# ==============================
# UI
# ==============================
if not st.session_state.logged_in:
    st.markdown("<div style='height:12vh'></div>", unsafe_allow_html=True)
    left, mid, right = st.columns([1, 1.2, 1])

    with mid:
        st.image("assets/uees_logo.png", width=160)
        st.markdown("<div class='login-title'>Iniciar Sesion</div>", unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            submit = st.form_submit_button("Entrar")

        if submit:
            if login(username.strip(), password):
                st.session_state.logged_in = True
                st.session_state.user = username.strip()
                st.success(f"Bienvenido(a), {st.session_state.user}")
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos.")

    st.markdown("<div style='height:10vh'></div>", unsafe_allow_html=True)
else:
    show_pagina_inicial()
