# CÓDIGO STREAMLIT - Clasificación: Hogares pobres / no pobres (ENAHO 2025)
# Activar el ambiente virtual:   .\.venv\Scripts\activate
# Ejecutar:   streamlit run app_enaho.py --server.port 8512

import streamlit as st
import pandas as pd
from joblib import load

st.set_page_config(page_title="Hogares Pobres ENAHO 2025", layout="wide")

# Cargar el modelo (GridSearchCV con pipeline XGBoost + SMOTETomek) y el codificador
modelo = load('modelo_BOOSTING_tunning.joblib')
le = load('label_encoder.joblib')

st.title("🏠 Clasificación de hogares: ¿Pobre o No pobre?")
st.markdown(
    "Modelo: **XGBoost + SMOTETomek** (GridSearchCV) · Dataset **ENAHO 2025 (INEI)** · \n\n"
    "Desarrollado por: \n\n"
    "* Wilbert Saúl Tijero Fuentes\n\n"
    "* Andrea Del Pilar Rivera Rodríguez\n\n"
    "Accuracy test: **79.5%** · AUC: **0.799**"
)

c1, c2 = st.columns(2)
mieperho = c1.number_input("Miembros del hogar (mieperho)", min_value=1, max_value=15, value=3, step=1)
percepho = c2.number_input("Perceptores de ingreso (percepho)", min_value=0, max_value=9, value=2, step=1)

c3, c4 = st.columns(2)
g_dominio = c3.selectbox("Dominio geográfico (g_dominio)",
                         ["Costa", "Sierra", "Selva", "Lima Metropolitana"])
area = c4.selectbox("Área (area)", ["Urbano", "Rural"])

c5, c6 = st.columns(2)
educ = c5.selectbox("Nivel educativo del jefe (educ_jefe_cat)",
                    ["Sin nivel/inicial", "Primaria", "Secundaria",
                     "Superior no universitaria", "Superior universitaria", "Básica especial"])
luz = c6.selectbox("Acceso a luz (acceso_luz)", ["Sí", "No"])

c7, c8 = st.columns(2)
desague = c7.selectbox("Acceso a desagüe (acceso_desague)", ["Conectado", "No conectado"])
agua = c8.selectbox("Acceso a agua (acceso_agua)", ["Red pública", "Otro medio"])

if st.button("Predecir", type="primary"):
    obs = pd.DataFrame({
        "mieperho": [float(mieperho)],
        "percepho": [float(percepho)],
        "g_dominio": [g_dominio],
        "area": [area],
        "educ_jefe_cat": [educ],
        "acceso_luz": [luz],
        "acceso_desague": [desague],
        "acceso_agua": [agua],
    })

    pred_num = modelo.predict(obs)[0]
    clase = le.inverse_transform([pred_num])[0]
    prob_pobre = modelo.predict_proba(obs)[0, 1]

    color = "#7a2e1d" if clase == "Pobre" else "#155c2e"
    fondo = "#fdece8" if clase == "Pobre" else "#e6f4ea"
    st.markdown(
        f'<p style="font-size:34px;color:{color};background:{fondo};border:2px solid {color};'
        f'border-radius:14px;padding:18px 22px;text-align:center;">'
        f'Clasificación: <b>{clase}</b><br>'
        f'<span style="font-size:22px;">Probabilidad de ser pobre: {prob_pobre:.1%}</span></p>',
        unsafe_allow_html=True
    )
else:
    st.info("Completa los campos y presiona **Predecir**.")