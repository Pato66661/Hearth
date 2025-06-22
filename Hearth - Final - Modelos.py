import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_models():
    model_completo = joblib.load("modelo_random_1.pkl")
    model_reducido = joblib.load("modelo_reducido.pkl")
    return model_completo, model_reducido

model_completo, model_reducido = load_models()

st.set_page_config(
    page_title="Predicción Cardíaca",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Análisis de Riesgo Cardíaco")
st.caption("Complete el formulario para evaluar el riesgo de enfermedad cardiovascular")

modelo_opcion = st.radio(
    "Seleccione el modelo a utilizar",
    ["Modelo completo", "Modelo reducido (7 variables)"]
)

with st.expander("Formulario del Paciente", expanded=True):
    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Edad", 20, 100, 50)
            sex = st.radio("Sexo", [("Mujer", 0), ("Hombre", 1)], format_func=lambda x: x[0])[1]
            cp = st.selectbox("Tipo de Dolor Torácico", [0, 1, 2, 3], format_func=lambda x: ["Típico", "Atípico", "No anginoso", "Asintomático"][x])
            trestbps = st.number_input("Presión Arterial (mm Hg)", 80, 200, 120)
        
        with col2:
            chol = st.number_input("Colesterol (mg/dl)", 100, 600, 200)
            fbs = st.radio("Glucosa > 120 mg/dl", [("No", 0), ("Sí", 1)], format_func=lambda x: x[0])[1]
            restecg = st.selectbox("Electrocardiograma", [0, 1, 2], format_func=lambda x: ["Normal", "ST-T anormal", "Hipertrofia ventricular"][x])
            thalach = st.number_input("Frecuencia Cardíaca Máxima", 60, 220, 150)
        
        with col3:
            exang = st.radio("Angina por Ejercicio", [("No", 0), ("Sí", 1)], format_func=lambda x: x[0])[1]
            oldpeak = st.slider("Depresión ST", 0.0, 6.2, 1.0, step=0.1)
            slope = st.selectbox("Pendiente del ST", [0, 1, 2], format_func=lambda x: ["Ascendente", "Plano", "Descendente"][x])
            ca = st.slider("Vasos Coloreados", 0, 3, 0)
            thal = st.selectbox("Talasemia", [0, 1, 2, 3], format_func=lambda x: ["Normal", "Defecto fijo", "Defecto reversible", "Otro"][x])

        submit = st.form_submit_button("Analizar")

if submit:
    try:
        all_features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                  thalach, exang, oldpeak, slope, ca, thal]])

        if modelo_opcion == "Modelo completo":
            prediction = model_completo.predict(all_features)[0]
            proba = model_completo.predict_proba(all_features)[0][1] * 100
        else:
            # Índices para el modelo reducido: cp, thal, thalach, oldpeak, ca, chol, age
            reduced_features = all_features[:, [2, 12, 7, 9, 11, 4, 0]]
            prediction = model_reducido.predict(reduced_features)[0]
            proba = model_reducido.predict_proba(reduced_features)[0][1] * 100

        st.subheader("Resultado del Análisis")
        if prediction == 1:
            st.warning(f"Riesgo Elevado — Probabilidad: {proba:.1f}%")
            st.markdown("""
            **Recomendaciones:**
            - Consulta con cardiólogo
            - Análisis clínicos
            - Monitoreo de presión y hábitos saludables
            """)
        else:
            st.success(f"Riesgo Bajo — Probabilidad: {proba:.1f}%")
            st.markdown("""
            **Consejos Preventivos:**
            - Mantener controles regulares
            - Estilo de vida saludable
            """)

        st.progress(int(proba), text=f"Probabilidad estimada: {proba:.1f}%")

    except Exception as e:
        st.error(f"Ocurrió un error: {str(e)}")

st.markdown("---")
st.markdown("""
Este sistema es solo orientativo. Consulte siempre con un profesional de salud.
""")
