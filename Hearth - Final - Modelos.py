import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score

# Cargar modelos y datos
@st.cache_resource
def load_models():
    return joblib.load("modelo_random_1.pkl"), joblib.load("modelo_random_reducido.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

model_completo, model_reducido = load_models()
df = load_data()

st.set_page_config("Predicción Cardíaca", layout="wide")
st.title("Análisis de Riesgo Cardíaco YACHAY")
st.caption("Complete el formulario para evaluar el riesgo cardíaco")

modelo_opcion = st.radio("Modelo a utilizar:", ["Modelo completo", "Modelo reducido"])

# Formulario
with st.expander("Formulario del Paciente", expanded=True):
    with st.form("form"):
        cols = st.columns(3)
        age = cols[0].slider("Edad", 20, 100, 50)
        sex = cols[0].radio("Sexo", [0, 1], format_func=lambda x: "Mujer" if x == 0 else "Hombre")
        cp = cols[0].selectbox("Dolor Torácico", range(4), format_func=lambda x: ["Típico", "Atípico", "No anginoso", "Asintomático"][x])
        trestbps = cols[0].number_input("Presión Arterial", 80, 200, 120)
        chol = cols[1].number_input("Colesterol", 100, 600, 200)
        fbs = cols[1].radio("Glucosa >120 mg/dl", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        restecg = cols[1].selectbox("ECG", range(3), format_func=lambda x: ["Normal", "ST-T", "Hipertrofia"][x])
        thalach = cols[1].number_input("Frec. Cardíaca Máxima", 60, 220, 150)
        exang = cols[2].radio("Angina por Ejercicio", [0, 1], format_func=lambda x: "No" if x == 0 else "Sí")
        oldpeak = cols[2].slider("Depresión ST", 0.0, 6.2, 1.0, 0.1)
        slope = cols[2].selectbox("Pendiente ST", range(3), format_func=lambda x: ["Ascendente", "Plano", "Descendente"][x])
        ca = cols[2].slider("Vasos Coloreados", 0, 3, 0)
        thal = cols[2].selectbox("Talasemia", range(4), format_func=lambda x: ["Normal", "Defecto fijo", "Defecto reversible", "Otro"][x])
        submit = st.form_submit_button("Analizar")

if submit:
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    try:
        if modelo_opcion == "Modelo completo":
            pred = model_completo.predict(features)[0]
            proba = model_completo.predict_proba(features)[0][1] * 100
        else:
            idx = [2, 12, 7, 9, 11, 4, 0]
            pred = model_reducido.predict(features[:, idx])[0]
            proba = model_reducido.predict_proba(features[:, idx])[0][1] * 100

        st.subheader("Resultado del Análisis")
        st.warning(f"🚨 Riesgo Elevado — {proba:.1f}%") if proba >= 80 else st.success(f"✅ Riesgo Bajo — {proba:.1f}%")
        st.progress(int(proba), text=f"Probabilidad estimada: {proba:.1f}%")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Visualización
st.markdown("---")
tabs = st.tabs(["Exploración", "Importancia", "Confusión", "Métricas"])

with tabs[0]:
    st.subheader("Distribución de la variable objetivo")
    st.bar_chart(df['target'].value_counts())

    st.subheader("Variables Numéricas")
    st.bar_chart(df.select_dtypes(np.number).iloc[:, :6])

    st.subheader("Matriz de Correlación")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with tabs[1]:
    nombres = ['cp', 'thal', 'thalach', 'oldpeak', 'ca', 'chol', 'age'] if modelo_opcion == "Modelo reducido" else df.columns[:-1]
    importancias = model_reducido.feature_importances_ if modelo_opcion == "Modelo reducido" else model_completo.feature_importances_
    imp_df = pd.DataFrame({"Variable": nombres, "Importancia": importancias}).sort_values("Importancia")
    fig, ax = plt.subplots()
    ax.barh(imp_df["Variable"], imp_df["Importancia"], color="steelblue")
    ax.set_title("Importancia de Variables")
    st.pyplot(fig)

with tabs[2]:
    X = df[nombres]
    y_true = df["target"]
    y_pred = model_reducido.predict(X) if modelo_opcion == "Modelo reducido" else model_completo.predict(X)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusión")
    st.pyplot(fig)

with tabs[3]:
    st.subheader("Comparación de Métricas")
    y_true = df["target"]
    y_pred_full = model_completo.predict(df[df.columns[:-1]])
    y_pred_red = model_reducido.predict(df[['cp', 'thal', 'thalach', 'oldpeak', 'ca', 'chol', 'age']])
    metricas = pd.DataFrame({
        "Modelo Completo": [accuracy_score(y_true, y_pred_full), f1_score(y_true, y_pred_full), roc_auc_score(y_true, y_pred_full)],
        "Modelo Reducido": [accuracy_score(y_true, y_pred_red), f1_score(y_true, y_pred_red), roc_auc_score(y_true, y_pred_red)]
    }, index=["Accuracy", "F1 Score", "AUC"])
    st.dataframe(metricas.style.format("{:.3f}"))
    fig, ax = plt.subplots()
    metricas.T.plot(kind='bar', ax=ax, rot=0)
    ax.set_title("Comparación de Métricas entre Modelos")
    ax.set_ylabel("Valor")
    st.pyplot(fig)
