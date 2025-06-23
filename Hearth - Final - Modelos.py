import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, f1_score
from sklearn.utils.multiclass import type_of_target
from tensorflow.keras.models import load_model

# Cargar modelos
@st.cache_resource
def load_all_models():
    rf_full = joblib.load("modelo_random_1.pkl")
    rf_reduced = joblib.load("modelo_random_reducido.pkl")
    rn_full = load_model("modeloRN.h5")
    rn_reduced = load_model("modeloRN_Reduced.h5")
    return rf_full, rf_reduced, rn_full, rn_reduced

# Cargar datos
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    if df['target'].dtype == object:
        df['target'] = df['target'].map({"No Disease": 0, "Disease": 1})
    return df

# Inicializar
model_completo, model_reducido, modelo_rn, modelo_rn_reducido = load_all_models()
df = load_data()

st.set_page_config("Predicción Cardíaca", layout="wide")
st.title("Análisis de Riesgo Cardíaco")
st.caption("Complete el formulario para evaluar el riesgo cardíaco")

modelo_opcion = st.radio("Modelo a utilizar:", [
    "Random Forest Completo", 
    "Random Forest Reducido",
    "Red Neuronal Completa",
    "Red Neuronal Reducida"
])

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
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    idx = [2, 12, 7, 9, 11, 4, 0]
    try:
        if modelo_opcion == "Random Forest Completo":
            pred = model_completo.predict(input_data)[0]
            proba = model_completo.predict_proba(input_data)[0][1] * 100
        elif modelo_opcion == "Random Forest Reducido":
            pred = model_reducido.predict(input_data[:, idx])[0]
            proba = model_reducido.predict_proba(input_data[:, idx])[0][1] * 100
        elif modelo_opcion == "Red Neuronal Completa":
            proba = modelo_rn.predict(input_data)[0][0] * 100
            pred = int(proba >= 50)
        else:
            proba = modelo_rn_reducido.predict(input_data[:, idx])[0][0] * 100
            pred = int(proba >= 50)

        st.subheader("Resultado del Análisis")
        if proba >= 80:
            st.warning(f"🚨 Riesgo Elevado — {proba:.1f}%")
        else:
            st.success(f"✅ Riesgo Bajo — {proba:.1f}%")
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
    nombres = ['cp', 'thal', 'thalach', 'oldpeak', 'ca', 'chol', 'age'] if "Reducido" in modelo_opcion else df.columns[:-1]
    importancias = model_reducido.feature_importances_ if modelo_opcion == "Random Forest Reducido" else model_completo.feature_importances_
    if "Red Neuronal" not in modelo_opcion:
        imp_df = pd.DataFrame({"Variable": nombres, "Importancia": importancias}).sort_values("Importancia")
        fig, ax = plt.subplots()
        ax.barh(imp_df["Variable"], imp_df["Importancia"], color="steelblue")
        ax.set_title("Importancia de Variables")
        st.pyplot(fig)
    else:
        st.info("La red neuronal no proporciona importancias de variables directamente.")

with tabs[2]:
    X = df[nombres]
    y_true = df["target"].astype(int).values
    if modelo_opcion == "Random Forest Reducido":
        y_pred = model_reducido.predict(X)
    elif modelo_opcion == "Random Forest Completo":
        y_pred = model_completo.predict(X)
    elif modelo_opcion == "Red Neuronal Completa":
        y_pred = (modelo_rn.predict(X) >= 0.5).astype(int)
    else:
        y_pred = (modelo_rn_reducido.predict(X) >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusión")
    st.pyplot(fig)

with tabs[3]:
    st.subheader("Comparación General de Modelos")
try:
    resultados = pd.DataFrame(columns=["Modelo", "Accuracy", "F1 Score", "AUC"])

    X_full = df[df.columns[:-1]].values
    X_reduc = df[['cp', 'thal', 'thalach', 'oldpeak', 'ca', 'chol', 'age']].values
    y_true = df["target"].astype(int).values

    # Random Forest Completo
    y_pred_rf = model_completo.predict(X_full)
    prob_rf = model_completo.predict_proba(X_full)[:, 1]
    resultados.loc[len(resultados)] = [
        "Random Forest Completo",
        accuracy_score(y_true, y_pred_rf),
        f1_score(y_true, y_pred_rf),
        roc_auc_score(y_true, prob_rf)
    ]

    # Random Forest Reducido
    y_pred_rf_red = model_reducido.predict(X_reduc)
    prob_rf_red = model_reducido.predict_proba(X_reduc)[:, 1]
    resultados.loc[len(resultados)] = [
        "Random Forest Reducido",
        accuracy_score(y_true, y_pred_rf_red),
        f1_score(y_true, y_pred_rf_red),
        roc_auc_score(y_true, prob_rf_red)
    ]

    # Red Neuronal Completa
    prob_rn = modelo_rn.predict(X_full).flatten()
    y_pred_rn = (prob_rn >= 0.5).astype(int)
    resultados.loc[len(resultados)] = [
        "Red Neuronal Completa",
        accuracy_score(y_true, y_pred_rn),
        f1_score(y_true, y_pred_rn),
        roc_auc_score(y_true, prob_rn)
    ]

    # Red Neuronal Reducida
    prob_rn_red = modelo_rn_reducido.predict(X_reduc).flatten()
    y_pred_rn_red = (prob_rn_red >= 0.5).astype(int)
    resultados.loc[len(resultados)] = [
        "Red Neuronal Reducida",
        accuracy_score(y_true, y_pred_rn_red),
        f1_score(y_true, y_pred_rn_red),
        roc_auc_score(y_true, prob_rn_red)
    ]

    st.dataframe(resultados.set_index("Modelo").style.format("{:.3f}"))

    fig, ax = plt.subplots(figsize=(8, 4))
    resultados.plot(kind="barh", x="Modelo", ax=ax)
    ax.set_title("Comparación de Accuracy, F1 y AUC por Modelo")
    ax.set_xlabel("Valor")
    ax.legend(loc="lower right")
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error al calcular métricas comparativas: {e}")