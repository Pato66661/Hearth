import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, f1_score

# Carga de modelos
@st.cache_resource
def load_models():
    model_completo = joblib.load("modelo_random_1.pkl")
    model_reducido = joblib.load("modelo_random_reducido.pkl")
    return model_completo, model_reducido

# Carga de dataset
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv")

model_completo, model_reducido = load_models()
df = load_data()

st.set_page_config(page_title="Predicción Cardíaca", layout="wide")
st.title("Análisis de Riesgo Cardíaco")
st.caption("Complete el formulario y explore los resultados del modelo")

# Selección de modelo
modelo_opcion = st.radio("Seleccione el modelo a utilizar:", ["Modelo completo", "Modelo reducido"])

# Formulario del paciente
with st.expander("📝 Formulario del Paciente", expanded=True):
    with st.form("form_paciente"):
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
    features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    try:
        if modelo_opcion == "Modelo completo":
            prediction = model_completo.predict(features)[0]
            proba = model_completo.predict_proba(features)[0][1] * 100
        else:
            reduced_features = features[:, [2, 12, 7, 9, 11, 4, 0]]
            prediction = model_reducido.predict(reduced_features)[0]
            proba = model_reducido.predict_proba(reduced_features)[0][1] * 100

        st.subheader("🔎 Resultado del Análisis")
        if proba >= 80:
            st.warning(f"Riesgo Elevado — Probabilidad: {proba:.1f}%")
            st.markdown("**Recomendaciones:**\n- Consulta con cardiólogo\n- Exámenes clínicos\n- Estilo de vida saludable")
        else:
            st.success(f"Riesgo Bajo — Probabilidad: {proba:.1f}%")
            st.markdown("**Consejos:**\n- Mantén controles regulares\n- Alimentación y actividad física equilibradas")

        st.progress(int(proba), text=f"Probabilidad estimada: {proba:.1f}%")
    except Exception as e:
        st.error(f"Ocurrió un error: {str(e)}")

# Visualizaciones interactivas
st.markdown("---")
st.markdown("## 📊 Visualización de Modelos y Datos")

tabs = st.tabs(["📈 Exploración", "⭐ Importancia", "🔥 Confusión", "📊 Métricas"])

with tabs[0]:
    st.subheader("Distribución de la variable objetivo")
    fig1, ax1 = plt.subplots(figsize=(9, 3))
    df["target"].value_counts().plot(kind="bar", color=["salmon", "lightgreen"], ax=ax1)
    ax1.set_xticklabels(["Sin enfermedad", "Con enfermedad"], rotation=0)
    st.pyplot(fig1)

    st.subheader("Histogramas")
    st.bar_chart(df.select_dtypes(include=np.number).iloc[:, :6])

    st.subheader("Matriz de correlaciones")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)

with tabs[1]:
    st.subheader("Importancia de Características")
    if modelo_opcion == "Modelo completo":
        importancias = model_completo.feature_importances_
        nombres = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                   'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    else:
        importancias = model_reducido.feature_importances_
        nombres = ['cp', 'thal', 'thalach', 'oldpeak', 'ca', 'chol', 'age']
    imp_df = pd.DataFrame({"Variable": nombres, "Importancia": importancias}).sort_values(by="Importancia")
    fig3, ax3 = plt.subplots()
    ax3.barh(imp_df["Variable"], imp_df["Importancia"], color="steelblue")
    st.pyplot(fig3)

with tabs[2]:
    st.subheader("Matriz de Confusión")
    if modelo_opcion == "Modelo completo":
        X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
        y_pred = model_completo.predict(X)
    else:
        X = df[['cp', 'thal', 'thalach', 'oldpeak', 'ca', 'chol', 'age']]
        y_pred = model_reducido.predict(X)

    y_true = df["target"]
    y_true = y_true.astype(int)
import numbers

# Convertir a array plano si es necesario
y_pred = np.array(y_pred).flatten()

# Asegurar que todos los valores son numéricos y válidos
if np.all([isinstance(val, numbers.Number) for val in y_pred]):
    y_pred = pd.Series(y_pred).fillna(0).astype(int)
else:
    st.error("El modelo ha devuelto valores no válidos para la predicción.")
    st.stop()


    cm = confusion_matrix(y_true, y_pred)
    fig4, ax4 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax4)
    ax4.set_xlabel("Predicción")
    ax4.set_ylabel("Real")
    st.pyplot(fig4)

with tabs[3]:
    st.subheader("Métricas Comparativas")
    y_true = df["target"].astype(int)

    try:
        y_pred_full = model_completo.predict(df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                                                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']])
        y_pred_full = np.array(y_pred_full).flatten().astype(int)

        y_pred_red = model_reducido.predict(df[['cp', 'thal', 'thalach', 'oldpeak', 'ca', 'chol', 'age']])
        y_pred_red = np.array(y_pred_red).flatten().astype(int)

        metricas = pd.DataFrame({
            "Modelo Completo": [accuracy_score(y_true, y_pred_full),
                                f1_score(y_true, y_pred_full),
                                roc_auc_score(y_true, y_pred_full)],
            "Modelo Reducido": [accuracy_score(y_true, y_pred_red),
                                f1_score(y_true, y_pred_red),
                                roc_auc_score(y_true, y_pred_red)]
        }, index=["Accuracy", "F1 Score", "AUC"])

        st.dataframe(metricas.style.format("{:.3f}"))

        fig5, ax5 = plt.subplots()
        metricas.T.plot(kind='bar', ax=ax5, rot=0, color=["#1f77b4", "#ff7f0e"])
        ax5.set_ylabel("Valor")
        ax5.set_title("Comparación de métricas entre modelos")
        st.pyplot(fig5)

    except Exception as e:
        st.error("Ocurrió un error al calcular las métricas: " + str(e))

    
    # Visualización de barras comparativas
    fig5, ax5 = plt.subplots()
    metricas.T.plot(kind='bar', ax=ax5, rot=0, color=["#1f77b4", "#4267a4"])
    ax5.set_ylabel("Valor")
    ax5.set_title("Comparación de métricas entre modelos")
    st.pyplot(fig5)