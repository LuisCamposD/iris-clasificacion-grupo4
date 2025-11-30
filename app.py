import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris

st.set_page_config(page_title="Clasificación Iris", layout="wide")

# ---------- Cargar modelos y datos ----------
@st.cache_resource
def cargar_modelos():
    knn = joblib.load("modelo_iris_knn.pkl")
    svm = joblib.load("modelo_iris_svm.pkl")
    arbol = joblib.load("modelo_iris_arbol.pkl")
    return knn, svm, arbol

@st.cache_data
def cargar_datos():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    df["target_name"] = df["target"].apply(lambda i: iris.target_names[i])
    return iris, df

knn, svm, arbol = cargar_modelos()
iris, df_iris = cargar_datos()

# ---------- Sidebar ----------
with st.sidebar:
    st.title("Aplicación Iris")
    pagina = st.selectbox("Sección", ["Glosario", "Dataset", "Predicciones"])

# ---------- Glosario ----------
if pagina == "Glosario":
    st.title("Glosario 🍀")
    st.markdown("""
    1. **KNN (K-Nearest Neighbors)**: clasifica según los vecinos más cercanos.  
    2. **SVM (Support Vector Machine)**: encuentra un hiperplano óptimo para separar clases.  
    3. **Árbol de decisión**: modelo basado en preguntas tipo árbol sobre las variables.  
    4. **IRIS**: dataset de flores de iris (150 muestras, 4 características, 3 clases).  
    5. **Joblib**: permite guardar y cargar modelos entrenados.  
    6. **Scikit-learn**: librería de Python para machine learning.
    """)

# ---------- Dataset ----------
elif pagina == "Dataset":
    st.title("Dataset Iris 🌸")
    st.write("Primeras filas del dataset:")
    st.dataframe(df_iris.head())

    st.write("Distribución de clases:")
    st.bar_chart(df_iris["target_name"].value_counts())

# ---------- Predicciones ----------
elif pagina == "Predicciones":
    st.title("Predicciones 🔮")

    st.write("Ajusta las características de la flor:")

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.number_input("Sepal length (cm)", 4.0, 8.0, 5.9, step=0.1)
        sepal_width  = st.number_input("Sepal width (cm)",  2.0, 4.5, 3.0, step=0.1)

    with col2:
        petal_length = st.number_input("Petal length (cm)", 1.0, 7.0, 5.0, step=0.1)
        petal_width  = st.number_input("Petal width (cm)",  0.1, 2.5, 1.8, step=0.1)

    X_nuevo = [[sepal_length, sepal_width, petal_length, petal_width]]

    modelo_nombre = st.selectbox(
        "Modelo a usar",
        ["KNN", "SVM", "Árbol de decisión"]
    )

    if st.button("Predecir"):
        # Elegir modelo
        if modelo_nombre == "KNN":
            modelo = knn
        elif modelo_nombre == "SVM":
            modelo = svm
        else:
            modelo = arbol

        # Predicción principal
        pred = modelo.predict(X_nuevo)[0]
        especie = iris.target_names[pred]
        st.success(f"✅ Predicción: **{especie}**")

        # Probabilidades por clase (solo si el modelo las soporta)
        if hasattr(modelo, "predict_proba"):
            proba = modelo.predict_proba(X_nuevo)[0]
            class_indices = modelo.classes_
            class_names = [iris.target_names[i] for i in class_indices]

            proba_df = pd.DataFrame({
                "Clase": class_names,
                "Probabilidad": proba
            }).set_index("Clase")

            st.write("Probabilidades por clase:")
            st.bar_chart(proba_df["Probabilidad"])
        else:
            st.info("Este modelo no entrega probabilidades (`predict_proba`).")
