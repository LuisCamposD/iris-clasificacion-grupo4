import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# ------------------ CONFIGURACIÓN GENERAL ------------------
st.set_page_config(
    page_title="Clasificación Iris",
    page_icon="🌸",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.0rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1.0rem;
        color: #aaaaaa;
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ CARGA DE MODELOS Y DATOS ------------------
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

MODELOS = {
    "KNN": knn,
    "SVM": svm,
    "Árbol de decisión": arbol
}

# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.title("Aplicación Iris")
    pagina = st.selectbox(
        "Sección",
        ["Glosario", "Dataset", "Modelos y desempeño", "Predicciones"]
    )
    st.markdown("---")
    st.caption("Desarrollado por Luis Campos 💻")


# ------------------ GLOSARIO ------------------
if pagina == "Glosario":
    st.markdown('<div class="main-title">Glosario 🌱</div>', unsafe_allow_html=True)
    st.markdown(
        """
        1. **IRIS**  
           Dataset clásico de *Machine Learning* con 150 flores de iris,  
           4 características (largo/ancho de sépalo y pétalo) y 3 especies.

        2. **KNN (K-Nearest Neighbors)**  
           Clasifica una muestra nueva según las clases de sus vecinos más cercanos.

        3. **SVM (Support Vector Machine)**  
           Encuentra el hiperplano que mejor separa las clases en el espacio de características.

        4. **Árbol de decisión**  
           Modelo basado en preguntas tipo árbol sobre las variables (¿petal length > X?).

        5. **Accuracy**  
           Porcentaje de predicciones correctas sobre el total de muestras.

        6. **Matriz de confusión**  
           Tabla que muestra cuántas muestras de cada clase se clasifican bien o mal.
        """
    )


# ------------------ DATASET ------------------
elif pagina == "Dataset":
    st.markdown('<div class="main-title">Dataset Iris 🌸</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Exploración rápida de los datos originales</div>',
        unsafe_allow_html=True
    )

    st.subheader("Vista general")
    st.dataframe(df_iris.head())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribución de clases")
        st.bar_chart(df_iris["target_name"].value_counts())

    with col2:
        st.subheader("Estadísticos descriptivos")
        st.dataframe(df_iris[iris.feature_names].describe().T)


# ------------------ MODELOS Y DESEMPEÑO ------------------
elif pagina == "Modelos y desempeño":
    st.markdown('<div class="main-title">Modelos y desempeño 🧠</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Compara cómo se comporta cada algoritmo en el dataset Iris</div>',
        unsafe_allow_html=True
    )

    X = df_iris[iris.feature_names]
    y = df_iris["target"]

    modelo_nombre = st.selectbox("Selecciona un modelo", list(MODELOS.keys()))
    modelo = MODELOS[modelo_nombre]

    # Predicciones y accuracy
    y_pred = modelo.predict(X)
    acc = accuracy_score(y, y_pred)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy en Iris", f"{acc:.3f}")
    with col2:
        st.metric("Nº de muestras", len(y))
    with col3:
        st.metric("Nº de clases", len(iris.target_names))

    st.markdown("---")

    # Matriz de confusión
    st.subheader("Matriz de confusión")
    cm = confusion_matrix(y, y_pred, labels=modelo.classes_)
    etiquetas = [iris.target_names[i] for i in modelo.classes_]

    fig, ax = plt.subplots()
    ax.imshow(cm)

    ax.set_xticks(range(len(etiquetas)))
    ax.set_yticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=45, ha="right")
    ax.set_yticklabels(etiquetas)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)

    st.caption(
        "Diagonal = aciertos. Valores fuera de la diagonal = errores de clasificación."
    )


# ------------------ PREDICCIONES ------------------
elif pagina == "Predicciones":
    st.markdown('<div class="main-title">Predicciones en vivo 🔮</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Ajusta las características y mira qué predice cada modelo</div>',
        unsafe_allow_html=True
    )

    st.write("Ingresa las características de la flor:")

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input("Sepal length (cm)", 4.0, 8.0, 5.9, step=0.1)
        sepal_width  = st.number_input("Sepal width (cm)",  2.0, 4.5, 3.0, step=0.1)
    with col2:
        petal_length = st.number_input("Petal length (cm)", 1.0, 7.0, 5.0, step=0.1)
        petal_width  = st.number_input("Petal width (cm)",  0.1, 2.5, 1.8, step=0.1)

    X_nuevo = [[sepal_length, sepal_width, petal_length, petal_width]]

    modelo_nombre = st.selectbox(
        "Modelo principal para la explicación",
        list(MODELOS.keys())
    )

    if st.button("Predecir"):
        modelo = MODELOS[modelo_nombre]

        # Predicción principal
        pred = modelo.predict(X_nuevo)[0]
        especie = iris.target_names[pred]
        st.success(f"✅ Predicción ({modelo_nombre}): **{especie}**")

        # Probabilidades del modelo principal
        if hasattr(modelo, "predict_proba"):
            proba = modelo.predict_proba(X_nuevo)[0]
            class_indices = modelo.classes_
            class_names = [iris.target_names[i] for i in class_indices]

            proba_df = pd.DataFrame({
                "Clase": class_names,
                "Probabilidad": proba
            }).set_index("Clase")

            st.write("Probabilidades por clase (modelo seleccionado):")
            st.bar_chart(proba_df["Probabilidad"])
        else:
            st.info(f"El modelo **{modelo_nombre}** no entrega probabilidades (`predict_proba`).")

        st.markdown("---")

        # Comparación de modelos
        st.subheader("Comparación de los 3 modelos")

        filas = []
        for nombre, m in MODELOS.items():
            pred_m = m.predict(X_nuevo)[0]
            especie_m = iris.target_names[pred_m]

            if hasattr(m, "predict_proba"):
                proba_m = m.predict_proba(X_nuevo)[0]
                proba_clase = max(proba_m)   # prob. de la clase predicha
            else:
                proba_clase = None

            filas.append({
                "Modelo": nombre,
                "Especie predicha": especie_m,
                "Probabilidad máx.": f"{proba_clase:.3f}" if proba_clase is not None else "N/A"
            })

        resultados_df = pd.DataFrame(filas)
        st.dataframe(resultados_df, hide_index=True)

        st.caption(
            "Así puedes ver cuándo los modelos coinciden y cuándo discrepan para la misma flor."
        )
