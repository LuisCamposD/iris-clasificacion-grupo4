import streamlit as st
import pandas as pd
import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
import altair as alt

# ------------------ CONFIGURACIÓN GENERAL ------------------
st.set_page_config(
    page_title="Predicción del dataset Iris",
    page_icon="🌸",
    layout="wide",
)

# Estilos básicos para títulos tipo “landing”
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1.0rem;
        color: #666666;
        margin-bottom: 1.5rem;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
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
    # Puedes cambiar esta URL por el logo de tu grupo / ISIL si lo subes a internet
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/4/4f/Iris_versicolor_3.jpg",
        use_column_width=True,
    )
    st.markdown("**Aplicación de Modelo de Clasificación**")
    st.caption("Despliegue en Streamlit – Dataset Iris")

    pagina = st.selectbox(
        "Selecciona la sección:",
        [
            "Introducción",
            "Dataset",
            "Glosario",
            "Modelos y desempeño",
            "Predicciones",
        ],
    )

    st.markdown("---")
    st.caption("Desarrollado por **Luis Campos** 💻")


# ------------------ INTRODUCCIÓN ------------------
if pagina == "Introducción":
    st.markdown(
        '<div class="main-title">Predicción del dataset Iris 🍀</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Aplicación web para comparar modelos de clasificación (KNN, SVM y Árbol de decisión)</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.info("**Autor:** Luis Campos\n\n**Curso:** Inteligencia Artificial\n**Tema:** Clasificación supervisada con el dataset Iris")

        st.markdown(
            """
            El **dataset Iris** es uno de los conjuntos de datos más conocidos en
            estadística y *machine learning*. Fue introducido por el botánico y estadístico
            **Ronald A. Fisher** en 1936, con el objetivo de demostrar cómo usar
            mediciones morfológicas para clasificar especies de plantas.

            El dataset contiene **150 muestras de flores de iris**, divididas en tres especies:

            - *Iris setosa*  
            - *Iris versicolor*  
            - *Iris virginica*  

            Cada flor se describe con 4 características numéricas:

            - Largo del sépalo (*sepal length*)  
            - Ancho del sépalo (*sepal width*)  
            - Largo del pétalo (*petal length*)  
            - Ancho del pétalo (*petal width*)

            En esta aplicación podrás:

            1. Explorar el dataset.  
            2. Revisar un glosario de conceptos clave.  
            3. Comparar el desempeño de tres modelos de clasificación.  
            4. Probar predicciones en vivo modificando las características de una flor.
            """
        )

    with col2:
        st.image(
            "https://miro.medium.com/v2/resize:fit:720/format:webp/1*d1W_3ayT8CXOOWjK-UQPlw.png",
            caption="Ejemplo de mediciones en el Iris (sépalo y pétalo).",
            use_column_width=True,
        )


# ------------------ DATASET ------------------
elif pagina == "Dataset":
    st.markdown(
        '<div class="main-title">Exploración del dataset Iris 🌸</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Vista rápida de las muestras y sus características</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Primeras filas</div>', unsafe_allow_html=True)
    st.dataframe(df_iris.head())

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="section-title">Distribución de clases</div>',
            unsafe_allow_html=True,
        )
        st.bar_chart(df_iris["target_name"].value_counts())

    with col2:
        st.markdown(
            '<div class="section-title">Estadísticos descriptivos</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(df_iris[iris.feature_names].describe().T)


# ------------------ GLOSARIO ------------------
elif pagina == "Glosario":
    st.markdown('<div class="main-title">Glosario 🌱</div>', unsafe_allow_html=True)

    st.markdown(
        """
        **IRIS**  
        Dataset clásico con 150 flores de iris, 4 características numéricas y 3 clases.

        **Características (features)**  
        Variables de entrada que describen a cada flor (largo/ancho de sépalo y pétalo).

        **Target / Etiqueta de clase**  
        Especie de la flor que queremos predecir (*setosa, versicolor, virginica*).

        **KNN (K-Nearest Neighbors)**  
        Clasifica una muestra nueva según las clases de sus vecinos más cercanos.

        **SVM (Support Vector Machine)**  
        Encuentra el hiperplano que mejor separa las clases en el espacio de características.

        **Árbol de decisión**  
        Modelo que toma decisiones en forma de árbol, haciendo preguntas del tipo:
        “¿petal length > 2.5 cm?”.

        **Accuracy**  
        Porcentaje de predicciones correctas sobre el total de muestras.

        **Matriz de confusión**  
        Tabla que muestra cuántas muestras de cada clase se clasifican bien
        y cuántas se confunden con otra clase.
        """
    )


# ------------------ MODELOS Y DESEMPEÑO ------------------
elif pagina == "Modelos y desempeño":
    st.markdown(
        '<div class="main-title">Modelos y desempeño 🧠</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Compara cómo se comporta cada algoritmo en el dataset Iris completo</div>',
        unsafe_allow_html=True,
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

    # Matriz de confusión (tabla + heatmap Altair)
    st.markdown(
        '<div class="section-title">Matriz de confusión</div>', unsafe_allow_html=True
    )

    cm = confusion_matrix(y, y_pred, labels=modelo.classes_)
    etiquetas = [iris.target_names[i] for i in modelo.classes_]

    cm_df = pd.DataFrame(cm, index=etiquetas, columns=etiquetas)
    cm_df.index.name = "Real"
    cm_df.columns.name = "Predicción"

    st.write("Tabla:")
    st.dataframe(cm_df)

    st.write("Heatmap:")
    cm_long = cm_df.reset_index().melt(
        id_vars="Real", var_name="Predicción", value_name="Muestras"
    )

    heatmap = (
        alt.Chart(cm_long)
        .mark_rect()
        .encode(
            x=alt.X("Predicción:N"),
            y=alt.Y("Real:N"),
            color=alt.Color("Muestras:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["Real", "Predicción", "Muestras"],
        )
        .properties(height=400)
    )

    st.altair_chart(heatmap, use_container_width=True)

    st.caption(
        "La diagonal son aciertos; los valores fuera de la diagonal son errores de clasificación."
    )


# ------------------ PREDICCIONES ------------------
elif pagina == "Predicciones":
    st.markdown(
        '<div class="main-title">Predicciones en vivo 🔮</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Ajusta las características y mira qué predice cada modelo</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Ingresa las características de la flor</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        sepal_length = st.number_input(
            "sepal length (cm)", 4.0, 8.0, 5.9, step=0.1
        )
        sepal_width = st.number_input(
            "sepal width (cm)", 2.0, 4.5, 3.0, step=0.1
        )
    with col2:
        petal_length = st.number_input(
            "petal length (cm)", 1.0, 7.0, 5.0, step=0.1
        )
        petal_width = st.number_input(
            "petal width (cm)", 0.1, 2.5, 1.8, step=0.1
        )

    X_nuevo = [[sepal_length, sepal_width, petal_length, petal_width]]

    modelo_nombre = st.selectbox(
        "Modelo principal para la explicación", list(MODELOS.keys())
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

            proba_df = pd.DataFrame(
                {"Clase": class_names, "Probabilidad": proba}
            ).set_index("Clase")

            st.markdown(
                '<div class="section-title">Probabilidades por clase (modelo seleccionado)</div>',
                unsafe_allow_html=True,
            )
            st.bar_chart(proba_df["Probabilidad"])
        else:
            st.info(
                f"El modelo **{modelo_nombre}** no entrega probabilidades (`predict_proba`)."
            )

        st.markdown("---")

        # Comparación de modelos
        st.markdown(
            '<div class="section-title">Comparación de los 3 modelos</div>',
            unsafe_allow_html=True,
        )

        filas = []
        for nombre, m in MODELOS.items():
            pred_m = m.predict(X_nuevo)[0]
            especie_m = iris.target_names[pred_m]

            if hasattr(m, "predict_proba"):
                proba_m = m.predict_proba(X_nuevo)[0]
                proba_clase = float(max(proba_m))
                proba_str = f"{proba_clase:.3f}"
            else:
                proba_str = "N/A"

            filas.append(
                {
                    "Modelo": nombre,
                    "Especie predicha": especie_m,
                    "Probabilidad máx.": proba_str,
                }
            )

        resultados_df = pd.DataFrame(filas)
        st.dataframe(resultados_df, hide_index=True)

        st.caption(
            "Aquí ves si los modelos coinciden o discrepan para la misma flor y cuán seguros están."
        )
