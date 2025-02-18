import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Cargar dataset
dataset_path = "heart_failure_clinical_records_dataset.csv"
df = pd.read_csv(dataset_path)

# Cargar modelos guardados
modelos = {
    "🌳 Árbol de Decisión": "Modelos/arbol_base.pkl",
    "🌐 Bagging": "Modelos/bagging.pkl",
    "🌲 Random Forest": "Modelos/random_forest.pkl",
    "📈 Gradient Boosting": "Modelos/gradient_boosting.pkl",
    "🚀 XGBoost": "Modelos/xg_boost.pkl"    
}

# Función para cargar cada modelo
def cargar_modelo(nombre_modelo):
    with open(modelos[nombre_modelo], "rb") as file:
        return pickle.load(file)

# Función para representar las métricas de cada modelo
def mostrar_matriz_y_reporte(metrica, clase):
    """
    Función para mostrar la matriz de confusión y el reporte de clasificación.
    
    Parámetros:
    - metrica: Diccionario con las métricas de Train/Test cargadas desde el archivo.
    - clase: "Train" o "Test".
    """
    conf_matrix = metrica[clase]["confusion_matrix"]
    df_report = pd.DataFrame(metrica[clase]["classification_report"]).transpose().drop(["accuracy"], errors="ignore")

    # Crear la figura y los subplots más compactos
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))  

    # Matriz de Confusión
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=axes[0])
    axes[0].set_title(f"Matriz de Confusión - {clase}", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Predicción", fontsize=10)
    axes[0].set_ylabel("Valor Real", fontsize=10)

    # Reporte de Clasificación
    df_report[['precision', 'recall', 'f1-score']].plot(kind='bar', colormap="coolwarm", ax=axes[1])
    axes[1].set_title(f"Reporte de Clasificación - {clase}", fontsize=12, fontweight='bold')
    axes[1].set_xticklabels(df_report.index, rotation=45, ha='right', fontsize=9)
    axes[1].set_ylabel("Puntuación", fontsize=10)
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="lower right", fontsize=8)

    plt.subplots_adjust(wspace=0.3)  

    # Mostrar en Streamlit
    st.pyplot(fig)

    
# Configurar la página con título y favicon personalizado
st.set_page_config(
    page_title="Predicción de Mortalidad",
    page_icon="favicon.ico", 
    layout="wide"
)

# Título de la app    
st.title("🩺 Mortalidad por fallo cardíaco 💓")

# Creación de pestañas 
pestañas = ["🔍 Predicción de Mortalidad", "📊 Información del Dataset", "⚙️ Modelos de Machine Learning"]
tab1, tab2, tab3 = st.tabs(pestañas)

# PESTAÑA 1: PREDICCIÓN DE MORTALIDAD
with tab1:
    st.markdown("<h1 style='text-align: center;'>🔍 Predicción de Mortalidad</h1>", unsafe_allow_html=True)
    
    # SECCIÓN 1: SELECCIÓN DE MODELOS
    with st.expander("🛠️ **Seleccionar Modelos de ML**", expanded=True):
        st.write("Selecciona los modelos con los que deseas hacer la predicción:")
        modelos_seleccionados = []
        for modelo in modelos.keys():
            if st.checkbox(modelo, value=True):
                modelos_seleccionados.append(modelo)

    # SECCIÓN 2: INGRESO DE DATOS
    with st.expander("📌 **Ingresar Datos del Paciente**", expanded=True):
        st.write("Introduce los datos clínicos del paciente:")

        # Diseño en columnas para organizar inputs
        col1, col_espacio, col2 = st.columns([1, 0.2, 1])
        with col1:
            age = st.number_input("Edad", min_value=0, max_value=150, value=50)
            ejection_fraction = st.number_input("Fracción de Eyección (%)", min_value=1, max_value=100, value=35)
            serum_creatinine = st.number_input("Creatinina en Suero (mg/dL)", min_value=0.0, max_value=15.0, value=1.1)
            serum_sodium = st.number_input("Sodio en Suero (mEq/L)", min_value=100.0, max_value=200.0, value=137.1)
            platelets = st.number_input("Plaquetas (x10^3/μL)", min_value=10000, max_value=1000000, value=250000)
            creatinine_phosphokinase = st.number_input("Creatinina Fosfocinasa (CPK)", min_value=0, max_value=10000, value=200)

        with col2:
            high_blood_pressure = st.radio("Hipertensión", ["No", "Sí"])
            smoking = st.radio("Fuma", ["No", "Sí"])
            diabetes = st.radio("Diabetes", ["No", "Sí"])
            anaemia = st.radio("Anemia", ["No", "Sí"])
            sex = st.radio("Sexo", ["Hombre", "Mujer"])
        

        # Convertir opciones a formato numérico
        high_blood_pressure = 1 if high_blood_pressure == "Sí" else 0
        smoking = 1 if smoking == "Sí" else 0
        diabetes = 1 if diabetes == "Sí" else 0
        anaemia = 1 if anaemia == "Sí" else 0
        sex = 1 if sex == "Hombre" else 0

    # SECCIÓN 3: BOTÓN PARA EJECUTAR PREDICCIÓN
    st.markdown("""
        <style>
            .stButton>button {
                width: 100%;
                border-radius: 8px;
                font-size: 18px;
                font-weight: bold;
                background-color: #007BFF;
                color: white;
                padding: 10px;
                transition: 0.3s;
                border: none;
            }
            .stButton>button:hover {
                background-color: #0056b3;
            }
        </style>
    """, unsafe_allow_html=True)

    # Botón de predicción
    if st.button("🔍 Predecir Mortalidad"):
        input_data = np.array([age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure,
                            platelets, serum_creatinine, serum_sodium, sex, smoking]).reshape(1, -1)

        # Predecir con los modelos seleccionados y mostrar resultados
        st.subheader("📊 **Resultados de Predicción**")
        
        for modelo_nombre in modelos_seleccionados:
            modelo = cargar_modelo(modelo_nombre)
            prediccion = modelo.predict(input_data)

            # Crear contenedor con el resultado del modelo
            with st.container():
                if prediccion[0] in ["yes", 1]:
                    st.error(f"❌ **{modelo_nombre}**: Predice Mortalidad")
                else:
                    st.success(f"✅ **{modelo_nombre}**: Predice Supervivencia")


# PESTAÑA 2: INFORMACIÓN SOBRE EL DATASET
with tab2:
    st.markdown("<h1 style='text-align: center;'>📊 Información del Dataset</h1>", unsafe_allow_html=True)

    with st.expander("🔗 Enlace al Dataset en Kaggle"):
        st.markdown("[📂 Dataset en Kaggle](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)", unsafe_allow_html=True)

    with st.expander("📜 Descripción del Dataset"):
        st.write("""
                Las enfermedades cardiovasculares (CVDs) son la principal causa de muerte a nivel mundial, cobrando aproximadamente 17.9 millones de vidas cada año, 
                lo que representa el 31% de todas las muertes en el mundo.

                La insuficiencia cardíaca es un evento común causado por las CVDs, y este conjunto de datos contiene 12 variables que pueden utilizarse para predecir 
                la mortalidad por insuficiencia cardíaca.

                La mayoría de las enfermedades cardiovasculares pueden prevenirse abordando los factores de riesgo conductuales, como el consumo de tabaco, una dieta 
                poco saludable y la obesidad, la inactividad física y el consumo perjudicial de alcohol, mediante estrategias de salud pública.

                Las personas con enfermedades cardiovasculares o que tienen un alto riesgo cardiovascular (debido a la presencia de uno o más factores de riesgo como 
                hipertensión, diabetes, hiperlipidemia o una enfermedad ya establecida) necesitan detección temprana y manejo adecuado, donde un modelo de aprendizaje 
                automático (Machine Learning) puede ser de gran ayuda.
                 
                A continuación se listan las variables que intervienen en el dataset, de las cuales, solo se excluye del estudio la variable 'time', que representa el 
                tiempo de seguimiento del paciente (en días), es decir, cuánto tiempo se monitoreó antes de registrar si falleció o no. En un caso de uso real, cuando 
                un nuevo paciente llegue a la consulta, no conoceremos su tiempo de seguimiento futuro, por lo que esta variable no estará disponible para hacer predicciones. 
                """)
        
        df_variables = pd.DataFrame({
        "Variable": ["age", "anaemia", "creatinine_phosphokinase", "diabetes", "ejection_fraction",
                     "high_blood_pressure", "platelets", "serum_creatinine", "serum_sodium",
                     "sex", "smoking", "time (no se usa)", "DEATH_EVENT"],
        "Descripción": [
            "Edad del paciente (en años).",
            "Indica si el paciente tiene anemia (0 = No, 1 = Sí).",
            "Nivel de creatina fosfocinasa (CPK) en la sangre (mcg/L).",
            "Indica si el paciente tiene diabetes (0 = No, 1 = Sí).",
            "Porcentaje de sangre expulsada por el ventrículo izquierdo en cada latido (%).",
            "Indica si el paciente tiene presión arterial alta (0 = No, 1 = Sí).",
            "Cantidad de plaquetas en la sangre (x10^3/μL).",
            "Nivel de creatinina en suero (mg/dL), usado para evaluar la función renal.",
            "Nivel de sodio en suero (mEq/L), importante para la regulación de líquidos en el cuerpo.",
            "Sexo del paciente (0 = Mujer, 1 = Hombre).",
            "Indica si el paciente es fumador (0 = No, 1 = Sí).",
            "Número de días de seguimiento del paciente en el estudio.",
            "Variable objetivo: indica si el paciente falleció durante el período de seguimiento (0 = No, 1 = Sí)."
            ]
        })

        st.table(df_variables)

    with st.expander("🗂️ Vista Previa del Dataset"):
        st.write(df.head())

    with st.expander("📊 Histogramas de las Variables Numéricas"):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Distribución de las Variables Numéricas", fontsize=18, fontweight='bold')

        columnas_numericas = ["age", "creatinine_phosphokinase", "ejection_fraction",
                            "platelets", "serum_creatinine", "serum_sodium"]

        for i, col in enumerate(columnas_numericas):
            row, col_index = divmod(i, 3)
            sns.histplot(df[col], bins=20, kde=True, ax=axes[row, col_index])
            axes[row, col_index].set_title(col, fontweight='bold', fontsize=14)
            axes[row, col_index].set_xlabel("", fontsize=12)
            axes[row, col_index].set_ylabel("Frecuencia", fontsize=12)
            axes[row, col_index].grid(True, linestyle="--", alpha=0.6)  

        # Ajustar márgenes para evitar superposiciones
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        st.pyplot(fig)


# PESTAÑA 3: INFORMACIÓN SOBRE LOS MODELOS
with tab3:
    st.markdown("<h1 style='text-align: center;'>⚙️ Modelos de Machine Learning</h1>", unsafe_allow_html=True)

    st.write("""A continuación se presentan los modelos empleados, dando una breve descripción de su funcionamiento y representando 
                las métricas de precisión alcanzadas por el modelo en los datos de entrenamiento y test.
             """)

    with st.expander("🌳 Árbol de Decisión"):
        st.write("""Modelo que divide los datos en subconjuntos en función de condiciones sobre las características. 
                 Cada nodo del árbol representa una condición sobre una variable, y cada rama es el resultado de esa condición.""")

        # Cargar las métricas guardadas
        with open("Metricas/arbol_base.pkl", "rb") as file:
            metricas = pickle.load(file)

        # Mostrar las métricas
        st.subheader("📊 Evaluación del Modelo en Train")
        mostrar_matriz_y_reporte(metricas, "Train")

        st.subheader("📊 Evaluación del Modelo en Test")
        mostrar_matriz_y_reporte(metricas, "Test")

    with st.expander("🌐 Bagging Classifier"):
        st.write("""técnica de ensamblado que mejora la estabilidad y precisión de modelos inestables, como los árboles de decisión. 
                    Su estrategia consiste en entrenar múltiples modelos en subconjuntos de datos obtenidos mediante bootstrap 
                    (muestreo con reemplazo) y luego combinar sus predicciones.""")

        # Cargar las métricas guardadas
        with open("Metricas/bagging.pkl", "rb") as file:
            metricas = pickle.load(file)

        # Mostrar las métricas
        st.subheader("📊 Evaluación del Modelo en Train")
        mostrar_matriz_y_reporte(metricas, "Train")

        st.subheader("📊 Evaluación del Modelo en Test")
        mostrar_matriz_y_reporte(metricas, "Test")

    with st.expander("🌲 Random Forest"):
        st.write("""Ensamblado de múltiples árboles de decisión para mayor estabilidad. Introduce aleatorización en la selección de 
                    características, creando múltiples árboles de decisión mas diversos y reduciendo la varianza.""")

        # Cargar las métricas guardadas
        with open("Metricas/random_forest.pkl", "rb") as file:
            metricas = pickle.load(file)

        # Mostrar las métricas
        st.subheader("📊 Evaluación del Modelo en Train")
        mostrar_matriz_y_reporte(metricas, "Train")

        st.subheader("📊 Evaluación del Modelo en Test")
        mostrar_matriz_y_reporte(metricas, "Test")
    
    with st.expander("📈 Gradient Boosting"):
        st.write("""Método de ensamblado basado en boosting, donde cada nuevo árbol trata de corregir los errores de los árboles anteriores. 
                    Se basa en minimizar el error de predicción en cada iteración.""")

        # Cargar las métricas guardadas
        with open("Metricas/gradient_boosting.pkl", "rb") as file:
            metricas = pickle.load(file)

        # Mostrar las métricas
        st.subheader("📊 Evaluación del Modelo en Train")
        mostrar_matriz_y_reporte(metricas, "Train")

        st.subheader("📊 Evaluación del Modelo en Test")
        mostrar_matriz_y_reporte(metricas, "Test")

    with st.expander("🚀 XGBoost"):
        st.write("Modelo avanzado de boosting optimizado para datos estructurados.")

        # Cargar las métricas guardadas
        with open("Metricas/xg_boost.pkl", "rb") as file:
            metricas = pickle.load(file)

        # Mostrar las métricas
        st.subheader("📊 Evaluación del Modelo en Train")
        mostrar_matriz_y_reporte(metricas, "Train")

        st.subheader("📊 Evaluación del Modelo en Test")
        mostrar_matriz_y_reporte(metricas, "Test")
