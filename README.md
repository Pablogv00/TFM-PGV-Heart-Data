# TFM-PGV-Heart-Data

# Predicción de Mortalidad por Insuficiencia Cardíaca con Machine Learning

Este proyecto implementa un modelo de Machine Learning para predecir la mortalidad por insuficiencia cardíaca a partir de datos clínicos.  
Incluye una aplicación en Streamlit donde los usuarios pueden ingresar datos y obtener predicciones de los modelos previamente entrenados.

## Dataset Original: [Heart Failure Clinical Records](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)

---

## Tecnologías Utilizadas
**Lenguaje**: Python 3.11  
**Librerías**: requirements.txt  
**Modelos de Machine Learning**: Decision Tree, Bagging, Random Forest, Gradient Boosting, XGBoost  

---

## Estructura del proyecto
```markdown
 TFM-PGV-Heart-Data/
 ├── Metricas/                                      # Carpeta con métricas de rendimiento de modelos en .pkl
 ├── Modelos/                                       # Carpeta con modelos entrenados en .pkl
 ├── app.py                                         # Código de la aplicación Streamlit
 ├── favicon.ico                                    # Logo de la aplicación Streamlit
 ├── heart_failure_clinical_records_dataset.csv     # Dataset en csv
 ├── README.md                                      # Este archivo con instrucciones
 ├── requirements.txt                               # Dependencias necesarias
 ├── tfm.ipynb                                      # Código de entrenamiento de modelos
```
---

##  Instalación y Configuración
Para ejecutar este proyecto en tu máquina local, sigue estos pasos:

### 1. Clonar el Repositorio
```bash
git clone https://github.com/Pablogv00/TFM-PGV-Heart-Data.git
cd TFM-PGV-Heart-Data

```

### 2. Crear un Entorno Virtual (Opcional)
```bash
conda create --name tfm_env python=3.11
conda activate tfm_env
```

### 3. Instalar Dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar el notebook *tfm.ipynb* para ver el código completo de generación y entrenamiento de los modelos.

### 5. Ejecutar la aplicación para predecir la mortalidad con datos input.
```bash
streamlit run app.py
```

---

## Contacto

**Autor**:  Pablo Guzmán Valle  
**Email**:  pabguzval@gmail.com  
**GitHub**: Pablogv00  
