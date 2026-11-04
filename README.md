# ☕ Clasificación del tueste de granos de café con CNN

Repositorio que reúne los dos proyectos desarrollados para la asignatura **Machine Learning** (INACAP): la clasificación automática del grado de tueste de granos de café a partir de imágenes, usando **Redes Neuronales Convolucionales (CNN)** implementadas con TensorFlow/Keras.

Cada imagen se clasifica en una de **4 categorías**:

| Clase | Descripción |
|-------|-------------|
| `Green` | Grano sin tostar (verde) |
| `Light` | Tueste claro |
| `Medium` | Tueste medio |
| `Dark` | Tueste oscuro |

## 📁 Estructura del repositorio

```
coffee-roast-classification/
├── README.md
├── evaluations/
│   ├── evaluation-3/              # Proyecto base
│   │   └── visual_recognition/
│   │       ├── src/               # Código fuente (6 módulos)
│   │       ├── models/            # Modelo entrenado (Keras)
│   │       ├── README.md          # Detalles y uso del proyecto
│   │       └── requirements.txt
│   └── evaluation-4/              # Proyecto mejorado
│       └── visual_recognition_enhanced/
│           ├── src/               # Código fuente (7 módulos, incl. augmentation y preprocessing)
│           ├── models/            # coffee_model.keras + best_model.keras
│           ├── results/           # Matriz de confusión, gráficos de entrenamiento y métricas
│           ├── README.md
│           └── requirements.txt
└── reports/
    ├── informe-evaluacion-3.pdf   # Informe completo de la Evaluación 3
    └── informe-evaluacion-4.pdf   # Informe completo de la Evaluación 4
```

## 🗂️ Dataset

El dataset proviene de Kaggle: [**Coffee Bean Dataset Resized (224 X 224)**](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224).

- **1600 imágenes** en total: **400 por cada categoría** (300 de entrenamiento + 100 de prueba por clase)
- Resolución: **224×224 px**, RGB
- Distribución esperada al colocar el dataset en cada proyecto:

```
dataset/
├── train/
│   ├── Dark/      # 300 imágenes
│   ├── Green/     # 300 imágenes
│   ├── Light/     # 300 imágenes
│   └── Medium/    # 300 imágenes
└── test/
    ├── Dark/      # 100 imágenes
    ├── Green/     # 100 imágenes
    ├── Light/     # 100 imágenes
    └── Medium/    # 100 imágenes
```

> ⚠️ **Nota:** el dataset no está incluido en el repositorio por su tamaño. Descárgalo desde Kaggle y descomprímelo en la raíz del proyecto que quieras ejecutar, respetando la estructura anterior (`dataset/train/<CLASE>/` y `dataset/test/<CLASE>/`).

## 🧩 Los dos proyectos

### Evaluación 3 — `visual_recognition` (proyecto base)
Implementación de una CNN desde cero para clasificar el tueste del grano de café. Incluye el pipeline completo: carga de datos, construcción del modelo, entrenamiento, evaluación y predicción sobre imágenes nuevas.

### Evaluación 4 — `visual_recognition_enhanced` (proyecto mejorado)
Versión optimizada del clasificador. Incorpora módulos adicionales de **aumento de datos** (`augmentation.py`) y **preprocesamiento** (`preprocessing.py`), además de la conservación del mejor modelo durante el entrenamiento (`best_model.keras`), lo que mejora la generalización y el rendimiento del clasificador.

## 📊 Resultados (Evaluación 4)

| Clase | Precisión | Recall | F1-Score |
|-------|-----------|--------|----------|
| Dark | 0.9510 | 0.9700 | 0.9604 |
| Green | 1.0000 | 1.0000 | 1.0000 |
| Light | 0.9900 | 0.9900 | 0.9900 |
| Medium | 0.9592 | 0.9400 | 0.9495 |

- **Accuracy: 97.5%**
- Average Precision / Recall / F1-Score: **0.9750**
- Loss: 0.0971

Evidencia visual disponible en `evaluations/evaluation-4/visual_recognition_enhanced/results/` (matriz de confusión y curvas de entrenamiento).

## 🚀 Instalación y uso

Los pasos son idénticos para ambos proyectos; ejecútalos desde la raíz del proyecto correspondiente (ej. `evaluations/evaluation-3/visual_recognition/`).

### 1. Requisitos previos
- Python 3.10+
- Crear y activar un entorno virtual:

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

### 2. Descargar el dataset
Descarga el dataset desde [Kaggle](https://www.kaggle.com/datasets/gpiosenka/coffee-bean-dataset-resized-224-x-224) y colócalo en `dataset/` dentro de la raíz del proyecto (ver estructura en la sección anterior).

### 3. Entrenar el modelo

```bash
python -m src.train
```

El entrenamiento puede tardar unos minutos; los gráficos se guardan en `results/training_plots.png`.

### 4. Evaluar el modelo

```bash
python -m src.evaluate
```

Genera la matriz de confusión y un reporte con accuracy, precision, recall, F1-score y el classification report completo (`results/evaluation_results.txt`).

### 5. Predecir sobre una imagen nueva

```bash
python -m src.predict "ruta/de/imagen.jpg"
```

Muestra la clase predicha, la confianza y el detalle de las 4 predicciones:

```
=== Prediction Results ===
Image: imagen.jpg
Predicted Class: Medium
Confidence: 0.9876

Top 4 Predictions:
  1. Medium: 0.9876
  2. Light: 0.0098
  3. Dark: 0.0015
  4. Green: 0.0011
```

## 🛠️ Tecnologías

- **Python 3**
- **TensorFlow / Keras** — construcción y entrenamiento de la CNN
- **NumPy / pandas** — manejo de datos
- **scikit-learn** — métricas de evaluación
- **OpenCV / Pillow** — carga y preprocesamiento de imágenes
- **matplotlib / seaborn** — visualización de resultados

## 📄 Informes

Los informes completos de ambas evaluaciones (contexto, arquitectura del modelo, metodología, resultados y conclusiones) se encuentran en la carpeta [`reports/`](reports/).

---

*Desarrollado por Evan Reyes — Machine Learning (INACAP)*
