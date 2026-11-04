# Clasificador de nivel de tueste de granos de café

## Resumen

El código implementa un sistema de clasificación de tueste de café usando una red neuronal convolucional, para clasificar una imagen de un grano en una de las 4 categorías disponibles: Green, Light, Medium o Dark.

## Conjunto de datos

Se tienen imágenes de granos de café con las siguientes características

- **Entrenamiento**: 1200 imagenes (300 por clase)
- **Testing**: 400 imagenes (100 por clase)
- **Tamaño**: 224x224 pixels RGB
- **Clasificaciones**: Dark, Green, Light, Medium

## Instalación

### Antes de

Crear y activar un entorno Venv de Python, luego instalar las dependencias.

```bash
pip install -r requirements.txt
```

## Uso

Los comandos deben ejecutarse en la raíz ( / ) del proyecto

### 1. entrenar modelo

```bash
python -m src.train
```

Parecerá que se queda pegado, pero está entrenando, luego los resultados se guardan en `results/training_plots.png`

### 2. evaluar modelo

```bash
python -m src.evaluate
```

Genera una matríz de confusión y un archivo de texto que muestra

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report

### 3. usar y predecir

```bash
python -m src.predict "ruta_imagen.jpg"
```

Va a usar el algoritmo entrenado y mostrará 

- Predicción de clasificación de tueste
- Confianza
- Detalle de clasificación

### Ejemplo de predicción

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

