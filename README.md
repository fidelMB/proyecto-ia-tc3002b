# Análisis de popularidad de canciones de Spotify

Proyecto del curso Desarrollo de Aplicaciones Avanzadas de Ciencias Computacionales
(Gpo 507), Tecnológico de Monterrey.

Se usa el [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
de Kaggle (~90k canciones) para analizar qué tanto las características de audio de una
canción explican su popularidad.

## Notebooks

Correr en orden; el 02 genera el dataset limpio que usan los demás.

| Notebook | Contenido |
|---|---|
| `01_eda.ipynb` | Análisis exploratorio |
| `02_cleaning.ipynb` | Limpieza y nuevas columnas |
| `03_regression.ipynb` | Regresión lineal, polinomial y boosting |
| `04_pca.ipynb` | Reducción de dimensionalidad |
| `05_neural_network_classification.ipynb` | Clasificación con red neuronal (PyTorch) |
| `06_clustering.ipynb` | Clustering con K-Means |

`user_compatibility/` es una extensión: compatibilidad musical entre dos usuarios
a partir de sus perfiles de audio.

## Instalación

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Autores

| Nombre | Matrícula |
|---|---|
| Fidel Morales Briones | A01198630 |
| Kaled Noé Enríquez Trejo | A01198666 |
| Luis Alberto Rodríguez Solís | A01612435 |
| Valentino Villegas Martínez | A01772130 |
