# Spotify Tracks — Music Popularity Analysis

An end-to-end data analysis of the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
(~90k tracks, 113 genres). It studies how far a track's **audio features** explain and
predict its **popularity**, and whether the catalog admits a low-dimensional
representation or an interpretable segmentation.

The work spans four techniques over a single, reproducible pipeline:

- **Regression** — linear, polynomial (Ridge) and gradient boosting; compared by MSE.
- **PCA** — dimensionality reduction and loading interpretation.
- **Clustering** — K-Means segmentation of the catalog.
- **Neural network** — feed-forward classifier for popularity classes.

> **Course:** Desarrollo de Aplicaciones Avanzadas de Ciencias Computacionales (Gpo 507)
> — Tecnológico de Monterrey, Campus Monterrey.

## Layout

```
data/         Datasets (see data/README.md)
notebooks/    Step-by-step analysis, one notebook per stage (see notebooks/README.md)
src/          Supporting scripts (see src/README.md)
docs/         Dataset and business context
user_compatibility/   Optional extension: taste compatibility between two users
```

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Open the notebooks in order, or run them all headless:

```bash
python src/execute_notebooks.py
```

Notebook 02 produces the cleaned dataset the other notebooks consume. Every step that
uses randomness is seeded (`RANDOM_STATE = 42`) so results are reproducible.

## Authors

| Name | ID |
|---|---|
| Fidel Morales Briones | A01198630 |
| Kaled Noé Enríquez Trejo | A01198666 |
| Luis Alberto Rodríguez Solís | A01612435 |
| Valentino Villegas Martínez | A01772130 |

## Note

Popularity is driven by factors outside the audio signal (artist reputation, editorial
playlists, release timing, marketing) that are not present in this dataset. The models
are decision-support signals, not forecasts of commercial success.
