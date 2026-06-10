# Spotify Tracks AI Project: Business Context

## Dataset Context

This project uses the Kaggle Spotify Tracks dataset by Maharshi Pandya. Each record represents a song and includes Spotify-style metadata, genre labels, a popularity score from 0 to 100, and audio characteristics such as danceability, energy, acousticness, instrumentalness, tempo, valence, liveness, loudness, and speechiness.

The dataset is useful for an AI project because it combines business-facing outcomes (`popularity`) with interpretable audio features. That makes it possible to study both prediction tasks and unsupervised structure in the music catalog.

## Project Goal

The goal is to analyze whether song-level audio characteristics can explain or predict Spotify popularity and whether the catalog can be summarized into simpler groups or components. The project covers exploratory analysis, data cleaning, regression, PCA, neural network classification, and clustering.

## Business Value for Spotify

For Spotify, this type of analysis can support recommendation, playlist strategy, catalog discovery, and artist-facing insights. Regression models estimate expected popularity and identify which audio features are most associated with popularity. Classification models separate songs into Low, Medium, and High popularity categories, which can help prioritize tracks for promotion or recommendation tests.

PCA helps reduce many audio characteristics into a smaller set of components. This can make dashboards, recommendation experiments, and catalog analysis easier to interpret. Clustering discovers natural groups of songs, such as energetic, acoustic, instrumental, or danceable segments, which can support playlist design and user taste profiling.

## Important Limitation

Popularity is influenced by more than audio. Artist reputation, playlist placement, release timing, marketing, geography, social trends, and platform exposure are not fully represented in this dataset. Therefore, model predictions should be treated as decision-support signals rather than exact forecasts of commercial success.
