# Notebooks

The analysis pipeline, one notebook per stage. Run them in order — notebook 02 builds the
cleaned dataset the rest depend on. To run them all headless: `python src/execute_notebooks.py`.

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_eda.ipynb` | Exploratory analysis: types, missing values, popularity distribution, correlations. |
| 02 | `02_cleaning_feature_engineering.ipynb` | Deduplication, imputation, derived features, popularity classes. |
| 03 | `03_regression.ipynb` | Linear, polynomial (Ridge) and gradient-boosting regression; MSE comparison, overfitting and residual analysis. |
| 04 | `04_pca.ipynb` | Standardization, explained variance, component selection, loadings. |
| 05 | `05_neural_network_classification.ipynb` | PyTorch MLP: train/val/test split, training metrics, confusion matrix, regularization. |
| 06 | `06_clustering.ipynb` | K-Means: choosing *k*, cluster profiles and interpretation. |

The notebooks are generated from `src/generate_project_artifacts.py`, which is their
source of truth — hand edits to a `.ipynb` are overwritten when it is regenerated.
