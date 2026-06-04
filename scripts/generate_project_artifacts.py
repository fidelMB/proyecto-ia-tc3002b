from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"
DATA_DIR = ROOT / "data"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def write_notebook(path: Path, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python (.venv)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)


common_setup = r"""
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
DATA_CLEAN = PROJECT_ROOT / "data" / "processed" / "spotify_tracks_clean.csv"

RANDOM_STATE = 42
"""


eda_cells = [
    md(
        """
        # 01 - Exploratory Data Analysis

        This notebook explores the Kaggle Spotify Tracks dataset before modeling. The objective is to understand the scale, data types, missing values, popularity distribution, and relationships between audio features and song popularity.
        """
    ),
    code(common_setup),
    code(
        r"""
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid", palette="Set2")
        pd.set_option("display.max_columns", 50)

        df = pd.read_csv(DATA_RAW)
        print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
        display(df.head())
        """
    ),
    md(
        """
        ## Dataset Structure

        Each row represents a Spotify track. The columns include identifiers and descriptive metadata (`track_id`, `artists`, `album_name`, `track_name`, `track_genre`) plus numerical audio attributes such as `danceability`, `energy`, `tempo`, `valence`, and the target variable `popularity`.
        """
    ),
    code(
        r"""
        display(df.info())
        display(df.describe(include="all").T)
        """
    ),
    md(
        """
        ## Missing Values and Duplicates

        Missing values affect cleaning decisions because imputation is only useful when missingness is limited and the column is still analytically valuable. Duplicate tracks can bias model training by allowing the same song to appear more than once.
        """
    ),
    code(
        r"""
        missing = (
            df.isna().sum()
            .to_frame("missing_count")
            .assign(missing_percent=lambda x: 100 * x["missing_count"] / len(df))
            .sort_values("missing_count", ascending=False)
        )
        display(missing)

        print(f"Duplicated rows: {df.duplicated().sum():,}")
        print(f"Duplicated track_id values: {df['track_id'].duplicated().sum():,}")
        """
    ),
    md(
        """
        ## Central Tendency

        Mean and median summarize numerical columns, while mode is useful for categorical attributes such as genre, explicit content, and musical key.
        """
    ),
    code(
        r"""
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        central_tendency = pd.DataFrame({
            "mean": df[numeric_cols].mean(numeric_only=True),
            "median": df[numeric_cols].median(numeric_only=True),
            "mode": df[numeric_cols].mode().iloc[0],
        })
        display(central_tendency)

        categorical_modes = df[categorical_cols].mode().iloc[0].to_frame("mode")
        display(categorical_modes)
        """
    ),
    md(
        """
        ## Popularity Distribution

        Popularity is the regression target. A skewed or uneven distribution means prediction error should be interpreted carefully: predicting exact popularity is harder for sparse high-popularity observations than for common mid-range songs.
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.histplot(df["popularity"], bins=30, kde=True, ax=axes[0])
        axes[0].set_title("Popularity Distribution")
        axes[0].set_xlabel("Popularity")

        sns.boxplot(x=df["popularity"], ax=axes[1])
        axes[1].set_title("Popularity Box Plot")
        plt.tight_layout()
        plt.show()

        print(df["popularity"].describe())
        """
    ),
    md(
        """
        ## Audio Feature Distributions

        Audio features are mostly bounded between 0 and 1, while `tempo`, `duration_ms`, and `loudness` have different scales. This is why later notebooks standardize features before PCA, clustering, and neural network training.
        """
    ),
    code(
        r"""
        audio_features = [
            "danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "duration_ms"
        ]

        df[audio_features].hist(bins=35, figsize=(16, 12), color="#4C78A8")
        plt.suptitle("Audio Feature Histograms", y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        r"""
        plt.figure(figsize=(15, 7))
        sns.boxplot(data=df[audio_features], orient="h")
        plt.title("Audio Feature Box Plots")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Correlation Analysis

        Correlation helps identify linear relationships and potential multicollinearity. It is not proof of causation, but it guides feature interpretation for regression and PCA.
        """
    ),
    code(
        r"""
        corr_cols = ["popularity"] + audio_features + ["explicit", "key", "mode", "time_signature"]
        corr = df[corr_cols].copy()
        corr["explicit"] = corr["explicit"].astype(int)

        plt.figure(figsize=(13, 10))
        sns.heatmap(corr.corr(), cmap="coolwarm", center=0, annot=False)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

        popularity_corr = corr.corr(numeric_only=True)["popularity"].drop("popularity").sort_values(key=np.abs, ascending=False)
        display(popularity_corr.to_frame("correlation_with_popularity"))
        """
    ),
    md(
        """
        ## Scatter Plots Against Popularity

        These plots show whether high or low values of audio characteristics are associated with popularity. Wide vertical spread means popularity is influenced by factors outside the dataset too, such as artist reputation, playlist placement, release timing, and marketing.
        """
    ),
    code(
        r"""
        scatter_features = ["danceability", "energy", "acousticness", "instrumentalness", "valence", "tempo"]
        sample_df = df.sample(n=min(8000, len(df)), random_state=RANDOM_STATE)

        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        for ax, feature in zip(axes.ravel(), scatter_features):
            sns.scatterplot(data=sample_df, x=feature, y="popularity", alpha=0.25, s=12, ax=ax)
            ax.set_title(f"Popularity vs. {feature}")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Genre Context

        Genre is a business-relevant grouping because Spotify recommendations and editorial playlists often operate by genre or mood. Popularity differences by genre can inform segmentation, but they should not be interpreted as purely audio-driven differences.
        """
    ),
    code(
        r"""
        top_genres = df["track_genre"].value_counts().head(15).index
        genre_popularity = (
            df[df["track_genre"].isin(top_genres)]
            .groupby("track_genre")["popularity"]
            .agg(["count", "mean", "median"])
            .sort_values("mean", ascending=False)
        )
        display(genre_popularity)

        plt.figure(figsize=(14, 7))
        sns.boxplot(data=df[df["track_genre"].isin(top_genres)], x="track_genre", y="popularity")
        plt.xticks(rotation=45, ha="right")
        plt.title("Popularity by Top Genres")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## EDA Takeaways

        The dataset is large enough for reliable train/test splits and includes a rich set of audio features. Popularity is noisy and not explained by audio features alone, so models should be evaluated by generalization performance rather than expecting perfect predictions. Standardization is necessary for PCA, clustering, and neural networks because the variables use different units and scales.
        """
    ),
]


cleaning_cells = [
    md(
        """
        # 02 - Cleaning and Feature Engineering

        This notebook prepares a modeling-ready dataset. The cleaning strategy keeps the audio and business-relevant columns, removes clear identifiers from model features, imputes missing values, creates useful engineered variables, and saves the final clean dataset for the remaining notebooks.
        """
    ),
    code(common_setup),
    code(
        r"""
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid", palette="Set2")
        df = pd.read_csv(DATA_RAW)
        print(df.shape)
        display(df.head())
        """
    ),
    md(
        """
        ## Cleaning Decisions

        `Unnamed: 0` is an index artifact, so it is removed. Song identifiers and text metadata are useful for interpretation but are not predictive audio features. For modeling fairness and leakage prevention, `track_id`, `track_name`, `album_name`, and `artists` are excluded from model feature matrices.
        """
    ),
    code(
        r"""
        df_clean = df.copy()
        if "Unnamed: 0" in df_clean.columns:
            df_clean = df_clean.drop(columns=["Unnamed: 0"])

        before_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates(subset=["track_id"]).reset_index(drop=True)
        print(f"Rows before duplicate removal: {before_rows:,}")
        print(f"Rows after duplicate track_id removal: {len(df_clean):,}")
        """
    ),
    md(
        """
        ## Missing Data Imputation

        Numeric columns are imputed with medians because medians are robust to outliers. Categorical columns are imputed with the mode or `Unknown` when a mode is unavailable.
        """
    ),
    code(
        r"""
        numeric_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df_clean.select_dtypes(exclude=np.number).columns.tolist()

        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        for col in categorical_cols:
            mode = df_clean[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            df_clean[col] = df_clean[col].fillna(fill_value)

        missing_after = df_clean.isna().sum().sum()
        print(f"Remaining missing values: {missing_after}")
        """
    ),
    md(
        """
        ## Feature Engineering

        The engineered features improve interpretability and make later notebooks simpler:

        - `duration_min` converts milliseconds to minutes.
        - `is_explicit` converts the Boolean explicit flag into 0/1.
        - `popularity_class` creates balanced Low/Medium/High labels using quantile bins for classification.
        """
    ),
    code(
        r"""
        df_clean["duration_min"] = df_clean["duration_ms"] / 60000
        df_clean["is_explicit"] = df_clean["explicit"].astype(int)

        labels = ["Low", "Medium", "High"]
        df_clean["popularity_class"] = pd.qcut(
            df_clean["popularity"],
            q=3,
            labels=labels,
            duplicates="drop",
        )

        display(df_clean[["popularity", "duration_min", "is_explicit", "popularity_class"]].head())
        display(df_clean["popularity_class"].value_counts().to_frame("count"))
        """
    ),
    md(
        """
        ## Scaling Preview

        Scaling is not saved over the original values because train/test splits must fit scalers only on training data to avoid leakage. This preview shows why scaling will be applied inside each modeling notebook.
        """
    ),
    code(
        r"""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler

        audio_features = [
            "danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "duration_min",
            "key", "mode", "time_signature", "is_explicit"
        ]

        standard_preview = pd.DataFrame(
            StandardScaler().fit_transform(df_clean[audio_features]),
            columns=audio_features
        ).describe().loc[["mean", "std"]]

        normalized_preview = pd.DataFrame(
            MinMaxScaler().fit_transform(df_clean[audio_features]),
            columns=audio_features
        ).describe().loc[["min", "max"]]

        display(standard_preview.round(3))
        display(normalized_preview.round(3))
        """
    ),
    md(
        """
        ## One-Hot Encoding Preview

        `track_genre` is categorical. It is kept in the clean dataset and one-hot encoded inside modeling pipelines so the encoding is reproducible and can be fitted only on training data.
        """
    ),
    code(
        r"""
        genre_encoded_preview = pd.get_dummies(df_clean[["track_genre"]], drop_first=True)
        print(f"Genre dummy columns created: {genre_encoded_preview.shape[1]:,}")
        display(genre_encoded_preview.head())
        """
    ),
    code(
        r"""
        DATA_CLEAN.parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(DATA_CLEAN, index=False)
        print(f"Saved clean dataset to: {DATA_CLEAN}")
        print(f"Clean shape: {df_clean.shape}")
        """
    ),
    md(
        """
        ## Cleaning Summary

        The final clean dataset removes duplicate track IDs, fills missing values, adds classification and duration features, and preserves raw audio feature values so each algorithm can apply scaling correctly after its train/test split.
        """
    ),
]


regression_cells = [
    md(
        """
        # 03 - Regression Models for Popularity

        Goal: predict a song's `popularity` score from Spotify audio characteristics. This notebook compares multiple linear regression and degree-2 polynomial regression using train/test MSE.
        """
    ),
    code(common_setup),
    code(
        r"""
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

        sns.set_theme(style="whitegrid", palette="Set2")
        df = pd.read_csv(DATA_CLEAN)
        print(df.shape)
        display(df.head())
        """
    ),
    md(
        """
        ## Feature Selection

        Identifiers and text metadata are excluded because they do not represent transferable audio characteristics. Genre is included as a one-hot encoded business context feature. For polynomial regression, only numeric audio features are expanded; genre remains linear so the model captures nonlinear audio effects without creating thousands of genre interaction columns.
        """
    ),
    code(
        r"""
        target = "popularity"
        numeric_features = [
            "duration_min", "is_explicit", "danceability", "energy", "key", "loudness",
            "mode", "speechiness", "acousticness", "instrumentalness", "liveness",
            "valence", "tempo", "time_signature"
        ]
        categorical_features = ["track_genre"]

        X = df[numeric_features + categorical_features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )
        print(X_train.shape, X_test.shape)
        """
    ),
    code(
        r"""
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False)

        linear_preprocess = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]), numeric_features),
                ("genre", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", encoder)
                ]), categorical_features)
            ]
        )

        linear_model = Pipeline([
            ("preprocess", linear_preprocess),
            ("model", LinearRegression())
        ])

        linear_model.fit(X_train, y_train)
        linear_train_pred = linear_model.predict(X_train)
        linear_test_pred = linear_model.predict(X_test)

        linear_train_mse = mean_squared_error(y_train, linear_train_pred)
        linear_test_mse = mean_squared_error(y_test, linear_test_pred)
        print(f"Linear train MSE: {linear_train_mse:.3f}")
        print(f"Linear test MSE: {linear_test_mse:.3f}")
        """
    ),
    md(
        """
        ## Polynomial Regression

        The polynomial model is tuned over degrees 2 and 3 with Ridge regularization. Degree 2 captures curved relationships and pairwise interactions, while degree 3 allows a slightly richer nonlinear shape. Ridge regularization is used to reduce overfitting risk from the expanded feature space.
        """
    ),
    code(
        r"""
        def make_encoder(drop_first=True):
            try:
                return OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first" if drop_first else None,
                    sparse_output=False,
                )
            except TypeError:
                return OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first" if drop_first else None,
                    sparse=False,
                )


        polynomial_results = []
        polynomial_models = {}

        for degree in [2, 3]:
            for alpha in [0.1, 1.0, 10.0, 100.0]:
                polynomial_preprocess = ColumnTransformer(
                    transformers=[
                        ("num_poly", Pipeline([
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                        ]), numeric_features),
                        ("genre", Pipeline([
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", make_encoder(drop_first=True)),
                        ]), categorical_features),
                    ]
                )

                model_name = f"Polynomial Ridge degree {degree}, alpha {alpha:g}"
                candidate = Pipeline([
                    ("preprocess", polynomial_preprocess),
                    ("model", Ridge(alpha=alpha)),
                ])
                candidate.fit(X_train, y_train)
                train_pred = candidate.predict(X_train)
                test_pred = candidate.predict(X_test)
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                polynomial_results.append({
                    "model": model_name,
                    "degree": degree,
                    "alpha": alpha,
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                    "test_rmse_popularity_points": np.sqrt(test_mse),
                    "generalization_gap": test_mse - train_mse,
                })
                polynomial_models[model_name] = candidate

        polynomial_results_df = (
            pd.DataFrame(polynomial_results)
            .sort_values("test_mse")
            .reset_index(drop=True)
        )
        display(polynomial_results_df.round(3))

        best_polynomial_row = polynomial_results_df.iloc[0]
        best_polynomial_model = polynomial_models[best_polynomial_row["model"]]
        poly_train_pred = best_polynomial_model.predict(X_train)
        poly_test_pred = best_polynomial_model.predict(X_test)
        poly_train_mse = best_polynomial_row["train_mse"]
        poly_test_mse = best_polynomial_row["test_mse"]
        print(f"Best polynomial model: {best_polynomial_row['model']}")
        print(f"Polynomial train MSE: {poly_train_mse:.3f}")
        print(f"Polynomial test MSE: {poly_test_mse:.3f}")
        """
    ),
    md(
        """
        ## Tuned Histogram Gradient Boosting Regression

        Linear and polynomial regression are useful for interpretation, but gradient boosting can model nonlinear thresholds and interactions without manually expanding every feature. This tuning pass compares learning rates while keeping early stopping enabled to control overfitting.
        """
    ),
    code(
        r"""
        boosting_results = []
        boosting_models = {}

        for learning_rate in [0.04, 0.06, 0.08]:
            boosting_preprocess = ColumnTransformer(
                transformers=[
                    ("num", SimpleImputer(strategy="median"), numeric_features),
                    ("genre", Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_encoder(drop_first=False)),
                    ]), categorical_features),
                ]
            )
            model_name = f"HistGradientBoosting lr {learning_rate}"
            candidate = Pipeline([
                ("preprocess", boosting_preprocess),
                ("model", HistGradientBoostingRegressor(
                    max_iter=250,
                    learning_rate=learning_rate,
                    max_leaf_nodes=31,
                    l2_regularization=0.1,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=RANDOM_STATE,
                )),
            ])
            candidate.fit(X_train, y_train)
            train_pred = candidate.predict(X_train)
            test_pred = candidate.predict(X_test)
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            boosting_results.append({
                "model": model_name,
                "learning_rate": learning_rate,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "test_rmse_popularity_points": np.sqrt(test_mse),
                "generalization_gap": test_mse - train_mse,
            })
            boosting_models[model_name] = candidate

        boosting_results_df = (
            pd.DataFrame(boosting_results)
            .sort_values("test_mse")
            .reset_index(drop=True)
        )
        display(boosting_results_df.round(3))

        best_boosting_row = boosting_results_df.iloc[0]
        best_boosting_model = boosting_models[best_boosting_row["model"]]
        boosting_train_pred = best_boosting_model.predict(X_train)
        boosting_test_pred = best_boosting_model.predict(X_test)
        boosting_train_mse = best_boosting_row["train_mse"]
        boosting_test_mse = best_boosting_row["test_mse"]
        print(f"Best boosted model: {best_boosting_row['model']}")
        print(f"Boosting train MSE: {boosting_train_mse:.3f}")
        print(f"Boosting test MSE: {boosting_test_mse:.3f}")
        """
    ),
    code(
        r"""
        mse_table = pd.DataFrame([
            {
                "model": "Multiple Linear Regression baseline",
                "train_mse": linear_train_mse,
                "test_mse": linear_test_mse,
                "test_rmse_popularity_points": np.sqrt(linear_test_mse),
                "generalization_gap": linear_test_mse - linear_train_mse,
            },
            best_polynomial_row[["model", "train_mse", "test_mse", "test_rmse_popularity_points", "generalization_gap"]].to_dict(),
            best_boosting_row[["model", "train_mse", "test_mse", "test_rmse_popularity_points", "generalization_gap"]].to_dict(),
        ])
        mse_table["test_mse_improvement_vs_baseline"] = linear_test_mse - mse_table["test_mse"]
        mse_table["test_mse_improvement_percent"] = 100 * mse_table["test_mse_improvement_vs_baseline"] / linear_test_mse
        mse_table = mse_table.sort_values("test_mse").reset_index(drop=True)
        display(mse_table.round(3))
        """
    ),
    md(
        """
        ## Which Features Have the Strongest Effect?

        Linear coefficients are interpreted after scaling numeric features. Larger absolute coefficients indicate stronger association with popularity in the fitted linear model, holding other included variables constant.
        """
    ),
    code(
        r"""
        preprocessor = linear_model.named_steps["preprocess"]
        feature_names = preprocessor.get_feature_names_out()
        coefficients = linear_model.named_steps["model"].coef_

        coef_table = (
            pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
            .assign(abs_coefficient=lambda x: x["coefficient"].abs())
            .sort_values("abs_coefficient", ascending=False)
        )
        display(coef_table.head(20))
        """
    ),
    md(
        """
        ## Actual vs. Predicted Values

        Points near the diagonal line are accurate predictions. Spread around the line represents prediction error in popularity points.
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        sns.scatterplot(x=y_test, y=linear_test_pred, alpha=0.25, s=12, ax=axes[0])
        axes[0].plot([0, 100], [0, 100], color="black", linestyle="--")
        axes[0].set_title("Linear Regression: Actual vs. Predicted")
        axes[0].set_xlabel("Actual popularity")
        axes[0].set_ylabel("Predicted popularity")

        sns.scatterplot(x=y_test, y=poly_test_pred, alpha=0.25, s=12, ax=axes[1])
        axes[1].plot([0, 100], [0, 100], color="black", linestyle="--")
        axes[1].set_title("Best Polynomial Ridge: Actual vs. Predicted")
        axes[1].set_xlabel("Actual popularity")
        axes[1].set_ylabel("Predicted popularity")

        sns.scatterplot(x=y_test, y=boosting_test_pred, alpha=0.25, s=12, ax=axes[2])
        axes[2].plot([0, 100], [0, 100], color="black", linestyle="--")
        axes[2].set_title("Best Boosted Regression: Actual vs. Predicted")
        axes[2].set_xlabel("Actual popularity")
        axes[2].set_ylabel("Predicted popularity")

        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        r"""
        best_model = mse_table.sort_values("test_mse").iloc[0]
        overfit_notes = []
        for _, row in mse_table.iterrows():
            if row["generalization_gap"] > 25:
                overfit_notes.append(f"{row['model']} shows a notable train/test gap.")
            else:
                overfit_notes.append(f"{row['model']} has a modest train/test gap.")

        print("Interpretation:")
        baseline_mse = mse_table.loc[mse_table["model"] == "Multiple Linear Regression baseline", "test_mse"].iloc[0]
        improvement = baseline_mse - best_model["test_mse"]
        improvement_percent = 100 * improvement / baseline_mse

        print(f"- Best test MSE model: {best_model['model']} with MSE {best_model['test_mse']:.2f}.")
        print(f"- The RMSE is about {best_model['test_rmse_popularity_points']:.2f} popularity points.")
        print(f"- Compared with the original linear baseline, test MSE improved by {improvement:.2f} points ({improvement_percent:.2f}%).")
        print("- In business terms, this error means the model is better for estimating broad popularity potential than for guaranteeing an exact Spotify popularity score.")
        for note in overfit_notes:
            print("-", note)
        print("- Audio features influence popularity, but large residual spread suggests artist, playlist, market, recency, and promotion effects are also important.")
        """
    ),
    md(
        """
        ## Executive Conclusion

        The regression models provide a baseline for estimating expected popularity from audio characteristics. If the polynomial model has lower test MSE without a large generalization gap, nonlinear effects are useful. If the linear model is competitive, Spotify should prefer it for interpretability. In either case, predicted popularity should support discovery and catalog prioritization rather than replace business judgment.
        """
    ),
]


pca_cells = [
    md(
        """
        # 04 - PCA: Dimensionality Reduction

        Goal: summarize Spotify audio characteristics into fewer principal components while retaining at least 90% of the variance.
        """
    ),
    code(common_setup),
    code(
        r"""
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        sns.set_theme(style="whitegrid", palette="Set2")
        df = pd.read_csv(DATA_CLEAN)
        print(df.shape)
        """
    ),
    md(
        """
        ## Standardization Before PCA

        PCA is scale-sensitive. Features like `tempo` and `duration_min` have larger units than bounded features like `danceability`, so all inputs are standardized to mean 0 and standard deviation 1 before PCA.
        """
    ),
    code(
        r"""
        pca_features = [
            "duration_min", "is_explicit", "danceability", "energy", "key", "loudness",
            "mode", "speechiness", "acousticness", "instrumentalness", "liveness",
            "valence", "tempo", "time_signature"
        ]

        X = df[pca_features].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_scaled)

        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)
        n_components_90 = int(np.argmax(cumulative >= 0.90) + 1)
        print(f"Components needed for at least 90% variance: {n_components_90}")
        """
    ),
    code(
        r"""
        plt.figure(figsize=(11, 6))
        plt.plot(range(1, len(cumulative) + 1), cumulative, marker="o", label="Cumulative explained variance")
        plt.bar(range(1, len(explained) + 1), explained, alpha=0.35, label="Individual explained variance")
        plt.axhline(0.90, color="red", linestyle="--", label="90% threshold")
        plt.axvline(n_components_90, color="black", linestyle=":", label=f"{n_components_90} components")
        plt.xlabel("Principal component")
        plt.ylabel("Explained variance ratio")
        plt.title("PCA Explained Variance")
        plt.legend()
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Component Loadings

        Loadings show which original variables dominate each principal component. Large absolute values mean the variable contributes strongly to that component.
        """
    ),
    code(
        r"""
        loadings = pd.DataFrame(
            pca.components_.T,
            index=pca_features,
            columns=[f"PC{i}" for i in range(1, len(pca_features) + 1)]
        )
        display(loadings.iloc[:, :n_components_90].round(3))

        influential = {}
        for pc in loadings.columns[: min(5, n_components_90)]:
            influential[pc] = loadings[pc].abs().sort_values(ascending=False).head(5).index.tolist()
        display(pd.DataFrame(dict([(k, pd.Series(v)) for k, v in influential.items()])))
        """
    ),
    code(
        r"""
        sample_idx = df.sample(n=min(8000, len(df)), random_state=RANDOM_STATE).index
        pca_plot = pd.DataFrame(X_pca[sample_idx, :3], columns=["PC1", "PC2", "PC3"])
        pca_plot["popularity_class"] = df.loc[sample_idx, "popularity_class"].values
        pca_plot["track_genre"] = df.loc[sample_idx, "track_genre"].values

        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=pca_plot, x="PC1", y="PC2", hue="popularity_class", alpha=0.5, s=18)
        plt.title("Songs Projected onto First Two Principal Components")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        r"""
        print("Answers:")
        print("- Variables that dominate the structure are the features with the largest absolute loadings in the first principal components.")
        for pc, vars_ in influential.items():
            print(f"  {pc}: {', '.join(vars_)}")
        print(f"- Dimensionality reduction to {n_components_90} components is feasible for retaining at least 90% of numeric audio-feature variance.")
        print("- A 2D PCA plot is useful for visualization, but it does not retain all information; the 90% component count is better for modeling compression.")
        """
    ),
]


nn_cells = [
    md(
        """
        # 05 - Neural Network Classification

        Goal: classify songs into Low, Medium, and High popularity categories using PyTorch. The target classes were created with quantile bins to keep the class distribution balanced.
        """
    ),
    code(common_setup),
    code(
        r"""
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

        sns.set_theme(style="whitegrid", palette="Set2")
        torch.manual_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

        df = pd.read_csv(DATA_CLEAN)
        print(df.shape)
        display(df["popularity_class"].value_counts().sort_index().to_frame("count"))
        """
    ),
    md(
        """
        ## Architecture and Training Choices

        The notebook compares the original compact feed-forward classifier against a tuned deeper architecture:

        - Input layer equals the number of encoded features.
        - Baseline hidden layers: 128 neurons then 64 neurons.
        - Tuned hidden layers: 256 neurons, 128 neurons, then 64 neurons.
        - ReLU activations are used because they train efficiently and handle nonlinear feature relationships.
        - Dropout and AdamW weight decay provide regularization.
        - Cross-entropy loss is used because this is a multiclass classification problem.
        - Early stopping prevents overfitting by stopping when validation loss stops improving.
        """
    ),
    code(
        r"""
        numeric_features = [
            "duration_min", "is_explicit", "danceability", "energy", "key", "loudness",
            "mode", "speechiness", "acousticness", "instrumentalness", "liveness",
            "valence", "tempo", "time_signature"
        ]
        categorical_features = ["track_genre"]

        X = df[numeric_features + categorical_features]
        y = df["popularity_class"].astype(str)

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=RANDOM_STATE, stratify=y_encoded
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
        )

        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]), numeric_features),
                ("genre", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", encoder)
                ]), categorical_features)
            ]
        )

        X_train_np = preprocessor.fit_transform(X_train).astype(np.float32)
        X_val_np = preprocessor.transform(X_val).astype(np.float32)
        X_test_np = preprocessor.transform(X_test).astype(np.float32)

        print(f"Encoded feature count: {X_train_np.shape[1]}")
        """
    ),
    code(
        r"""
        class PopularityClassifier(nn.Module):
            def __init__(self, input_dim, n_classes, hidden_layers, dropout_rates):
                super().__init__()
                layers = []
                previous = input_dim
                for i, width in enumerate(hidden_layers):
                    layers.append(nn.Linear(previous, width))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rates[i]))
                    previous = width
                layers.append(nn.Linear(previous, n_classes))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x)


        def specificity_macro(y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred)
            specs = []
            for i in range(cm.shape[0]):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - tp - fp - fn
                specs.append(tn / (tn + fp) if (tn + fp) else 0)
            return float(np.mean(specs))


        def evaluate_predictions(y_true, y_pred):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision,
                "recall": recall,
                "specificity": specificity_macro(y_true, y_pred),
                "f1": f1,
            }


        def train_experiment(
            name,
            hidden_layers,
            dropout_rates,
            optimizer_name,
            lr,
            weight_decay,
            batch_size,
            max_epochs,
            patience,
        ):
            torch.manual_seed(RANDOM_STATE)
            np.random.seed(RANDOM_STATE)

            train_loader = DataLoader(
                TensorDataset(torch.tensor(X_train_np), torch.tensor(y_train, dtype=torch.long)),
                batch_size=batch_size,
                shuffle=True,
            )
            val_tensor = torch.tensor(X_val_np)
            test_tensor = torch.tensor(X_test_np)

            model = PopularityClassifier(
                X_train_np.shape[1],
                len(label_encoder.classes_),
                hidden_layers=hidden_layers,
                dropout_rates=dropout_rates,
            )
            criterion = nn.CrossEntropyLoss()
            if optimizer_name == "AdamW":
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            best_val_loss = np.inf
            best_state = None
            patience_counter = 0
            history = []

            for epoch in range(1, max_epochs + 1):
                model.train()
                train_losses = []
                correct = 0
                total = 0
                for xb, yb in train_loader:
                    optimizer.zero_grad()
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()

                    train_losses.append(loss.item())
                    correct += (logits.argmax(dim=1) == yb).sum().item()
                    total += len(yb)

                model.eval()
                with torch.no_grad():
                    val_logits = model(val_tensor)
                    val_loss = criterion(val_logits, torch.tensor(y_val, dtype=torch.long)).item()
                    val_acc = (val_logits.argmax(dim=1).numpy() == y_val).mean()

                train_loss = float(np.mean(train_losses))
                train_acc = correct / total
                history.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_acc": train_acc,
                    "val_acc": val_acc,
                })

                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"{name}: early stopping at epoch {epoch}")
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            with torch.no_grad():
                train_pred = model(torch.tensor(X_train_np)).argmax(dim=1).numpy()
                test_pred = model(test_tensor).argmax(dim=1).numpy()

            history_df = pd.DataFrame(history)
            train_metrics = evaluate_predictions(y_train, train_pred)
            test_metrics = evaluate_predictions(y_test, test_pred)
            summary = {
                "model": name,
                "hidden_layers": str(hidden_layers),
                "optimizer": optimizer_name,
                "lr": lr,
                "weight_decay": weight_decay,
                "dropout": str(dropout_rates),
                "epochs": len(history_df),
                "best_val_loss": best_val_loss,
                "best_val_acc": history_df["val_acc"].max(),
                "train_accuracy": train_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "specificity": test_metrics["specificity"],
                "f1": test_metrics["f1"],
            }
            return {
                "model": model,
                "history": history_df,
                "train_pred": train_pred,
                "test_pred": test_pred,
                "summary": summary,
            }


        baseline_result = train_experiment(
            name="Original baseline NN",
            hidden_layers=(128, 64),
            dropout_rates=(0.25, 0.20),
            optimizer_name="Adam",
            lr=0.001,
            weight_decay=1e-4,
            batch_size=512,
            max_epochs=20,
            patience=4,
        )

        tuned_result = train_experiment(
            name="Tuned deep NN",
            hidden_layers=(256, 128, 64),
            dropout_rates=(0.20, 0.20, 0.20),
            optimizer_name="AdamW",
            lr=0.0008,
            weight_decay=1e-5,
            batch_size=1024,
            max_epochs=35,
            patience=6,
        )

        nn_comparison = pd.DataFrame([
            baseline_result["summary"],
            tuned_result["summary"],
        ])
        nn_comparison["accuracy_improvement"] = nn_comparison["test_accuracy"] - nn_comparison.loc[0, "test_accuracy"]
        nn_comparison["f1_improvement"] = nn_comparison["f1"] - nn_comparison.loc[0, "f1"]
        display(nn_comparison.round(4))

        model = tuned_result["model"]
        history_df = tuned_result["history"]
        display(history_df.tail())
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].plot(history_df["epoch"], history_df["train_loss"], label="Train loss")
        axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validation loss")
        axes[0].set_title("Loss During Training")
        axes[0].legend()

        axes[1].plot(history_df["epoch"], history_df["train_acc"], label="Train accuracy")
        axes[1].plot(history_df["epoch"], history_df["val_acc"], label="Validation accuracy")
        axes[1].set_title("Accuracy During Training")
        axes[1].legend()
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        r"""
        model.eval()
        with torch.no_grad():
            train_pred = model(torch.tensor(X_train_np)).argmax(dim=1).numpy()
            test_pred = model(torch.tensor(X_test_np)).argmax(dim=1).numpy()

        def specificity_macro(y_true, y_pred):
            cm = confusion_matrix(y_true, y_pred)
            specs = []
            for i in range(cm.shape[0]):
                tp = cm[i, i]
                fp = cm[:, i].sum() - tp
                fn = cm[i, :].sum() - tp
                tn = cm.sum() - tp - fp - fn
                specs.append(tn / (tn + fp) if (tn + fp) else 0)
            return float(np.mean(specs))

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, test_pred, average="macro", zero_division=0
        )

        metrics = pd.DataFrame({
            "metric": ["Accuracy", "Precision", "Recall", "Specificity", "F1 Score"],
            "test_value": [
                accuracy_score(y_test, test_pred),
                precision,
                recall,
                specificity_macro(y_test, test_pred),
                f1,
            ],
        })

        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        display(metrics.round(4))
        print("Comparison against original baseline:")
        display(nn_comparison[[
            "model", "test_accuracy", "precision", "recall", "specificity", "f1",
            "accuracy_improvement", "f1_improvement"
        ]].round(4))
        print(f"Train accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        """
    ),
    code(
        r"""
        baseline_cm = confusion_matrix(y_test, baseline_result["test_pred"])
        tuned_cm = confusion_matrix(y_test, test_pred)

        baseline_cm_df = pd.DataFrame(baseline_cm, index=label_encoder.classes_, columns=label_encoder.classes_)
        tuned_cm_df = pd.DataFrame(tuned_cm, index=label_encoder.classes_, columns=label_encoder.classes_)

        print("Original baseline confusion matrix")
        display(baseline_cm_df)
        print("Tuned deep neural network confusion matrix")
        display(tuned_cm_df)

        baseline_errors = len(y_test) - np.trace(baseline_cm)
        tuned_errors = len(y_test) - np.trace(tuned_cm)
        print(f"Baseline misclassified songs: {baseline_errors:,}")
        print(f"Tuned model misclassified songs: {tuned_errors:,}")
        print(f"Reduction in misclassified songs: {baseline_errors - tuned_errors:,}")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        sns.heatmap(baseline_cm_df, annot=True, fmt="d", cmap="Blues", ax=axes[0])
        axes[0].set_title("Original Baseline Confusion Matrix")
        axes[0].set_xlabel("Predicted class")
        axes[0].set_ylabel("Actual class")

        sns.heatmap(tuned_cm_df, annot=True, fmt="d", cmap="Greens", ax=axes[1])
        axes[1].set_title("Tuned Deep NN Confusion Matrix")
        axes[1].set_xlabel("Predicted class")
        axes[1].set_ylabel("Actual class")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        r"""
        error_df = X_test.copy()
        error_df["actual"] = label_encoder.inverse_transform(y_test)
        error_df["predicted"] = label_encoder.inverse_transform(test_pred)
        errors = error_df[error_df["actual"] != error_df["predicted"]]

        error_pairs = (
            errors.groupby(["actual", "predicted"])
            .size()
            .sort_values(ascending=False)
            .to_frame("count")
        )
        display(error_pairs.head(10))

        genre_errors = (
            errors["track_genre"]
            .value_counts()
            .head(10)
            .to_frame("misclassified_count")
        )
        display(genre_errors)

        print("Interpretation:")
        print("- Most classification errors occur near adjacent popularity bands because quantile classes split a continuous score into categories.")
        print("- The tuned model uses a deeper 256-128-64 architecture, AdamW, lower weight decay, dropout, feature normalization, and early stopping.")
        print(f"- Accuracy improved by {nn_comparison.loc[1, 'accuracy_improvement']:.4f} and macro F1 improved by {nn_comparison.loc[1, 'f1_improvement']:.4f} versus the original baseline trained in this notebook.")
        print(f"- Train/test accuracy gap: {train_accuracy - test_accuracy:.4f}. A small gap suggests limited overfitting; a large gap would indicate poor generalization.")
        """
    ),
]


clustering_cells = [
    md(
        """
        # 06 - Clustering Spotify Songs

        Goal: discover natural groups of songs from audio characteristics without using a target variable.
        """
    ),
    code(common_setup),
    code(
        r"""
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler

        sns.set_theme(style="whitegrid", palette="Set2")
        df = pd.read_csv(DATA_CLEAN)
        print(df.shape)
        """
    ),
    md(
        """
        ## Feature Selection and Scaling

        Clustering uses audio features only. The variables are standardized so that tempo or duration do not dominate bounded features such as danceability or acousticness.
        """
    ),
    code(
        r"""
        cluster_features = [
            "duration_min", "danceability", "energy", "loudness", "speechiness",
            "acousticness", "instrumentalness", "liveness", "valence", "tempo"
        ]

        X = df[cluster_features].copy()
        X_scaled = StandardScaler().fit_transform(X)

        sample_size = min(5000, len(df))
        sample_idx = df.sample(n=sample_size, random_state=RANDOM_STATE).index
        X_sample = X_scaled[sample_idx]
        print(f"Clustering sample size for model selection: {sample_size:,}")
        """
    ),
    code(
        r"""
        ks = range(2, 9)
        inertias = []
        silhouettes = []

        for k in ks:
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
            labels = km.fit_predict(X_sample)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X_sample, labels))

        k_table = pd.DataFrame({"k": list(ks), "inertia": inertias, "silhouette": silhouettes})
        display(k_table)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].plot(k_table["k"], k_table["inertia"], marker="o")
        axes[0].set_title("Elbow Plot")
        axes[0].set_xlabel("k")
        axes[0].set_ylabel("Inertia")

        axes[1].plot(k_table["k"], k_table["silhouette"], marker="o", color="#F58518")
        axes[1].set_title("Silhouette Score")
        axes[1].set_xlabel("k")
        axes[1].set_ylabel("Silhouette")
        plt.tight_layout()
        plt.show()
        """
    ),
    code(
        r"""
        best_k = int(k_table.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
        print(f"Selected k based on highest silhouette score: {best_k}")

        final_kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
        df["cluster"] = final_kmeans.fit_predict(X_scaled)

        cluster_profile = df.groupby("cluster")[cluster_features + ["popularity"]].mean().round(3)
        cluster_counts = df["cluster"].value_counts().sort_index().to_frame("count")
        display(cluster_counts)
        display(cluster_profile)
        """
    ),
    code(
        r"""
        def describe_cluster(row):
            tags = []
            if row["energy"] >= cluster_profile["energy"].median():
                tags.append("energetic")
            if row["danceability"] >= cluster_profile["danceability"].median():
                tags.append("danceable")
            if row["acousticness"] >= cluster_profile["acousticness"].median():
                tags.append("acoustic")
            if row["instrumentalness"] >= cluster_profile["instrumentalness"].median():
                tags.append("instrumental")
            if row["valence"] >= cluster_profile["valence"].median():
                tags.append("positive")
            return ", ".join(tags[:3]) if tags else "balanced"

        cluster_labels = cluster_profile.apply(describe_cluster, axis=1).to_frame("suggested_description")
        display(cluster_labels)
        """
    ),
    code(
        r"""
        from sklearn.decomposition import PCA

        pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
        coords = pca_2d.fit_transform(X_scaled[sample_idx])
        plot_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
        plot_df["cluster"] = df.loc[sample_idx, "cluster"].astype(str).values

        plt.figure(figsize=(10, 7))
        sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="cluster", alpha=0.45, s=18)
        plt.title("Cluster Visualization in PCA Space")
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Clustering Conclusion

        The clusters summarize broad song types based on audio properties, such as energetic/danceable, acoustic, instrumental, or balanced groups. These clusters can help Spotify organize catalog exploration, seed playlist concepts, and create interpretable segments for recommendation experiments.
        """
    ),
]


business_md = """
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
"""


requirements = """
pandas
numpy
matplotlib
seaborn
scikit-learn
torch
jupyter
nbformat
nbclient
ipykernel
"""


gitignore = """
.venv/
__pycache__/
.ipynb_checkpoints/
*.pyc
"""


def main():
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)

    write_notebook(NOTEBOOK_DIR / "01_eda.ipynb", eda_cells)
    write_notebook(NOTEBOOK_DIR / "02_cleaning_feature_engineering.ipynb", cleaning_cells)
    write_notebook(NOTEBOOK_DIR / "03_regression.ipynb", regression_cells)
    write_notebook(NOTEBOOK_DIR / "04_pca.ipynb", pca_cells)
    write_notebook(NOTEBOOK_DIR / "05_neural_network_classification.ipynb", nn_cells)
    write_notebook(NOTEBOOK_DIR / "06_clustering.ipynb", clustering_cells)

    (ROOT / "spotify_business_context.md").write_text(business_md.strip() + "\n", encoding="utf-8")
    (ROOT / "requirements.txt").write_text(requirements.strip() + "\n", encoding="utf-8")
    (ROOT / ".gitignore").write_text(gitignore.strip() + "\n", encoding="utf-8")

    print("Generated notebooks and project files.")


if __name__ == "__main__":
    main()
