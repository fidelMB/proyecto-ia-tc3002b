import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

RANDOM_STATE = 42
torch.set_num_threads(4)


def set_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


@dataclass
class Config:
    name: str
    hidden_layers: tuple[int, ...]
    dropout: float
    lr: float
    weight_decay: float
    batch_norm: bool
    label_smoothing: float
    batch_size: int = 1024
    max_epochs: int = 35
    patience: int = 6


class PopularityClassifier(nn.Module):
    def __init__(self, input_dim, n_classes, hidden_layers, dropout, batch_norm):
        super().__init__()
        layers = []
        previous = input_dim
        for width in hidden_layers:
            layers.append(nn.Linear(previous, width))
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_config(config, data):
    set_seed()
    X_train_np, X_val_np, X_test_np, y_train, y_val, y_test, classes = data

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train_np), torch.tensor(y_train, dtype=torch.long)),
        batch_size=config.batch_size,
        shuffle=True,
        # Avoid a size-1 final batch, which BatchNorm cannot normalize.
        drop_last=config.batch_norm,
    )
    val_tensor = torch.tensor(X_val_np)
    test_tensor = torch.tensor(X_test_np)

    model = PopularityClassifier(
        X_train_np.shape[1],
        len(classes),
        config.hidden_layers,
        config.dropout,
        config.batch_norm,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    best_val_loss = np.inf
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, config.max_epochs + 1):
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
            val_pred = val_logits.argmax(dim=1).numpy()
            val_acc = accuracy_score(y_val, val_pred)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
                "train_acc": correct / total,
                "val_acc": val_acc,
            }
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_pred = model(torch.tensor(X_train_np)).argmax(dim=1).numpy()
        test_pred = model(test_tensor).argmax(dim=1).numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, test_pred, average="macro", zero_division=0
    )
    return {
        "name": config.name,
        "epochs": len(history),
        "best_val_loss": best_val_loss,
        "best_val_acc": max(row["val_acc"] for row in history),
        "train_accuracy": accuracy_score(y_train, train_pred),
        "test_accuracy": accuracy_score(y_test, test_pred),
        "precision": precision,
        "recall": recall,
        "specificity": specificity_macro(y_test, test_pred),
        "f1": f1,
        "confusion_matrix": confusion_matrix(y_test, test_pred).tolist(),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    df = pd.read_csv(root / "data" / "processed" / "spotify_tracks_clean.csv")
    numeric_features = [
        "duration_min",
        "is_explicit",
        "danceability",
        "energy",
        "key",
        "loudness",
        "mode",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
        "time_signature",
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

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "genre",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", encoder),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    data = (
        preprocessor.fit_transform(X_train).astype(np.float32),
        preprocessor.transform(X_val).astype(np.float32),
        preprocessor.transform(X_test).astype(np.float32),
        y_train,
        y_val,
        y_test,
        label_encoder.classes_,
    )

    configs = [
        Config("baseline_128_64", (128, 64), 0.25, 0.001, 1e-4, False, 0.0),
        Config("wide_256_128_dropout_015", (256, 128), 0.15, 0.001, 1e-5, False, 0.0),
        Config("wide_bn_256_128_dropout_015", (256, 128), 0.15, 0.001, 1e-5, True, 0.0),
        Config("deep_256_128_64_dropout_020", (256, 128, 64), 0.20, 0.0008, 1e-5, False, 0.0),
        Config("deep_bn_256_128_64_smooth", (256, 128, 64), 0.15, 0.0008, 1e-5, True, 0.02),
        Config("compact_low_dropout", (128, 64), 0.10, 0.0012, 1e-5, False, 0.0),
    ]

    results = []
    for config in configs:
        result = train_config(config, data)
        results.append(result)
        printable = {k: v for k, v in result.items() if k != "confusion_matrix"}
        print(printable, flush=True)

    results_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "confusion_matrix"} for r in results]
    )
    print("\nRANKED BY VALIDATION ACCURACY")
    print(results_df.sort_values("best_val_acc", ascending=False).to_string(index=False))

    best = max(results, key=lambda row: row["best_val_acc"])
    print("\nSELECTED CONFIG (by validation):", best["name"])
    print(f"Held-out test accuracy: {best['test_accuracy']:.4f}")
    print("Test confusion matrix:")
    print(np.array(best["confusion_matrix"]))


if __name__ == "__main__":
    main()
