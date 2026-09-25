from pathlib import Path
import json

import numpy as np
import torch

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EMBEDDING_DIR = Path("outputs/scaled_v2/embeddings")
OUTPUT_DIR = Path("outputs/scaled_v2/ridge")


def load(split):
    X, y = [], []

    for path in sorted(EMBEDDING_DIR.glob("*.pt")):
        item = torch.load(path, map_location="cpu")

        if str(item["Split"]).upper() != split:
            continue

        X.append(item["embedding"].float().numpy())
        y.append(float(item["EF"]))

    return (
        np.stack(X).astype(np.float32),
        np.asarray(y, dtype=np.float32),
    )


def metrics(y, pred):
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(
            np.sqrt(mean_squared_error(y, pred))
        ),
        "r2": float(r2_score(y, pred)),
    }


def main():
    X_train, y_train = load("TRAIN")
    X_val, y_val = load("VAL")

    print("TRAIN:", X_train.shape)
    print("VAL:", X_val.shape)

    # Mean baseline
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)

    dummy_pred = dummy.predict(X_val)
    dummy_metrics = metrics(y_val, dummy_pred)

    # Ridge with TRAIN-only CV
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    search = GridSearchCV(
        model,
        {
            "ridge__alpha":
                np.logspace(-3, 4, 32)
        },
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)

    pred = search.predict(X_val)

    ridge_metrics = metrics(y_val, pred)

    result = {
        "n_train": len(y_train),
        "n_val": len(y_val),
        "best_alpha": float(
            search.best_params_["ridge__alpha"]
        ),
        "train_cv_mae": float(
            -search.best_score_
        ),
        "mean_baseline": dummy_metrics,
        "ridge": ridge_metrics,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
    ) as f:
        json.dump(result, f, indent=2)

    print("\nBest alpha:")
    print(result["best_alpha"])

    print("\nTRAIN CV MAE:")
    print(result["train_cv_mae"])

    print("\nMean baseline:")
    print(dummy_metrics)

    print("\nFrozen EchoPrime + Ridge:")
    print(ridge_metrics)


if __name__ == "__main__":
    main()
