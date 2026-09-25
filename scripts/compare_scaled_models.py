from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EMBEDDING_DIR = Path("outputs/scaled_v2/embeddings")

MLP_PRED = Path(
    "outputs/scaled_v2/model_comparison/"
    "frozen_mlp_ef/val_predictions.csv"
)

MULTI_PRED = Path(
    "outputs/scaled_v2/model_comparison/"
    "frozen_multitask/val_predictions.csv"
)

OUTPUT_DIR = Path(
    "outputs/scaled_v2/paired_comparison"
)

SEED = 42
N_BOOTSTRAP = 5000


def load_embeddings(split):
    rows = []

    for path in sorted(EMBEDDING_DIR.glob("*.pt")):
        item = torch.load(path, map_location="cpu")

        if str(item["Split"]).upper() != split:
            continue

        rows.append(
            {
                "FileName": item["FileName"],
                "embedding": item["embedding"].float().numpy(),
                "EF": float(item["EF"]),
            }
        )

    X = np.stack(
        [r["embedding"] for r in rows]
    ).astype(np.float32)

    y = np.asarray(
        [r["EF"] for r in rows],
        dtype=np.float32,
    )

    names = [
        r["FileName"]
        for r in rows
    ]

    return X, y, names


def train_ridge(
    X_train,
    y_train,
):
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    search = GridSearchCV(
        pipeline,
        {
            "ridge__alpha":
                np.logspace(-3, 4, 32)
        },
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(
        X_train,
        y_train,
    )

    return search


def bootstrap_difference(
    errors_a,
    errors_b,
    n_bootstrap=N_BOOTSTRAP,
):
    """
    Returns MAE(A) - MAE(B).

    Negative value:
        A better than B.

    Positive value:
        B better than A.
    """

    rng = np.random.default_rng(SEED)

    n = len(errors_a)

    diffs = []

    for _ in range(n_bootstrap):
        idx = rng.integers(
            0,
            n,
            size=n,
        )

        diff = (
            errors_a[idx].mean()
            - errors_b[idx].mean()
        )

        diffs.append(diff)

    diffs = np.asarray(diffs)

    lower, upper = np.percentile(
        diffs,
        [2.5, 97.5],
    )

    return {
        "mean_difference":
            float(diffs.mean()),
        "lower_95":
            float(lower),
        "upper_95":
            float(upper),
        "probability_a_better":
            float(
                np.mean(diffs < 0)
            ),
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------
    # Ridge predictions
    # -----------------------------------------

    X_train, y_train, _ = load_embeddings(
        "TRAIN"
    )

    X_val, y_val, val_names = load_embeddings(
        "VAL"
    )

    ridge = train_ridge(
        X_train,
        y_train,
    )

    ridge_pred = ridge.predict(
        X_val
    )

    ridge_df = pd.DataFrame(
        {
            "FileName": val_names,
            "EF_true": y_val,
            "ridge": ridge_pred,
        }
    )

    # -----------------------------------------
    # Load neural-head predictions
    # -----------------------------------------

    mlp = pd.read_csv(
        MLP_PRED
    )[
        [
            "FileName",
            "EF_pred",
        ]
    ].rename(
        columns={
            "EF_pred": "mlp"
        }
    )

    multi = pd.read_csv(
        MULTI_PRED
    )[
        [
            "FileName",
            "EF_pred",
        ]
    ].rename(
        columns={
            "EF_pred": "multitask"
        }
    )

    df = (
        ridge_df
        .merge(
            mlp,
            on="FileName",
            how="inner",
        )
        .merge(
            multi,
            on="FileName",
            how="inner",
        )
    )

    assert len(df) == 200

    # -----------------------------------------
    # Absolute errors
    # -----------------------------------------

    df["ridge_error"] = np.abs(
        df["ridge"]
        - df["EF_true"]
    )

    df["mlp_error"] = np.abs(
        df["mlp"]
        - df["EF_true"]
    )

    df["multitask_error"] = np.abs(
        df["multitask"]
        - df["EF_true"]
    )

    print("\nMAE")
    print(
        "Ridge:",
        df["ridge_error"].mean(),
    )

    print(
        "MLP:",
        df["mlp_error"].mean(),
    )

    print(
        "Multi-task:",
        df["multitask_error"].mean(),
    )

    # -----------------------------------------
    # Paired bootstrap
    # -----------------------------------------

    comparisons = {}

    comparisons["ridge_vs_mlp"] = (
        bootstrap_difference(
            df["ridge_error"].to_numpy(),
            df["mlp_error"].to_numpy(),
        )
    )

    comparisons[
        "ridge_vs_multitask"
    ] = bootstrap_difference(
        df["ridge_error"].to_numpy(),
        df["multitask_error"].to_numpy(),
    )

    comparisons[
        "multitask_vs_mlp"
    ] = bootstrap_difference(
        df["multitask_error"].to_numpy(),
        df["mlp_error"].to_numpy(),
    )

    print("\nPaired bootstrap:")
    for name, result in comparisons.items():
        print(
            f"\n{name}"
        )
        print(result)

    df.to_csv(
        OUTPUT_DIR
        / "paired_predictions.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR
        / "bootstrap_comparison.json",
        "w",
    ) as f:
        json.dump(
            comparisons,
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
