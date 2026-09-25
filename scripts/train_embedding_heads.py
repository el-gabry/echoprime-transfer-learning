from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from echoprime_transfer.models import (
    EFRegressionHead,
    MultiTaskRegressionHead,
    trainable_parameter_count,
)


EMBEDDING_DIR = Path("outputs/mini_v1/embeddings")
OUTPUT_DIR = Path("outputs/mini_v1/model_comparison")

SEED = 42
BATCH_SIZE = 32
MAX_EPOCHS = 400
PATIENCE = 35

TARGETS = ["EF", "EDV", "ESV"]


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records():
    records = []

    for path in sorted(EMBEDDING_DIR.glob("*.pt")):
        item = torch.load(path, map_location="cpu")

        records.append(
            {
                "FileName": item["FileName"],
                "embedding": item["embedding"].float().numpy(),
                "EF": float(item["EF"]),
                "EDV": float(item["EDV"]),
                "ESV": float(item["ESV"]),
                "Split": str(item["Split"]).upper(),
            }
        )

    return records


def build_arrays(records, split):
    subset = [r for r in records if r["Split"] == split]

    X = np.stack(
        [r["embedding"] for r in subset]
    ).astype(np.float32)

    y = np.array(
        [
            [r["EF"], r["EDV"], r["ESV"]]
            for r in subset
        ],
        dtype=np.float32,
    )

    names = [r["FileName"] for r in subset]

    return X, y, names


def scale_fit(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_with_internal_validation(
    model_factory,
    X,
    y,
    device,
):
    indices = np.arange(len(X))

    train_idx, monitor_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=SEED,
    )

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_monitor = X[monitor_idx]
    y_monitor = y[monitor_idx]

    x_mean, x_std = scale_fit(X_train)
    y_mean, y_std = scale_fit(y_train)

    X_train = (X_train - x_mean) / x_std
    X_monitor = (X_monitor - x_mean) / x_std

    y_train_scaled = (y_train - y_mean) / y_std

    dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train_scaled),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = model_factory().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    criterion = nn.SmoothL1Loss()

    best_state = None
    best_score = float("inf")
    best_epoch = 0
    patience_counter = 0

    X_monitor_t = torch.tensor(
        X_monitor,
        device=device,
    )

    for epoch in range(MAX_EPOCHS):

        model.train()

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            pred = model(xb)

            if pred.ndim == 1:
                pred = pred.unsqueeze(1)

            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.inference_mode():
            pred = model(X_monitor_t)

            if pred.ndim == 1:
                pred = pred.unsqueeze(1)

            pred = pred.cpu().numpy()

        pred_original = (
            pred * y_std + y_mean
        )

        # Dimensionless criterion so EF / EDV / ESV
        # contribute comparably.
        normalized_mae = np.mean(
            np.mean(
                np.abs(
                    pred_original - y_monitor
                ),
                axis=0,
            )
            / y_std
        )

        if normalized_mae < best_score - 1e-5:
            best_score = normalized_mae
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            break

    return best_epoch


def refit_all_train(
    model_factory,
    X_train,
    y_train,
    epochs,
    device,
):
    x_mean, x_std = scale_fit(X_train)
    y_mean, y_std = scale_fit(y_train)

    X_scaled = (X_train - x_mean) / x_std
    y_scaled = (y_train - y_mean) / y_std

    dataset = TensorDataset(
        torch.tensor(X_scaled),
        torch.tensor(y_scaled),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    seed_everything()

    model = model_factory().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )

    criterion = nn.SmoothL1Loss()

    for _ in range(epochs):

        model.train()

        for xb, yb in loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)

            pred = model(xb)

            if pred.ndim == 1:
                pred = pred.unsqueeze(1)

            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    return model, x_mean, x_std, y_mean, y_std


def evaluate(
    model,
    X,
    y,
    x_mean,
    x_std,
    y_mean,
    y_std,
    device,
):
    X_scaled = (
        (X - x_mean) / x_std
    ).astype(np.float32)

    model.eval()

    with torch.inference_mode():
        pred = model(
            torch.tensor(
                X_scaled,
                device=device,
            )
        )

        if pred.ndim == 1:
            pred = pred.unsqueeze(1)

    pred = pred.cpu().numpy()

    pred = pred * y_std + y_mean

    return pred


def run_model(
    name,
    model_factory,
    X_train,
    y_train,
    X_val,
    y_val,
    val_names,
    target_names,
    device,
):
    print(f"\n{'=' * 60}")
    print(name)
    print("=" * 60)

    best_epoch = train_with_internal_validation(
        model_factory,
        X_train,
        y_train,
        device,
    )

    print("Selected epoch:", best_epoch)

    model, x_mean, x_std, y_mean, y_std = refit_all_train(
        model_factory,
        X_train,
        y_train,
        best_epoch,
        device,
    )

    pred = evaluate(
        model,
        X_val,
        y_val,
        x_mean,
        x_std,
        y_mean,
        y_std,
        device,
    )

    result_metrics = {}

    for i, target in enumerate(target_names):
        result_metrics[target] = metrics(
            y_val[:, i],
            pred[:, i],
        )

        print(
            target,
            result_metrics[target],
        )

    print(
        "Trainable parameters:",
        trainable_parameter_count(model),
    )

    result_dir = OUTPUT_DIR / name
    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions = pd.DataFrame(
        {"FileName": val_names}
    )

    for i, target in enumerate(target_names):
        predictions[f"{target}_true"] = y_val[:, i]
        predictions[f"{target}_pred"] = pred[:, i]

    predictions.to_csv(
        result_dir / "val_predictions.csv",
        index=False,
    )

    result = {
        "model": name,
        "selected_epoch": best_epoch,
        "trainable_parameters":
            trainable_parameter_count(model),
        "metrics": result_metrics,
    }

    with open(
        result_dir / "metrics.json",
        "w",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
        )

    torch.save(
        model.state_dict(),
        result_dir / "model.pt",
    )

    return result


def main():
    seed_everything()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    records = load_records()

    X_train, y_train_all, _ = build_arrays(
        records,
        "TRAIN",
    )

    X_val, y_val_all, val_names = build_arrays(
        records,
        "VAL",
    )

    print("TRAIN:", X_train.shape)
    print("VAL:", X_val.shape)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Model 1 — nonlinear EF-only head
    # --------------------------------------------------

    ef_result = run_model(
        name="frozen_mlp_ef",
        model_factory=lambda: EFRegressionHead(
            embedding_dim=512,
            hidden_dim=128,
            dropout=0.20,
        ),
        X_train=X_train,
        y_train=y_train_all[:, [0]],
        X_val=X_val,
        y_val=y_val_all[:, [0]],
        val_names=val_names,
        target_names=["EF"],
        device=device,
    )

    # --------------------------------------------------
    # Model 2 — multi-task EF / EDV / ESV
    # --------------------------------------------------

    multitask_result = run_model(
        name="frozen_multitask",
        model_factory=lambda: MultiTaskRegressionHead(
            embedding_dim=512,
            hidden_dim=128,
            dropout=0.20,
        ),
        X_train=X_train,
        y_train=y_train_all,
        X_val=X_val,
        y_val=y_val_all,
        val_names=val_names,
        target_names=TARGETS,
        device=device,
    )

    comparison = {
        "frozen_mlp_ef": ef_result,
        "frozen_multitask": multitask_result,
    }

    with open(
        OUTPUT_DIR / "comparison.json",
        "w",
    ) as f:
        json.dump(
            comparison,
            f,
            indent=2,
        )

    print("\nComparison saved to:")
    print(OUTPUT_DIR / "comparison.json")


if __name__ == "__main__":
    main()
