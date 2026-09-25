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
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.models.video import mvit_v2_s

from echoprime_transfer.preprocessing import (
    preprocess_avi_for_echoprime,
)


SEED = 42

MANIFEST = Path("data/EchoNet-Dynamic/mini_manifest.csv")
VIDEO_DIR = Path("data/EchoNet-Dynamic/Videos")
EMBEDDING_DIR = Path("outputs/mini_v1/embeddings")

CHECKPOINT = Path(
    "external/EchoPrime/model_data/weights/"
    "echo_prime_encoder.pt"
)

OUTPUT_DIR = Path(
    "outputs/mini_v1/two_stage_finetune"
)

# Stage 1: frozen encoder, train head only
STAGE1_MAX_EPOCHS = 200
STAGE1_PATIENCE = 25
STAGE1_LR = 1e-3
STAGE1_BATCH_SIZE = 32

# Stage 2: adapt final EchoPrime block
STAGE2_MAX_EPOCHS = 15
STAGE2_PATIENCE = 4

ENCODER_LR = 3e-6
HEAD_LR = 1e-4

VIDEO_BATCH_SIZE = 1
ACCUMULATION_STEPS = 4

WEIGHT_DECAY = 1e-4


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EFHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class EchoDataset(Dataset):
    def __init__(
        self,
        dataframe,
        y_mean,
        y_std,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.y_mean = float(y_mean)
        self.y_std = float(y_std)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        path = VIDEO_DIR / f"{row['FileName']}.avi"

        video = preprocess_avi_for_echoprime(path)

        ef = float(row["EF"])

        ef_z = (
            ef - self.y_mean
        ) / self.y_std

        return (
            video,
            torch.tensor(
                ef_z,
                dtype=torch.float32,
            ),
        )


class EchoPrimeEF(nn.Module):
    def __init__(self, head_state=None):
        super().__init__()

        encoder = mvit_v2_s()

        encoder.head[1] = nn.Linear(
            encoder.head[1].in_features,
            512,
        )

        state = torch.load(
            CHECKPOINT,
            map_location="cpu",
        )

        encoder.load_state_dict(state)

        self.encoder = encoder
        self.regressor = EFHead()

        if head_state is not None:
            self.regressor.load_state_dict(
                head_state
            )

    def forward(self, x):
        embedding = self.encoder(x)
        return self.regressor(embedding)


def freeze_encoder(model):
    for p in model.encoder.parameters():
        p.requires_grad = False


def unfreeze_last_block(model):
    freeze_encoder(model)

    for p in model.encoder.blocks[-1].parameters():
        p.requires_grad = True

    for p in model.encoder.norm.parameters():
        p.requires_grad = True

    for p in model.encoder.head.parameters():
        p.requires_grad = True

    for p in model.regressor.parameters():
        p.requires_grad = True


def load_embedding_dataframe():
    records = []

    for path in sorted(
        EMBEDDING_DIR.glob("*.pt")
    ):
        item = torch.load(
            path,
            map_location="cpu",
        )

        records.append(
            {
                "FileName": item["FileName"],
                "embedding":
                    item["embedding"].float(),
                "EF": float(item["EF"]),
                "Split":
                    str(item["Split"]).upper(),
            }
        )

    return records


def embedding_arrays(records, filenames):
    mapping = {
        x["FileName"]: x
        for x in records
    }

    X = torch.stack(
        [
            mapping[name]["embedding"]
            for name in filenames
        ]
    )

    y = torch.tensor(
        [
            mapping[name]["EF"]
            for name in filenames
        ],
        dtype=torch.float32,
    )

    return X, y


def evaluate_head(
    model,
    X,
    y,
    y_mean,
    y_std,
    device,
):
    model.eval()

    with torch.inference_mode():
        pred_z = model(
            X.to(device)
        ).cpu().numpy()

    pred = pred_z * y_std + y_mean
    true = y.numpy()

    return float(
        mean_absolute_error(
            true,
            pred,
        )
    )


def train_stage1_select(
    fit_df,
    monitor_df,
    records,
    device,
):
    print("\n========== STAGE 1 ==========")
    print("Frozen EchoPrime embeddings")
    print("Train prediction head only")

    fit_names = fit_df["FileName"].tolist()
    monitor_names = monitor_df["FileName"].tolist()

    X_fit, y_fit = embedding_arrays(
        records,
        fit_names,
    )

    X_monitor, y_monitor = embedding_arrays(
        records,
        monitor_names,
    )

    y_mean = y_fit.mean().item()
    y_std = y_fit.std().item()

    y_fit_z = (
        y_fit - y_mean
    ) / y_std

    dataset = TensorDataset(
        X_fit,
        y_fit_z,
    )

    loader = DataLoader(
        dataset,
        batch_size=STAGE1_BATCH_SIZE,
        shuffle=True,
    )

    model = EFHead().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=STAGE1_LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.SmoothL1Loss()

    best_state = None
    best_mae = float("inf")
    best_epoch = 1
    patience = 0

    for epoch in range(
        1,
        STAGE1_MAX_EPOCHS + 1,
    ):
        model.train()

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(xb)

            loss = criterion(
                pred,
                yb,
            )

            loss.backward()
            optimizer.step()

        monitor_mae = evaluate_head(
            model,
            X_monitor,
            y_monitor,
            y_mean,
            y_std,
            device,
        )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Stage1 epoch {epoch:03d} | "
                f"monitor MAE {monitor_mae:.4f}"
            )

        if monitor_mae < best_mae - 1e-4:
            best_mae = monitor_mae
            best_epoch = epoch
            best_state = deepcopy(
                model.state_dict()
            )
            patience = 0
        else:
            patience += 1

        if patience >= STAGE1_PATIENCE:
            break

    print(
        "Stage 1 best epoch:",
        best_epoch,
    )

    print(
        "Stage 1 monitor MAE:",
        best_mae,
    )

    return (
        best_state,
        best_epoch,
        y_mean,
        y_std,
    )


def evaluate_video_model(
    model,
    loader,
    y_mean,
    y_std,
    device,
):
    model.eval()

    true_z = []
    pred_z = []

    with torch.inference_mode():
        for videos, targets in loader:
            videos = videos.to(
                device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=device.type == "cuda",
            ):
                outputs = model(videos)

            true_z.extend(
                targets.numpy()
            )

            pred_z.extend(
                outputs.cpu().numpy()
            )

    true_z = np.asarray(true_z)
    pred_z = np.asarray(pred_z)

    true = true_z * y_std + y_mean
    pred = pred_z * y_std + y_mean

    return {
        "mae": float(
            mean_absolute_error(
                true,
                pred,
            )
        ),
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    true,
                    pred,
                )
            )
        ),
        "r2": float(
            r2_score(
                true,
                pred,
            )
        ),
    }


def train_stage2_select(
    fit_df,
    monitor_df,
    stage1_state,
    y_mean,
    y_std,
    device,
):
    print("\n========== STAGE 2 ==========")
    print("Unfreeze last EchoPrime block")

    model = EchoPrimeEF(
        head_state=stage1_state
    ).to(device)

    unfreeze_last_block(model)

    fit_loader = DataLoader(
        EchoDataset(
            fit_df,
            y_mean,
            y_std,
        ),
        batch_size=VIDEO_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    monitor_loader = DataLoader(
        EchoDataset(
            monitor_df,
            y_mean,
            y_std,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    encoder_params = (
        list(
            model.encoder.blocks[-1].parameters()
        )
        + list(
            model.encoder.norm.parameters()
        )
        + list(
            model.encoder.head.parameters()
        )
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_params,
                "lr": ENCODER_LR,
            },
            {
                "params":
                    model.regressor.parameters(),
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.SmoothL1Loss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    best_state = None
    best_mae = float("inf")
    best_epoch = 1
    patience = 0

    for epoch in range(
        1,
        STAGE2_MAX_EPOCHS + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        for step, (
            videos,
            targets,
        ) in enumerate(
            fit_loader,
            start=1,
        ):
            videos = videos.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=device.type == "cuda",
            ):
                pred = model(videos)

                loss = criterion(
                    pred,
                    targets,
                )

                loss = (
                    loss /
                    ACCUMULATION_STEPS
                )

            scaler.scale(
                loss
            ).backward()

            if (
                step % ACCUMULATION_STEPS == 0
                or step == len(fit_loader)
            ):
                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

        metrics = evaluate_video_model(
            model,
            monitor_loader,
            y_mean,
            y_std,
            device,
        )

        print(
            f"Stage2 epoch {epoch:02d} | "
            f"monitor MAE "
            f"{metrics['mae']:.4f} | "
            f"R2 {metrics['r2']:.4f}"
        )

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            best_epoch = epoch
            best_state = deepcopy(
                model.state_dict()
            )
            patience = 0
        else:
            patience += 1

        if patience >= STAGE2_PATIENCE:
            print(
                "Stage 2 early stopping."
            )
            break

    print(
        "Stage 2 best epoch:",
        best_epoch,
    )

    return best_epoch


def refit_stage1_all_train(
    train_df,
    records,
    epochs,
    device,
):
    names = train_df[
        "FileName"
    ].tolist()

    X, y = embedding_arrays(
        records,
        names,
    )

    y_mean = y.mean().item()
    y_std = y.std().item()

    y_z = (
        y - y_mean
    ) / y_std

    loader = DataLoader(
        TensorDataset(
            X,
            y_z,
        ),
        batch_size=STAGE1_BATCH_SIZE,
        shuffle=True,
    )

    seed_everything()

    model = EFHead().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=STAGE1_LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.SmoothL1Loss()

    for _ in range(epochs):
        model.train()

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            loss = criterion(
                model(xb),
                yb,
            )

            loss.backward()
            optimizer.step()

    return (
        deepcopy(model.state_dict()),
        y_mean,
        y_std,
    )


def refit_stage2_all_train(
    train_df,
    stage1_state,
    y_mean,
    y_std,
    epochs,
    device,
):
    model = EchoPrimeEF(
        head_state=stage1_state
    ).to(device)

    unfreeze_last_block(model)

    loader = DataLoader(
        EchoDataset(
            train_df,
            y_mean,
            y_std,
        ),
        batch_size=VIDEO_BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    encoder_params = (
        list(
            model.encoder.blocks[-1].parameters()
        )
        + list(
            model.encoder.norm.parameters()
        )
        + list(
            model.encoder.head.parameters()
        )
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_params,
                "lr": ENCODER_LR,
            },
            {
                "params":
                    model.regressor.parameters(),
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.SmoothL1Loss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda",
    )

    for epoch in range(
        1,
        epochs + 1,
    ):
        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        for step, (
            videos,
            targets,
        ) in enumerate(
            loader,
            start=1,
        ):
            videos = videos.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type="cuda",
                enabled=device.type == "cuda",
            ):
                loss = criterion(
                    model(videos),
                    targets,
                )

                loss = (
                    loss /
                    ACCUMULATION_STEPS
                )

            scaler.scale(
                loss
            ).backward()

            if (
                step % ACCUMULATION_STEPS == 0
                or step == len(loader)
            ):
                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

        print(
            f"Final refit Stage2 "
            f"epoch {epoch}/{epochs}"
        )

    return model


def main():
    seed_everything()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    df = pd.read_csv(MANIFEST)

    train_df = df[
        df["Split"].str.upper()
        == "TRAIN"
    ].copy()

    val_df = df[
        df["Split"].str.upper()
        == "VAL"
    ].copy()

    fit_df, monitor_df = train_test_split(
        train_df,
        test_size=0.20,
        random_state=SEED,
    )

    print("TRAIN total:", len(train_df))
    print("Fit:", len(fit_df))
    print("Monitor:", len(monitor_df))
    print("Official VAL:", len(val_df))

    records = load_embedding_dataframe()

    # ---------------------------------------
    # Select Stage 1 epoch using TRAIN only
    # ---------------------------------------

    (
        stage1_state,
        stage1_epoch,
        fit_mean,
        fit_std,
    ) = train_stage1_select(
        fit_df,
        monitor_df,
        records,
        device,
    )

    # ---------------------------------------
    # Select Stage 2 epoch using TRAIN only
    # ---------------------------------------

    stage2_epoch = train_stage2_select(
        fit_df,
        monitor_df,
        stage1_state,
        fit_mean,
        fit_std,
        device,
    )

    # ---------------------------------------
    # Refit using ALL 200 TRAIN cases
    # ---------------------------------------

    print(
        "\n========== FINAL REFIT =========="
    )

    final_stage1_state, y_mean, y_std = (
        refit_stage1_all_train(
            train_df,
            records,
            stage1_epoch,
            device,
        )
    )

    final_model = refit_stage2_all_train(
        train_df,
        final_stage1_state,
        y_mean,
        y_std,
        stage2_epoch,
        device,
    )

    # ---------------------------------------
    # Official VAL evaluated once
    # ---------------------------------------

    val_loader = DataLoader(
        EchoDataset(
            val_df,
            y_mean,
            y_std,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    metrics = evaluate_video_model(
        final_model,
        val_loader,
        y_mean,
        y_std,
        device,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        final_model.state_dict(),
        OUTPUT_DIR / "best_model.pt",
    )

    results = {
        "stage1_selected_epoch":
            stage1_epoch,
        "stage2_selected_epoch":
            stage2_epoch,
        "n_train":
            int(len(train_df)),
        "n_val":
            int(len(val_df)),
        "metrics":
            metrics,
    }

    with open(
        OUTPUT_DIR / "metrics.json",
        "w",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        "\n========== OFFICIAL VAL =========="
    )

    print(metrics)

    print(
        "\nSelected Stage 1 epoch:",
        stage1_epoch,
    )

    print(
        "Selected Stage 2 epoch:",
        stage2_epoch,
    )


if __name__ == "__main__":
    main()
