from __future__ import annotations

from copy import deepcopy
from pathlib import Path
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
from torch.utils.data import Dataset, DataLoader

from torchvision.models.video import mvit_v2_s

from echoprime_transfer.preprocessing import (
    preprocess_avi_for_echoprime,
)


SEED = 42

MANIFEST = Path(
    "data/EchoNet-Dynamic/mini_manifest.csv"
)

VIDEO_DIR = Path(
    "data/EchoNet-Dynamic/Videos"
)

CHECKPOINT = Path(
    "external/EchoPrime/model_data/weights/"
    "echo_prime_encoder.pt"
)

OUTPUT_DIR = Path(
    "outputs/mini_v1/partial_finetune"
)

BATCH_SIZE = 1
ACCUMULATION_STEPS = 4

MAX_EPOCHS = 20
PATIENCE = 4

ENCODER_LR = 1e-5
HEAD_LR = 1e-3

WEIGHT_DECAY = 1e-4


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


class EchoDataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        video_path = (
            VIDEO_DIR
            / f"{row['FileName']}.avi"
        )

        video = preprocess_avi_for_echoprime(
            video_path
        )

        ef = torch.tensor(
            float(row["EF"]),
            dtype=torch.float32,
        )

        return video, ef


class EchoPrimeEF(nn.Module):

    def __init__(self):

        super().__init__()

        encoder = mvit_v2_s()

        encoder.head[1] = nn.Linear(
            encoder.head[1].in_features,
            512,
        )

        checkpoint = torch.load(
            CHECKPOINT,
            map_location="cpu",
        )

        encoder.load_state_dict(
            checkpoint
        )

        # Freeze full foundation model
        for parameter in encoder.parameters():
            parameter.requires_grad = False

        # Adapt only last transformer block
        for parameter in encoder.blocks[-1].parameters():
            parameter.requires_grad = True

        # Also adapt final normalization
        for parameter in encoder.norm.parameters():
            parameter.requires_grad = True

        # EchoPrime 512-D projection
        for parameter in encoder.head.parameters():
            parameter.requires_grad = True

        self.encoder = encoder

        self.regressor = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, 1),
        )

    def forward(self, x):

        embedding = self.encoder(x)

        ef = self.regressor(
            embedding
        ).squeeze(-1)

        return ef


def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    true = []
    pred = []

    with torch.inference_mode():

        for videos, targets in loader:

            videos = videos.to(device)
            targets = targets.to(device)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=device.type == "cuda",
            ):
                outputs = model(videos)

            true.extend(
                targets.cpu().numpy()
            )

            pred.extend(
                outputs.cpu().numpy()
            )

    true = np.asarray(true)
    pred = np.asarray(pred)

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

    # -------------------------------------------
    # Internal TRAIN split for model selection
    # -------------------------------------------

    fit_df, monitor_df = train_test_split(
        train_df,
        test_size=0.20,
        random_state=SEED,
    )

    print("Fit:", len(fit_df))
    print("Monitor:", len(monitor_df))
    print("Official VAL:", len(val_df))

    train_loader = DataLoader(
        EchoDataset(fit_df),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    monitor_loader = DataLoader(
        EchoDataset(monitor_df),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        EchoDataset(val_df),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = EchoPrimeEF().to(device)

    encoder_params = []

    encoder_params += list(
        model.encoder.blocks[-1].parameters()
    )

    encoder_params += list(
        model.encoder.norm.parameters()
    )

    encoder_params += list(
        model.encoder.head.parameters()
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

    patience = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        running_loss = 0.0

        for step, (
            videos,
            targets,
        ) in enumerate(
            train_loader,
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

                outputs = model(videos)

                loss = criterion(
                    outputs,
                    targets,
                )

                loss = (
                    loss
                    / ACCUMULATION_STEPS
                )

            scaler.scale(
                loss
            ).backward()

            if (
                step
                % ACCUMULATION_STEPS
                == 0
                or step
                == len(train_loader)
            ):

                scaler.step(
                    optimizer
                )

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

            running_loss += (
                loss.item()
                * ACCUMULATION_STEPS
            )

        monitor_metrics = evaluate(
            model,
            monitor_loader,
            device,
        )

        print(
            f"Epoch {epoch:02d} | "
            f"loss "
            f"{running_loss / len(train_loader):.4f} | "
            f"monitor MAE "
            f"{monitor_metrics['mae']:.4f} | "
            f"R2 "
            f"{monitor_metrics['r2']:.4f}"
        )

        if (
            monitor_metrics["mae"]
            < best_mae
        ):

            best_mae = (
                monitor_metrics["mae"]
            )

            best_state = deepcopy(
                model.state_dict()
            )

            patience = 0

        else:

            patience += 1

            if patience >= PATIENCE:
                print(
                    "Early stopping."
                )
                break

    # -------------------------------------------
    # Restore best internal-TRAIN checkpoint
    # -------------------------------------------

    model.load_state_dict(
        best_state
    )

    # Official validation evaluated once
    val_metrics = evaluate(
        model,
        val_loader,
        device,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        OUTPUT_DIR / "best_model.pt",
    )

    print("\nOfficial VAL:")
    print(val_metrics)

    print(
        "\nSaved:",
        OUTPUT_DIR / "best_model.pt",
    )


if __name__ == "__main__":
    main()
