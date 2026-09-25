from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


ECHOPRIME_MEAN = torch.tensor(
    [29.110628, 28.076836, 29.096405],
    dtype=torch.float32,
).view(3, 1, 1, 1)

ECHOPRIME_STD = torch.tensor(
    [47.989223, 46.456997, 47.20083],
    dtype=torch.float32,
).view(3, 1, 1, 1)


def crop_and_scale(
    frame: np.ndarray,
    size: tuple[int, int] = (224, 224),
    zoom: float = 0.1,
) -> np.ndarray:
    height, width = frame.shape[:2]

    input_ratio = width / height
    output_ratio = size[0] / size[1]

    if input_ratio > output_ratio:
        padding = int(round((width - output_ratio * height) / 2))
        if padding > 0:
            frame = frame[:, padding:-padding]

    elif input_ratio < output_ratio:
        padding = int(round((height - width / output_ratio) / 2))
        if padding > 0:
            frame = frame[padding:-padding]

    if zoom:
        h, w = frame.shape[:2]
        pad_x = int(round(w * zoom))
        pad_y = int(round(h * zoom))

        if pad_x > 0 and pad_y > 0:
            frame = frame[
                pad_y : h - pad_y,
                pad_x : w - pad_x,
            ]

    return cv2.resize(
        frame,
        size,
        interpolation=cv2.INTER_CUBIC,
    )


def read_avi_frames(path: str | Path) -> np.ndarray:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frames = []

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from: {path}")

    return np.stack(frames)


def preprocess_avi_for_echoprime(
    path: str | Path,
    frames_to_take: int = 32,
    frame_stride: int = 2,
    video_size: int = 224,
) -> torch.Tensor:
    frames = read_avi_frames(path)

    processed = np.empty(
        (len(frames), video_size, video_size, 3),
        dtype=np.float32,
    )

    for i, frame in enumerate(frames):
        processed[i] = crop_and_scale(
            frame,
            size=(video_size, video_size),
            zoom=0.1,
        )

    x = torch.from_numpy(processed).permute(3, 0, 1, 2)

    x = (x - ECHOPRIME_MEAN) / ECHOPRIME_STD

    if x.shape[1] < frames_to_take:
        padding = torch.zeros(
            (
                3,
                frames_to_take - x.shape[1],
                video_size,
                video_size,
            ),
            dtype=x.dtype,
        )

        x = torch.cat([x, padding], dim=1)

    x = x[
        :,
        0:frames_to_take:frame_stride,
        :,
        :,
    ]

    return x