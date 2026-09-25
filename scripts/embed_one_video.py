from __future__ import annotations

import argparse

import torch
import torchvision

from echoprime_transfer.preprocessing import preprocess_avi_for_echoprime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")

    # Official EchoPrime video encoder architecture
    model = torchvision.models.video.mvit_v2_s()

    model.head[-1] = torch.nn.Linear(
        model.head[-1].in_features,
        512,
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    # AVI -> EchoPrime tensor [3, 16, 224, 224]
    video = preprocess_avi_for_echoprime(args.video)

    print("preprocessed shape:", tuple(video.shape))
    print("preprocessed finite:", torch.isfinite(video).all().item())

    # Add batch dimension -> [1, 3, 16, 224, 224]
    video = video.unsqueeze(0).to(device)

    with torch.inference_mode():
        embedding = model(video)

    print("embedding shape:", tuple(embedding.shape))
    print("embedding finite:", torch.isfinite(embedding).all().item())
    print("embedding norm:", embedding.norm(dim=1).item())


if __name__ == "__main__":
    main()
