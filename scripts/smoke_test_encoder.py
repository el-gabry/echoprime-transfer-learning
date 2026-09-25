"""Smoke-test the official EchoPrime video encoder checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
import torchvision

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists(): raise FileNotFoundError(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.video.mvit_v2_s()
    model.head[-1] = torch.nn.Linear(model.head[-1].in_features, 512)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    x = torch.zeros((1, 3, 16, 224, 224), device=device)
    with torch.inference_mode(): embedding = model(x)
    print(f"device={device}")
    print(f"embedding_shape={tuple(embedding.shape)}")
    if tuple(embedding.shape) != (1, 512): raise SystemExit("Unexpected embedding shape")

if __name__ == "__main__":
    main()
