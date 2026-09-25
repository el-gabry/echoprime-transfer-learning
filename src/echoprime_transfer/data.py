from __future__ import annotations

from pathlib import Path
import pandas as pd

REQUIRED_FILELIST_COLUMNS = {"FileName", "EF", "Split"}

def load_filelist(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    path = root / "FileList.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    frame = pd.read_csv(path)
    missing = REQUIRED_FILELIST_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"FileList.csv is missing expected columns: {sorted(missing)}")
    frame = frame.copy()
    frame["Split"] = frame["Split"].astype(str).str.upper()
    frame["EF"] = pd.to_numeric(frame["EF"], errors="raise")
    if frame["FileName"].duplicated().any():
        dupes = frame.loc[frame["FileName"].duplicated(), "FileName"].tolist()
        raise ValueError(f"Duplicate FileName values detected: {dupes[:5]}")
    return frame

def video_path(root: str | Path, file_name: str) -> Path:
    root = Path(root)
    name = str(file_name)
    if not name.lower().endswith(".avi"):
        name += ".avi"
    return root / "Videos" / name

def split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame.groupby("Split", dropna=False)["EF"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index())
