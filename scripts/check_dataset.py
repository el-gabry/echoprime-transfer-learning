from __future__ import annotations

import argparse
from pathlib import Path
from echoprime_transfer.data import load_filelist, split_summary, video_path

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    frame = load_filelist(root)
    print(split_summary(frame).to_string(index=False))
    if args.check_files:
        missing = [str(video_path(root, name)) for name in frame["FileName"] if not video_path(root, name).exists()]
        print(f"\nMissing videos: {len(missing)}")
        for item in missing[:20]: print(item)
        if missing: raise SystemExit(1)

if __name__ == "__main__":
    main()
