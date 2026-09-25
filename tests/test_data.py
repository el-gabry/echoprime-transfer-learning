from pathlib import Path
import pandas as pd
from echoprime_transfer.data import load_filelist, split_summary, video_path

def test_load_filelist_and_summary(tmp_path: Path):
    pd.DataFrame({"FileName":["a","b","c"],"EF":[50.,60.,70.],"Split":["train","val","test"]}).to_csv(tmp_path/"FileList.csv", index=False)
    loaded = load_filelist(tmp_path)
    assert loaded["Split"].tolist() == ["TRAIN","VAL","TEST"]
    assert split_summary(loaded)["count"].sum() == 3
    assert video_path(tmp_path, "a").name == "a.avi"
