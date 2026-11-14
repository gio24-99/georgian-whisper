
import csv
import os

import librosa
import pandas as pd

SAMPLE_RATE = 16000


def _read_tsv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)


def load_split(cv_dir: str, split: str, use_validated: bool = False) -> list[dict]:

    clips_dir = os.path.join(cv_dir, "clips")

    if split == "train" and use_validated:
        df = _read_tsv(os.path.join(cv_dir, "validated.tsv"))
        held_out = pd.concat([_read_tsv(os.path.join(cv_dir, f)) for f in ("dev.tsv", "test.tsv")])
        before = len(df)
        df = df[~df["client_id"].isin(set(held_out["client_id"]))]
        df = df[~df["sentence"].isin(set(held_out["sentence"]))]
        print(f"[data] validated.tsv: {before} rows -> {len(df)} after removing dev/test speakers & sentences")
    else:
        df = _read_tsv(os.path.join(cv_dir, f"{split}.tsv"))

    df = df[df["sentence"].str.strip() != ""]
    items = [{"path": os.path.join(clips_dir, p), "text": s.strip()} for p, s in zip(df["path"], df["sentence"])]
    print(f"[data] {split}: {len(items)} clips")
    return items


def load_audio(path: str):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return audio
