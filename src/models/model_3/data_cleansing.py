#!/usr/bin/env python
import os
import re
import pandas as pd
from ftfy import fix_text

def clean_text_dataset_rf(
    in_path: str = "./src/data/raw/missinfo_v2.csv",
    out_dir: str = "./src/data/processed",
    preview: bool = False,
):
    """
    Cleaner for stylistic / linguistic feature models.
    Keeps punctuation, capitalization, emojis, and symbols in the text itself,
    but the source code contains only standard ASCII characters.
    """

    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    # --- Load minimal columns
    df = pd.read_csv(
        in_path,
        usecols=["id", "text", "date", "label"],
        encoding="utf-8",
        encoding_errors="replace",
    )

    # --- Basic validation
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).map(fix_text)
    df = df[df["label"].isin([0, 1])]

    # --- Light cleanup: remove HTML and normalize whitespace
    RE_HTML = re.compile(r"<.*?>")
    df["text"] = df["text"].str.replace(RE_HTML, " ", regex=True)
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()

    # --- Parse and validate dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # --- Output
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "clean_dataset_2.csv")
    df.to_csv(out_path, index=False, encoding="utf-8", date_format="%y-%m-%d")

    if preview:
        print("\n=== RF Data Cleansing Summary ===")
        print(f"Source file: {os.path.abspath(in_path)}")
        print(f"Output file: {out_path}")
        print(f"Total rows retained: {len(df)}")
        print("Preserved stylistic content: punctuation, case, and symbols\n")

    return df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Clean raw dataset for RF stylistic model.")
    p.add_argument("--in", dest="in_path", default="./src/data/raw/missinfo_v2.csv")
    p.add_argument("--out", dest="out_dir", default="./src/data/processed")
    p.add_argument("--preview", action="store_true")
    args = p.parse_args()

    clean_text_dataset_rf(args.in_path, args.out_dir, preview=args.preview)

