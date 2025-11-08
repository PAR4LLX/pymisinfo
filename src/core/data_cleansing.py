#!/usr/bin/env python
import os
import re
import json
import nltk
import pandas as pd
from ftfy import fix_text
from datetime import datetime


# -------------------------------------------------------------------
# Shared utilities
# -------------------------------------------------------------------
def _load_stopwords():
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    sw = set(stopwords.words("english"))
    for w in {"not", "no", "nor", "never", "without"}:
        sw.discard(w)
    return sw


def _clean_text(s: str, stop_words: set[str]) -> str:
    re_html = re.compile(r"<.*?>")
    re_url = re.compile(r"http\S+|www\S+")
    re_handle = re.compile(r"@\w+")
    re_nonword = re.compile(r"[^\w\s'-]")
    re_num_spacing = re.compile(r"(?<=\d)(?=[a-zA-Z])|(?<=[a-zA-Z])(?=\d)")

    contractions = {
        "n't": " not", "'re": " are", "'s": " is",
        "'ll": " will", "'ve": " have", "'d": " would"
    }

    s = str(s)
    s = fix_text(s)
    s = re_html.sub(" ", s)
    s = re_url.sub(" ", s)
    s = re_handle.sub(" ", s)
    s = re_num_spacing.sub(" ", s)
    for k, v in contractions.items():
        s = s.replace(k, v)
    s = re_nonword.sub(" ", s)
    s = s.lower()
    s = re.sub(r"\s*-\s*", "-", s)
    s = " ".join(w for w in s.split() if w not in stop_words)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -------------------------------------------------------------------
# Dataset cleaner (CSV)
# -------------------------------------------------------------------
def clean_dataset(in_path: str, out_dir: str, preview: bool = False):
    stop_words = _load_stopwords()
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    df_raw = pd.read_csv(
        in_path,
        usecols=["id", "text", "date", "label"],
        encoding="utf-8",
        encoding_errors="replace",
    )

    df = df_raw.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["label"].isin([0, 1])]

    df["text"] = df["text"].map(lambda s: _clean_text(s, stop_words))
    mask_nonempty = df["text"].str.strip().astype(bool)
    empty_removed = int((~mask_nonempty).sum())
    df = df[mask_nonempty]

    before_count = len(df_raw)
    after_count = len(df)
    total_removed = before_count - after_count

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "clean_dataset.csv")
    df.to_csv(out_path, index=False, encoding="utf-8", date_format="%y-%m-%d")

    if preview:
        print("\n=== Data Cleansing Summary ===")
        print(f"Source:  {os.path.abspath(in_path)}")
        print(f"Output:  {out_path}")
        print(f"Initial rows: {before_count}")
        print(f"Rows retained: {after_count} (removed {total_removed} total)")
        print(f"  ├─ Invalid date/label: {total_removed - empty_removed}")
        print(f"  └─ Empty text removed: {empty_removed}")
        print(f"Columns: {list(df.columns)}")
        print("Cleaned dataset saved successfully.\n")

    return out_path


# -------------------------------------------------------------------
# Single JSON cleaner
# -------------------------------------------------------------------
def clean_json_input(data, out_path: str | None = None, save_back: bool = False):
    if isinstance(data, str) and os.path.isfile(data):
        with open(data, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif isinstance(data, str):
        data = json.loads(data)

    if not isinstance(data, dict):
        raise TypeError("Expected a dict, JSON string, or path to JSON file.")

    for key in ("id", "text", "date"):
        if key not in data:
            raise ValueError(f"Missing required field: '{key}'")

    stop_words = _load_stopwords()
    cleaned_text = _clean_text(data["text"], stop_words)

    try:
        parsed_date = datetime.strptime(data["date"], "%Y-%m-%d").date()
        date_str = parsed_date.isoformat()
    except Exception:
        date_str = None

    cleaned = {"id": data["id"], "text": cleaned_text, "date": date_str}

    if save_back:
        if not out_path:
            out_path = "cleaned_input.json"
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)

    return cleaned


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Clean misinformation data (CSV dataset or single JSON input)."
    )
    p.add_argument("--mode", choices=["dataset", "json"], default="dataset",
                   help="Choose whether to clean a dataset CSV or a single JSON input. Default: dataset")
    p.add_argument("--in", dest="in_path",
                   default="./src/data/raw/misinfo_v2.csv",
                   help="Input CSV file or JSON string/file path. Default: ./src/data/raw/misinfo_dataset.csv")
    p.add_argument("--out", dest="out_path",
                   default="./src/data/processed",
                   help="Output file or directory path. Default: ./src/data/processed")
    p.add_argument("--preview", action="store_true",
                   default=True,
                   help="Show preview summary (dataset mode only). Default: True")
    p.add_argument("--save", action="store_true",
                   help="Save cleaned output to file (JSON mode only).")

    args = p.parse_args()

    if args.mode == "dataset":
        clean_dataset(args.in_path, args.out_path, preview=args.preview)
    else:
        cleaned = clean_json_input(args.in_path, out_path=args.out_path, save_back=args.save)
        print(json.dumps(cleaned, indent=2, ensure_ascii=False))

