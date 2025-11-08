#!/usr/bin/env python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import joblib
import numpy as np
import pandas as pd
import logging
import warnings

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- import shared split utility ---
from src.utils.data_split import stratified_split

tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

MAXLEN = 567


# ---------- Helper Functions ----------
def _class_counts_only(y: np.ndarray) -> dict:
    """Return a count dictionary of labels."""
    u, c = np.unique(y, return_counts=True)
    return {int(k): int(v) for k, v in zip(u, c)}


# ---------- Main Feature Preparation ----------
def prepare_features(
    in_path: str = "./src/data/processed/clean_dataset.csv",
    out_dir: str = "./src/models/model_2/features",
    maxlen: int = MAXLEN,
    random_state: int = 42,
    preview: bool = False,
):
    """Tokenize, encode, and split an already-cleaned dataset into train/val/test NPZs."""

    # Validate input
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Cleaned CSV not found: {in_path}")

    # Load cleaned dataset
    df = pd.read_csv(in_path)
    required_cols = {"id", "text", "date", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {required_cols}")

    # Validate label range and drop NAs
    df = df.dropna(subset=["text", "label", "date"])
    df = df[df["label"].isin([0, 1])].reset_index(drop=True)

    # --- Parse and encode dates safely (match cleaner format "%y-%m-%d") ---
    df["date"] = pd.to_datetime(df["date"], format="%y-%m-%d", errors="coerce")
    min_date = df["date"].min()
    df["date_days"] = (df["date"] - min_date).dt.days.astype(np.int32)

    # --- Prepare arrays ---
    ids = df["id"].astype(np.int32).to_numpy()
    y = df["label"].astype(np.int32).to_numpy()
    dates = df["date_days"].to_numpy()
    texts = df["text"].astype(str).tolist()

    # --- Use universal stratified split utility ---
    df_split = pd.DataFrame({
        "id": ids,
        "text": texts,
        "label": y,
        "date_days": dates,
    })
    train_df, val_df, test_df = stratified_split(df_split, label_col="label", random_state=random_state)

    # Extract split arrays
    X_train_txt = train_df["text"].tolist()
    X_val_txt   = val_df["text"].tolist()
    X_test_txt  = test_df["text"].tolist()

    y_train = train_df["label"].to_numpy()
    y_val   = val_df["label"].to_numpy()
    y_test  = test_df["label"].to_numpy()

    id_train = train_df["id"].to_numpy()
    id_val   = val_df["id"].to_numpy()
    id_test  = test_df["id"].to_numpy()

    date_train = train_df["date_days"].to_numpy()
    date_val   = val_df["date_days"].to_numpy()
    date_test  = test_df["date_days"].to_numpy()

    # --- Tokenization (fit on TRAIN only) ---
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_txt)

    # --- Convert to sequences and pad ---
    X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train_txt), maxlen=maxlen, padding="post", truncating="post")
    X_val_seq   = pad_sequences(tokenizer.texts_to_sequences(X_val_txt),   maxlen=maxlen, padding="post", truncating="post")
    X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(X_test_txt),  maxlen=maxlen, padding="post", truncating="post")

    # --- Save artifacts ---
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "train_data.npz"), X=X_train_seq, y=y_train, id=id_train, date=date_train)
    np.savez_compressed(os.path.join(out_dir, "val_data.npz"),   X=X_val_seq,   y=y_val,   id=id_val,   date=date_val)
    np.savez_compressed(os.path.join(out_dir, "test_data.npz"),  X=X_test_seq,  y=y_test,  id=id_test,  date=date_test)
    joblib.dump(tokenizer, os.path.join(out_dir, "tokenizer.pkl"))

    summary = {
        "csv": os.path.abspath(in_path),
        "maxlen": int(maxlen),
        "vocab_size": int(len(tokenizer.word_index) + 1),
        "date_range": [str(min_date.date()), str(df["date"].max().date())],
        "rows_used": int(len(df)),
        "label_meaning": {"0": "Fake", "1": "Real"},
        "splits": {
            "train": {"n": len(y_train), "classes": _class_counts_only(y_train)},
            "val":   {"n": len(y_val),   "classes": _class_counts_only(y_val)},
            "test":  {"n": len(y_test),  "classes": _class_counts_only(y_test)},
        },
    }

    with open(os.path.join(out_dir, "preprocess_report.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if preview:
        print("\n=== Feature Preparation Summary ===")
        print(f"Source file: {os.path.abspath(in_path)}")
        print(f"Saved to:    {os.path.abspath(out_dir)}")
        print(f"Total rows:  {len(df)}")
        print(f"Vocabulary:  {summary['vocab_size']}")
        print(f"Split sizes: train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
        print(f"Date range:  {summary['date_range'][0]} → {summary['date_range'][1]}\n")

    return summary


# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Tokenize, encode, and split the cleaned dataset.")
    p.add_argument("--in", dest="in_path", default="./src/data/processed/clean_dataset.csv", help="Path to cleaned CSV file")
    p.add_argument("--out", dest="out_dir", default="./src/models/model_2/features", help="Output directory for tokenized data")
    p.add_argument("--maxlen", type=int, default=MAXLEN)
    p.add_argument("--preview", action="store_true", help="Print output summary to console")
    args = p.parse_args()

    prepare_features(args.in_path, args.out_dir, args.maxlen, preview=args.preview)

