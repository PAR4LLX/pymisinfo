#!/usr/bin/env python
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer


def suggest_maxlen(
    in_path: str = "./src/data/processed/clean_dataset.csv",
    percentile: float = 95.0,
) -> dict:

    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    try:
        df = pd.read_csv(
            in_path, usecols=["text", "label"], encoding="utf-8", encoding_errors="replace"
        )
    except TypeError:
        df = pd.read_csv(in_path, usecols=["text", "label"], encoding="utf-8")

    df = df.dropna(subset=["text"]).reset_index(drop=True)
    texts = df["text"].astype(str).tolist()

    tok = Tokenizer(oov_token="<OOV>")
    tok.fit_on_texts(texts)
    seqs = tok.texts_to_sequences(texts)
    lengths = np.array([len(s) for s in seqs], dtype=np.int32)

    if lengths.size == 0:
        raise ValueError("No non-empty texts found; cannot compute lengths.")

    stats = {
        "count": int(lengths.size),
        "min": int(lengths.min()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "p90": int(np.percentile(lengths, 90)),
        "p95": int(np.percentile(lengths, 95)),
        "p99": int(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
    }

    clamp_min = stats["min"]
    clamp_max = stats["max"]

    raw_rec = int(np.percentile(lengths, percentile))
    recommended = int(np.clip(raw_rec, clamp_min, clamp_max))

    return {
        "recommended_maxlen": recommended,
        "percentile_used": percentile,
        "raw_percentile_value": raw_rec,
        "clamp_range": [clamp_min, clamp_max],
        "stats": stats,
    }


def _print_report(r: dict):
    """Print formatted maxlen statistics."""
    s = r["stats"]
    print("Token length stats (from processed text):")
    print(f"  count={s['count']}  min={s['min']}  mean={s['mean']:.2f}  median={s['median']:.0f}")
    print(f"  p90={s['p90']}  p95={s['p95']}  p99={s['p99']}  max={s['max']}")
    print(f"\nRecommended maxlen @ p{int(r['percentile_used'])}: {r['raw_percentile_value']}")
    print(f"Dynamic clamp range ({s['min']}–{s['max']}): **{r['recommended_maxlen']}**")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Suggest optimal sequence max length from processed dataset CSV."
    )
    ap.add_argument(
        "--in",
        dest="in_path",
        default="./src/data/processed/clean_dataset.csv",
        help="Path to the processed CSV (must contain a 'text' column).",
    )
    ap.add_argument("--percentile", type=float, default=95.0)
    args = ap.parse_args()

    report = suggest_maxlen(
        in_path=args.in_path,
        percentile=args.percentile,
    )
    _print_report(report)

