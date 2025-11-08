#!/usr/bin/env python
import os
import re
import joblib
import warnings
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from textblob import TextBlob
import textstat
import spacy
from empath import Empath
from src.utils.data_split import stratified_split

# ─────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning)
nlp = spacy.load("en_core_web_sm")
lexicon = Empath()

DATA_PATH = "./src/data/processed/clean_dataset_2.csv"
FEATURE_DIR = "./src/models/model_3/features"
os.makedirs(FEATURE_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
def extract_features(texts: list[str]) -> pd.DataFrame:
    """Compute stylistic and linguistic statistics for each text with live ETA."""
    rows = []
    start = time.time()
    total = len(texts)

    for i, t in enumerate(tqdm(texts, desc="Extracting features", ncols=90)):
        doc = nlp(t)
        blob = TextBlob(t)
        words = [w.text for w in doc if w.is_alpha]
        n_words = len(words)
        n_chars = len(t)
        n_sents = max(1, len(list(doc.sents)))

        # Stylometric & lexical
        exclam = t.count("!")
        caps_ratio = sum(1 for c in t if c.isupper()) / max(1, n_chars)
        punct_ratio = sum(1 for c in t if re.match(r"[^\w\s]", c)) / max(1, n_chars)
        unique_ratio = len(set(words)) / max(1, n_words)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0

        # Sentiment & readability
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        flesch = textstat.flesch_reading_ease(t)
        fog = textstat.gunning_fog(t)

        # POS distribution
        pos_counts = doc.count_by(spacy.attrs.POS)
        noun_ratio = pos_counts.get(92, 0) / max(1, n_words)
        adj_ratio = pos_counts.get(84, 0) / max(1, n_words)
        adv_ratio = pos_counts.get(85, 0) / max(1, n_words)
        verb_ratio = pos_counts.get(100, 0) / max(1, n_words)
        ner_count = len(doc.ents)

        # Bias intensity via Empath
        empath_scores = lexicon.analyze(
            t,
            categories=[
                "politics", "government", "violence", "deception",
                "trust", "law", "money", "emotion", "hate"
            ],
            normalize=True
        )
        bias_intensity = sum(empath_scores.values())

        rows.append({
            "n_chars": n_chars,
            "n_words": n_words,
            "n_sents": n_sents,
            "exclam": exclam,
            "caps_ratio": caps_ratio,
            "punct_ratio": punct_ratio,
            "unique_ratio": unique_ratio,
            "avg_word_len": avg_word_len,
            "polarity": polarity,
            "subjectivity": subjectivity,
            "flesch": flesch,
            "fog": fog,
            "noun_ratio": noun_ratio,
            "adj_ratio": adj_ratio,
            "adv_ratio": adv_ratio,
            "verb_ratio": verb_ratio,
            "ner_count": ner_count,
            "bias_intensity": bias_intensity,
        })

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            remaining = (total - (i + 1)) / max(rate, 1e-5)
            print(f"Processed {i+1}/{total} ({(i+1)/total:.1%}) | ETA {remaining/60:.1f} min")

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
def main(limit: int | None = None):
    print("[1] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"])
    df = df[df["label"].isin([0, 1])]

    if limit is not None:
        df = df.head(limit)
        print(f"[TEST MODE] Limiting to first {limit} rows for faster run.")

    print(f"[2] Extracting stylistic features from {len(df)} texts...")
    X_df = extract_features(df["text"].tolist())

    df_full = pd.concat([df[["id", "label"]], X_df], axis=1)

    # Clean and cast labels
    df_full = df_full.dropna(subset=["label"]).copy()
    df_full["label"] = df_full["label"].astype(int)

    print("[3] Performing stratified split...")
    train_df, val_df, test_df = stratified_split(df_full, label_col="label", random_state=42)

    def _save_split(name, dframe):
        X = dframe.drop(columns=["id", "label"]).to_numpy(dtype=np.float32)
        y = dframe["label"].to_numpy(dtype=np.int32)
        ids = dframe["id"].to_numpy()
        np.savez_compressed(os.path.join(FEATURE_DIR, f"{name}_data.npz"), X=X, y=y, id=ids)

    _save_split("train", train_df)
    _save_split("val", val_df)
    _save_split("test", test_df)
    joblib.dump(list(X_df.columns), os.path.join(FEATURE_DIR, "feature_names.pkl"))

    print(f"\n[4] Features saved → {os.path.abspath(FEATURE_DIR)}")
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Feature count: {X_df.shape[1]}\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare linguistic features for RandomForest model.")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N rows for testing.")
    parser.add_argument("--preview", action="store_true", help="Print dataset stats after processing.")
    args = parser.parse_args()

    main(limit=args.limit)

