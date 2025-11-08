#!/usr/bin/env python
"""
Pipeline Integrity Test for BiLSTM Misinformation Model
-------------------------------------------------------
Validates:
1. Dataset and feature alignment
2. Label consistency
3. Duplicate/leakage detection
4. Evaluation metric sanity

Run:
    python -m src.models.model_2.test_pipeline_integrity
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model

# Import custom Attention layer from training script
from src.models.model_2.train_bilstm import Attention


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
BASE_DIR = "./src/models/model_2"
FEATURE_DIR = os.path.join(BASE_DIR, "features")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
DATASET_PATH = "./src/data/processed/clean_dataset.csv"


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def load_npz_split(name):
    path = os.path.join(FEATURE_DIR, f"{name}_data.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    data = np.load(path)
    return data["X"], data["y"], data["id"]


def safe_pass(msg):
    return ("PASS", msg)


def safe_fail(msg):
    return ("FAIL", msg)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_alignment():
    results = []
    df = pd.read_csv(DATASET_PATH)

    for split in ["train", "val", "test"]:
        try:
            X, y, ids = load_npz_split(split)
            if len(X) != len(y) or len(y) != len(ids):
                results.append(safe_fail(f"{split}: feature, label, id lengths mismatch"))
                continue
            if not np.isin(ids, df["id"]).all():
                results.append(safe_fail(f"{split}: some ids not found in dataset"))
                continue
            if len(np.unique(ids)) != len(ids):
                results.append(safe_fail(f"{split}: duplicate IDs found"))
                continue
            results.append(safe_pass(f"{split}: alignment OK"))
        except Exception as e:
            results.append(safe_fail(f"{split}: {e}"))
    return results


def test_label_consistency():
    results = []
    df = pd.read_csv(DATASET_PATH)
    _, y_test, test_ids = load_npz_split("test")
    df_test = df[df["id"].isin(test_ids)].sort_values("id")
    y_sorted = y_test[np.argsort(test_ids)]
    mismatch = (df_test["label"].to_numpy() != y_sorted).sum()

    if mismatch == 0:
        results.append(safe_pass("Test labels align with dataset"))
    else:
        results.append(safe_fail(f"Label mismatches found: {mismatch}"))
    return results


def test_duplicate_leakage(max_allowed_sim=0.9):
    results = []
    df = pd.read_csv(DATASET_PATH)
    train_ids = np.load(os.path.join(FEATURE_DIR, "train_data.npz"))["id"]
    test_ids = np.load(os.path.join(FEATURE_DIR, "test_data.npz"))["id"]

    train_texts = df[df["id"].isin(train_ids)]["text"].tolist()
    test_texts = df[df["id"].isin(test_ids)]["text"].tolist()

    n_sample = min(300, len(train_texts), len(test_texts))
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf.fit(train_texts[:n_sample] + test_texts[:n_sample])
    X_tr = tfidf.transform(train_texts[:n_sample])
    X_te = tfidf.transform(test_texts[:n_sample])
    sim = cosine_similarity(X_tr, X_te).max()

    if sim < max_allowed_sim:
        results.append(safe_pass(f"No duplicate text leakage (max similarity={sim:.3f})"))
    else:
        results.append(safe_fail(f"Possible text leakage detected (max similarity={sim:.3f})"))
    return results


def test_evaluation_metrics(thresholds=(0.95, 0.95)):
    results = []
    model_path = os.path.join(ARTIFACT_DIR, "bilstm_best.keras")

    # FIX: Provide custom_objects for Attention layer
    model = load_model(model_path, custom_objects={"Attention": Attention})

    test = np.load(os.path.join(FEATURE_DIR, "test_data.npz"))
    X_test, y_test = test["X"], test["y"]

    y_prob = model.predict(X_test, batch_size=64, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    acc = report["accuracy"]
    f1 = report["weighted avg"]["f1-score"]

    if acc >= thresholds[0]:
        results.append(safe_pass(f"Accuracy >= {thresholds[0]} (got {acc:.3f})"))
    else:
        results.append(safe_fail(f"Accuracy below threshold (got {acc:.3f})"))

    if f1 >= thresholds[1]:
        results.append(safe_pass(f"Weighted F1 >= {thresholds[1]} (got {f1:.3f})"))
    else:
        results.append(safe_fail(f"Weighted F1 below threshold (got {f1:.3f})"))

    cm = confusion_matrix(y_test, y_pred)
    results.append(safe_pass(f"Confusion Matrix:\n{cm}"))
    return results


# ----------------------------------------------------------------------
# Main Runner
# ----------------------------------------------------------------------
def main():
    all_results = []
    print("Running BiLSTM Pipeline Integrity Tests\n")

    all_results += test_alignment()
    all_results += test_label_consistency()
    all_results += test_duplicate_leakage()
    all_results += test_evaluation_metrics()

    # Summary
    print("\nSummary of Results")
    print("-" * 60)
    failures = 0
    for status, msg in all_results:
        print(f"{status:<6} | {msg}")
        if status == "FAIL":
            failures += 1

    print("-" * 60)
    if failures == 0:
        print("All checks passed successfully.")
    else:
        print(f"{failures} checks failed. Review the log above.")
    print()


if __name__ == "__main__":
    main()

