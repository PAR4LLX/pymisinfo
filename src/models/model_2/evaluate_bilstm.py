#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.models.model_2.train_bilstm import Attention

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_DIR = "./src/models/model_2/features"
ARTIFACT_DIR = "./src/models/model_2/artifacts"
RAW_DATASET = "./src/data/raw/misinfo_v2.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Safe tokenizer load (for keras version mismatches)
# ─────────────────────────────────────────────────────────────────────────────
def safe_load_tokenizer(path: str):
    try:
        return joblib.load(path)
    except ModuleNotFoundError as e:
        if "keras.src.preprocessing" in str(e):
            import sys, keras.preprocessing.text as keras_pre_old
            sys.modules["keras.src.preprocessing.text"] = keras_pre_old
            return joblib.load(path)
        raise

# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation (no calibration)
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Load model + tokenizer
    model_path = os.path.join(ARTIFACT_DIR, "bilstm_best.keras")
    tok_path = os.path.join(FEATURE_DIR, "tokenizer.pkl")

    print(f"Loading model → {model_path}")
    model = load_model(model_path, custom_objects={"Attention": Attention})

    print(f"Loading tokenizer → {tok_path}")
    tokenizer = safe_load_tokenizer(tok_path)

    # Load test split
    test_npz = os.path.join(FEATURE_DIR, "test_data.npz")
    test = np.load(test_npz)
    X_test, y_test, test_ids = test["X"], test["y"], test["id"]

    # Load the raw dataset and align IDs
    raw_df = pd.read_csv(RAW_DATASET)
    df_test = raw_df[raw_df["id"].isin(test_ids)].copy().reset_index(drop=True)

    # Predict
    print("\nRunning predictions...")
    y_prob = model.predict(X_test, batch_size=64, verbose=1).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    # Compute metrics
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)
    auc = roc_auc_score(y_test, y_prob)

    print("\n=== BiLSTM Evaluation Report ===")
    print(report)
    print("Confusion Matrix:\n", cm)
    print(f"AUC: {auc:.4f}")

    # Build output CSV (raw text + probabilities)
    df_test["pred_label"] = y_pred
    df_test["pred_prob"] = y_prob
    df_test["correct"] = df_test["label"] == df_test["pred_label"]

    out_csv = os.path.join(ARTIFACT_DIR, "predictions_detailed.csv")
    df_test.to_csv(out_csv, index=False)
    print(f"\nSaved predictions → {out_csv}")
    print(f"Included columns: {list(df_test.columns)}")

if __name__ == "__main__":
    main()
