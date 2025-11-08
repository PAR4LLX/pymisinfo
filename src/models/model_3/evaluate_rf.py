#!/usr/bin/env python
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_DIR = "./src/models/model_3/features"
ARTIFACT_DIR = "./src/models/model_3/artifacts"
CLEAN_DATASET = "./src/data/processed/clean_dataset_2.csv"

MODEL_PATH = os.path.join(ARTIFACT_DIR, "rf_linguistic_model.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}")

    print("[1] Loading trained model and test data...")
    model = joblib.load(MODEL_PATH)
    data = np.load(os.path.join(FEATURE_DIR, "test_data.npz"))
    X_test, y_test, test_ids = data["X"], data["y"], data["id"]

    # Load cleaned dataset for label + text alignment
    df = pd.read_csv(CLEAN_DATASET)
    if "id" not in df.columns:
        raise KeyError("Expected 'id' column in cleaned dataset.")
    df_test = df[df["id"].isin(test_ids)].copy()

    if len(df_test) != len(y_test):
        print(f"[WARN] ID alignment mismatch: dataset({len(df_test)}) vs npz({len(y_test)})")
        df_test = df_test.sort_values("id").head(len(y_test)).reset_index(drop=True)

    # ── Predict ─────────────────────────────────────────────────────────────
    print("[2] Running predictions on test set...")
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # ── Metrics ─────────────────────────────────────────────────────────────
    print("\n=== RandomForest Evaluation ===")
    report = classification_report(y_test, y_pred, digits=3)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(report)
    print("Confusion Matrix:\n", cm)
    print(f"AUC: {auc:.4f}\n")

    # ── Merge results with text for inspection ──────────────────────────────
    df_test = df_test.reset_index(drop=True)
    df_test["pred_label"] = y_pred
    df_test["pred_prob"] = y_prob
    df_test["correct"] = df_test["label"] == df_test["pred_label"]

    # ── Save predictions ────────────────────────────────────────────────────
    out_csv = os.path.join(ARTIFACT_DIR, "predictions_detailed.csv")
    df_test.to_csv(out_csv, index=False)
    print(f"Detailed predictions saved → {out_csv}")

    # ── Save metrics JSON ───────────────────────────────────────────────────
    out_json = os.path.join(ARTIFACT_DIR, "evaluation_metrics.json")
    with open(out_json, "w") as f:
        json.dump({
            "auc": float(auc),
            "confusion_matrix": cm.tolist(),
            "classification_report": report
        }, f, indent=2)
    print(f"Metrics saved → {out_json}\n")

if __name__ == "__main__":
    main()
