#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ─────────────────────────────────────────────────────────────────────────────
# Import custom Attention layer from training code
# ─────────────────────────────────────────────────────────────────────────────
from src.models.model_2.train_bilstm import Attention

FEATURE_DIR = "./src/models/model_2/features"
ARTIFACT_DIR = "./src/models/model_2/artifacts"
CLEAN_DATASET = "./src/data/processed/clean_dataset.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Safe Tokenizer Loader (fixes keras.src.preprocessing error)
# ─────────────────────────────────────────────────────────────────────────────
def safe_load_tokenizer(path: str):
    """Load Keras Tokenizer from pickle, patching module path if needed."""
    try:
        return joblib.load(path)
    except ModuleNotFoundError as e:
        if "keras.src.preprocessing" in str(e):
            import sys, keras.preprocessing.text as keras_pre_old
            # Patch missing module path so joblib can find Tokenizer class
            sys.modules["keras.src.preprocessing.text"] = keras_pre_old
            return joblib.load(path)
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # --- Load model and tokenizer ---
    model_path = os.path.join(ARTIFACT_DIR, "bilstm_best.keras")
    tokenizer_path = os.path.join(FEATURE_DIR, "tokenizer.pkl")

    print(f"Loading model from: {model_path}")
    model = load_model(model_path, custom_objects={"Attention": Attention})

    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = safe_load_tokenizer(tokenizer_path)

    # --- Load features & labels ---
    test_npz = os.path.join(FEATURE_DIR, "test_data.npz")
    test = np.load(test_npz)
    X_test, y_test = test["X"], test["y"]

    # --- Load dataset for ID alignment ---
    df = pd.read_csv(CLEAN_DATASET)
    id_path = os.path.join(FEATURE_DIR, "test_ids.npy")

    if os.path.exists(id_path):
        test_ids = np.load(id_path)
        df_test = df[df["id"].isin(test_ids)].copy()
    else:
        print("[WARN] test_ids.npy not found — results will not be ID-aligned.")
        df_test = df.sample(n=len(y_test), random_state=42).copy()

    # --- Run predictions ---
    print("\nRunning predictions...")
    y_prob = model.predict(X_test, batch_size=64, verbose=1).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    # --- Compute metrics ---
    report = classification_report(y_test, y_pred, output_dict=False)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n=== BiLSTM Evaluation Report ===")
    print(report)
    print("Confusion Matrix:\n", cm)
    print(f"AUC: {auc:.4f}")

    # --- Merge results for inspection ---
    df_test = df_test.reset_index(drop=True)
    df_test["pred_label"] = y_pred
    df_test["pred_prob"] = y_prob
    df_test["correct"] = df_test["label"] == df_test["pred_label"]

    # --- Save output CSV ---
    out_csv = os.path.join(ARTIFACT_DIR, "predictions_detailed.csv")
    df_test.to_csv(out_csv, index=False)
    print(f"\nSaved detailed predictions → {out_csv}")


if __name__ == "__main__":
    main()

