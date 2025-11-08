#!/usr/bin/env python
import os
import json
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_DIR = "./src/models/model_3/features"
ARTIFACT_DIR = "./src/models/model_3/artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Utility to load feature splits
# ─────────────────────────────────────────────────────────────────────────────
def load_split(name: str):
    path = os.path.join(FEATURE_DIR, f"{name}_data.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    data = np.load(path)
    return data["X"], data["y"], data["id"]

# ─────────────────────────────────────────────────────────────────────────────
# Main training logic
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("[1] Loading feature splits...")
    X_train, y_train, id_train = load_split("train")
    X_val, y_val, id_val = load_split("val")
    X_test, y_test, id_test = load_split("test")

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Combine train and val for full retraining later
    X_full = np.concatenate([X_train, X_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)

    # ── Model definition ────────────────────────────────────────────────
    print("[2] Training RandomForest classifier...")
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    pipe.fit(X_train, y_train)

    # ── Evaluation on validation ───────────────────────────────────────
    print("\n[3] Validation results:")
    y_pred = pipe.predict(X_val)
    probs = pipe.predict_proba(X_val)

    # Defensive: handle 1-class validation folds
    if probs.shape[1] == 2:
        y_prob = probs[:, 1]
    else:
        # Only one class seen; create dummy probabilities
        y_prob = np.zeros(len(probs))
        if pipe.classes_[0] == 0:
            y_prob[:] = 0.0
        else:
            y_prob[:] = 1.0

    report = classification_report(y_val, y_pred, digits=3)
    cm = confusion_matrix(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob) if len(np.unique(y_val)) > 1 else 0.0

    print(report)
    print("Confusion Matrix:\n", cm)
    print(f"AUC: {auc:.4f}")

    # ── Final test performance (after retraining on full data) ─────────
    print("\n[4] Retraining on full train+val set...")
    pipe.fit(X_full, y_full)
    y_test_pred = pipe.predict(X_test)
    probs_test = pipe.predict_proba(X_test)

    if probs_test.shape[1] == 2:
        y_test_prob = probs_test[:, 1]
    else:
        y_test_prob = np.zeros(len(probs_test))
        if pipe.classes_[0] == 0:
            y_test_prob[:] = 0.0
        else:
            y_test_prob[:] = 1.0

    test_report = classification_report(y_test, y_test_pred, digits=3)
    test_cm = confusion_matrix(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_prob) if len(np.unique(y_test)) > 1 else 0.0

    print("\n[5] Final Test Set Evaluation:")
    print(test_report)
    print("Confusion Matrix:\n", test_cm)
    print(f"AUC: {test_auc:.4f}")

    # ── Save model and reports ─────────────────────────────────────────
    model_path = os.path.join(ARTIFACT_DIR, "rf_linguistic_model.pkl")
    joblib.dump(pipe, model_path)

    with open(os.path.join(ARTIFACT_DIR, "rf_linguistic_report.json"), "w") as f:
        json.dump({
            "val_auc": float(auc),
            "test_auc": float(test_auc),
            "val_report": report,
            "test_report": test_report
        }, f, indent=2)

    print(f"\n[6] Training complete. Model saved → {model_path}")
    print(f"Metrics saved → {os.path.join(ARTIFACT_DIR, 'rf_linguistic_report.json')}\n")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

