import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from utils import load_features


def run_logreg():
    model_path = "./assets/logreg_model.pkl"
    os.makedirs("./assets/plots", exist_ok=True)

    # Load pre-split features from utils
    print("Loading train/val/test features…")
    X_train, y_train = load_features("train")
    X_val, y_val = load_features("val")
    X_test, y_test = load_features("test")

    print(f"Train samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    # Train or load model
    if os.path.exists(model_path):
        print("\nLoading existing Logistic Regression model…")
        model = joblib.load(model_path)
    else:
        print("\nTraining new Logistic Regression model…")
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        print(f"Model saved → {model_path}")

    # Validation evaluation
    print("\nValidation Set Results")
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_prec = precision_score(y_val, val_preds)
    val_rec = recall_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds)
    print(classification_report(y_val, val_preds))
    print(
        f"Validation Accuracy: {val_acc:.4f}  Precision: {val_prec:.4f}  Recall: {val_rec:.4f}  F1: {val_f1:.4f}"
    )

    # Test evaluation
    print("\nTest Set Results")
    test_preds = model.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_prec = precision_score(y_test, test_preds)
    test_rec = recall_score(y_test, test_preds)
    test_f1 = f1_score(y_test, test_preds)
    print(classification_report(y_test, test_preds))
    print(
        f"Test Accuracy: {test_acc:.4f}  Precision: {test_prec:.4f}  Recall: {test_rec:.4f}  F1: {test_f1:.4f}"
    )

    # Confusion matrices
    print("\nGenerating confusion matrices…")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        confusion_matrix(y_val, val_preds),
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
    )
    axes[0].set_title("Validation Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(
        confusion_matrix(y_test, test_preds),
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=axes[1],
    )
    axes[1].set_title("Test Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("./assets/plots/logreg_confusion_matrices.png")
    print("Saved → ./assets/plots/logreg_confusion_matrices.png")

    print("\nSummary")
    print(
        f"Validation F1: {val_f1:.3f} | Test F1: {test_f1:.3f}   Validation Acc: {val_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%"
    )


if __name__ == "__main__":
    run_logreg()
