import os
import joblib
import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
)


def evaluate_model(
    model_path="./assets/logreg_model.pkl",
    vec_path="./assets/features/vectoriser.pkl",
    x_test_path="./assets/features/X_test_tfidf.npz",
    y_test_path="./assets/features/y_test.csv",
    model_name="Logistic Regression",
):
    # Load trained model and vectoriser
    model = joblib.load(model_path)
    vectoriser = joblib.load(vec_path)

    # Load unseen TEST split
    print("Loading test split data...")
    X_test = sp.load_npz(x_test_path)
    y_test = pd.read_csv(y_test_path)["label"]

    print(f"Test samples: {X_test.shape[0]}")

    # Predictions and probabilities
    print(f"\nEvaluating {model_name} on unseen test data...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    score = model.score(X_test, y_test)
    print(f"\n- - - - - {model_name} Test Evaluation - - - - -")
    print(f"Accuracy: {score:.4f} ({score * 100:.2f}%)\n")
    print(classification_report(y_test, y_pred, digits=3))

    os.makedirs("./assets/plots", exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["True", "Misinformation"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix – {model_name}")
    plt.tight_layout()
    plt.savefig(f"./assets/plots/{model_name.lower().replace(' ', '_')}_confusion.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="red", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve – {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"./assets/plots/{model_name.lower().replace(' ', '_')}_roc.png")
    plt.close()

    # Precision–Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="darkorange", lw=2, label=f"AP = {ap:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve – {model_name}")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"./assets/plots/{model_name.lower().replace(' ', '_')}_pr.png")
    plt.close()

    # Label Distribution (Test Set)
    plt.figure(figsize=(5, 4))
    sns.countplot(x=y_test, palette="pastel")
    plt.title("Label Distribution (Test Set)")
    plt.xlabel("Label (0=True, 1=Misinformation)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("./assets/plots/label_distribution_test.png")
    plt.close()

    # Summary Output
    print("\nSummary Metrics:")
    print(f"AUC (ROC): {roc_auc:.3f}")
    print(f"Average Precision (AP): {ap:.3f}")
    print("All plots saved in ./assets/plots/")

    return {
        "accuracy": score,
        "auc": roc_auc,
        "average_precision": ap,
    }


if __name__ == "__main__":
    evaluate_model(
        model_path="./assets/logreg_model.pkl",
        vec_path="./assets/features/vectoriser.pkl",
        x_test_path="./assets/features/X_test_tfidf.npz",
        y_test_path="./assets/features/y_test.csv",
        model_name="Logistic Regression",
    )
