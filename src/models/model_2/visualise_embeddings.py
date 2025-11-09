#!/usr/bin/env python3
"""
Visualize BiLSTM embedding space and save to CSV.

Generates 2D t-SNE coordinates for each test sample,
with correctness flags and predicted labels, saved to:
    src/models/model_2/artifacts/tsne_embeddings.csv

Usage:
  python -m src.models.model_2.visualize_embeddings
"""

import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model, Model
import tensorflow as tf

# Import custom Attention layer
from src.models.model_2.train_bilstm import Attention


# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
BASE_DIR = os.path.join(PROJECT_ROOT, "src", "models", "model_2")

ARTIFACTS = os.path.join(BASE_DIR, "artifacts")
FEATURES = os.path.join(BASE_DIR, "features")

MODEL_PATH = os.path.join(ARTIFACTS, "bilstm_best.keras")
TOKENIZER_PATH = os.path.join(FEATURES, "tokenizer.pkl")
TEST_DATA_PATH = os.path.join(FEATURES, "test_data.npz")
PREDICTIONS_PATH = os.path.join(ARTIFACTS, "predictions_detailed.csv")
OUTPUT_PATH = os.path.join(ARTIFACTS, "tsne_embeddings.csv")


# ---------------------------------------------------------------------
# Load model + data
# ---------------------------------------------------------------------
print("[INFO] Loading model and data...")

model = load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={"Attention": Attention}
)

tokenizer = load(TOKENIZER_PATH)
test_data = np.load(TEST_DATA_PATH)
X_test, y_test = test_data["X"], test_data["y"]

preds_df = pd.read_csv(PREDICTIONS_PATH)
correct_flags = preds_df["correct"].to_numpy()

min_len = min(len(X_test), len(correct_flags))
X_test = X_test[:min_len]
correct_flags = correct_flags[:min_len]
preds_df = preds_df.iloc[:min_len].copy()


# ---------------------------------------------------------------------
# Find the embedding layer dynamically
# ---------------------------------------------------------------------
embedding_layer = None
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Embedding):
        embedding_layer = layer
        break

if embedding_layer is None:
    raise ValueError("No Embedding layer found in model! Check model summary.")

print(f"[INFO] Found embedding layer: {embedding_layer.name}")

embedding_extractor = Model(inputs=model.input, outputs=embedding_layer.output)

# Extract embeddings
embeddings = embedding_extractor.predict(X_test, batch_size=64, verbose=1)
print("[DEBUG] Embedding array shape:", embeddings.shape)

# Average token vectors → one per sequence
avg_embeddings = np.mean(embeddings, axis=1)
print("[DEBUG] Averaged embedding shape:", avg_embeddings.shape)

if avg_embeddings.ndim == 1:
    avg_embeddings = avg_embeddings.reshape(-1, 1)


# ---------------------------------------------------------------------
# Dimensionality reduction (t-SNE)
# ---------------------------------------------------------------------
print("[INFO] Reducing to 2D with t-SNE (this can take a few minutes)...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, init="pca")
reduced = tsne.fit_transform(avg_embeddings)
print("[INFO] t-SNE complete. Shape:", reduced.shape)


# ---------------------------------------------------------------------
# Merge and save results
# ---------------------------------------------------------------------
print("[INFO] Saving reduced embeddings to CSV...")
output_df = pd.DataFrame({
    "id": preds_df["id"].values,
    "label": preds_df["label"].values,
    "pred_label": preds_df["pred_label"].values,
    "correct": preds_df["correct"].values,
    "tsne_x": reduced[:, 0],
    "tsne_y": reduced[:, 1]
})

output_df.to_csv(OUTPUT_PATH, index=False)
print(f"[DONE] Saved 2D embeddings to: {OUTPUT_PATH}")

