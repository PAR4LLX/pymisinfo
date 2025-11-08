#!/usr/bin/env python
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load as joblib_load

# Unified cleaner
from src.core.data_cleansing import clean_json_input
from src.models.model_2.train_bilstm import Attention  # custom attention layer

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
MODEL_PATH = "./src/models/model_2/artifacts/bilstm_best.keras"
TOKENIZER_PATH = "./src/models/model_2/features/tokenizer.pkl"
MAXLEN = 567


# ---------------------------------------------------------------------
# Safe tokenizer loader (for keras.src.preprocessing compatibility)
# ---------------------------------------------------------------------
def safe_load_tokenizer(path: str):
    try:
        return joblib_load(path)
    except ModuleNotFoundError as e:
        if "keras.src.preprocessing" in str(e):
            import sys, keras.preprocessing.text as keras_pre_old
            sys.modules["keras.src.preprocessing.text"] = keras_pre_old
            return joblib_load(path)
        raise


# ---------------------------------------------------------------------
# Prediction + Token Weight Extraction
# ---------------------------------------------------------------------
def predict_from_json(json_input: str):
    # Load and clean JSON input
    data = json.loads(json_input)
    cleaned = clean_json_input(data)
    text = cleaned["text"]

    tokenizer = safe_load_tokenizer(TOKENIZER_PATH)
    seq = tokenizer.texts_to_sequences([text])
    seq_padded = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")

    model = load_model(MODEL_PATH, custom_objects={"Attention": Attention})

    prob = float(model.predict(seq_padded, verbose=0)[0][0])
    predicted_label = int(round(prob))
    confidence = prob if predicted_label == 1 else (1 - prob)

    # Extract attention weights if available
    attention_layer = next((l for l in model.layers if isinstance(l, Attention)), None)
    tokens_output = []
    summary = {}

    if attention_layer is not None:
        att_model = tf.keras.Model(inputs=model.input, outputs=attention_layer.output)
        att_raw = att_model.predict(seq_padded, verbose=0)

        att_raw = np.squeeze(att_raw)
        if att_raw.ndim == 1:
            att_raw = np.expand_dims(att_raw, axis=0)

        weights = np.mean(att_raw, axis=-1)
        token_len = len(seq[0])
        if weights.ndim == 0:
            weights = np.zeros(token_len)
        elif weights.shape[0] != token_len:
            weights = np.pad(weights, (0, max(0, token_len - len(weights))), mode="constant")[:token_len]

        weights = np.exp(weights) / np.sum(np.exp(weights)) if np.sum(weights) > 0 else np.zeros_like(weights)

        inv_vocab = {v: k for k, v in tokenizer.word_index.items()}
        tokens = [inv_vocab.get(t, "") for t in seq[0]]
        tokens_output = [{"word": w, "weight": float(weights[i])} for i, w in enumerate(tokens) if w]

        top_weights = sorted(weights, reverse=True)[:10]
        summary = {
            "token_count": len(tokens),
            "avg_weight": float(np.mean(weights)),
            "max_weight": float(np.max(weights)),
            "top10_total": float(np.sum(top_weights)),
            "top10_mean": float(np.mean(top_weights))
        }
    else:
        summary = {
            "token_count": 0,
            "avg_weight": 0.0,
            "max_weight": 0.0,
            "top10_total": 0.0,
            "top10_mean": 0.0
        }

    return {
        "id": data.get("id"),
        "date": data.get("date"),
        "text": text,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "tokens": tokens_output,
        "summary": summary
    }


# ---------------------------------------------------------------------
# CLI Interface
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Predict misinformation probability and attention weights from JSON input.")
    p.add_argument("--json", required=True, help="JSON string or path to JSON file containing id, text, and date.")
    args = p.parse_args()

    if os.path.isfile(args.json):
        with open(args.json, "r", encoding="utf-8") as f:
            json_str = f.read()
    else:
        json_str = args.json

    result = predict_from_json(json_str)
    print(json.dumps(result, indent=2))

