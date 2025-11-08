#!/usr/bin/env python
import os
import warnings
import json
import argparse
import time
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, metrics
from sklearn.utils.class_weight import compute_class_weight

# Optional quiet mode (keep GPU active)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# warnings.filterwarnings("ignore")

DEFAULT_FEATURE_DIR = "./src/models/model_2/features"
DEFAULT_ARTIFACT_DIR = "./src/models/model_2/artifacts"
DEFAULT_MAXLEN = 567
USE_DATE_FEATURE = False


# ─────────────────────────────────────────────────────────────────────────────
# Attention Layer
# ─────────────────────────────────────────────────────────────────────────────

class Attention(layers.Layer):
    """Custom attention layer that can optionally return attention weights."""
    def __init__(self, return_attention=False, **kwargs):
        super().__init__(**kwargs)
        self.return_attention = return_attention

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        context = inputs * a
        context_vector = tf.keras.backend.sum(context, axis=1)
        if self.return_attention:
            return [context_vector, a]  # return both context and weights
        return context_vector

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1], 1)]
        return (input_shape[0], input_shape[-1])



# ─────────────────────────────────────────────────────────────────────────────
# Data Loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_splits(feature_dir: str):
    """Load NPZ splits (X, y, id, date)."""
    def _load_split(name):
        path = os.path.join(feature_dir, f"{name}_data.npz")
        data = np.load(path)
        X = data["X"]
        y = data["y"]
        ids = data["id"] if "id" in data.files else None
        dates = data["date"] if "date" in data.files else None
        return X, y, ids, dates

    train = _load_split("train")
    val = _load_split("val")
    test = _load_split("test")
    return train, val, test


def load_tokenizer(feature_dir: str):
    """Load saved tokenizer."""
    return joblib.load(os.path.join(feature_dir, "tokenizer.pkl"))


# ─────────────────────────────────────────────────────────────────────────────
# Model Builder (BiLSTM + Attention)
# ─────────────────────────────────────────────────────────────────────────────
def build_bilstm_with_attention(
    vocab_size: int,
    maxlen: int,
    embed_dim: int = 192,
    lstm_units: int = 160,
    use_date: bool = False
):
    """BiLSTM with attention layer and optional date input."""
    text_input = layers.Input(shape=(maxlen,), name="text_input")

    x = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        input_length=maxlen,
        mask_zero=True,
    )(text_input)

    # Bidirectional LSTM with full sequence output for attention
    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.3,
            implementation=2  # non-CuDNN for broader GPU compatibility
        )
    )(x)

    x = Attention()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    if use_date:
        date_input = layers.Input(shape=(1,), name="date_input")
        concat = layers.concatenate([x, date_input])
        x = layers.Dense(16, activation="relu")(concat)
        output = layers.Dense(1, activation="sigmoid")(x)
        model = models.Model(inputs=[text_input, date_input], outputs=output)
    else:
        output = layers.Dense(1, activation="sigmoid")(x)
        model = models.Model(inputs=text_input, outputs=output)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=[
            metrics.BinaryAccuracy(name="accuracy"),
            metrics.AUC(name="auc"),
        ],
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Class Weights
# ─────────────────────────────────────────────────────────────────────────────
def compute_weights_if_binary(y_train: np.ndarray):
    """Automatically balance class weights for binary classification."""
    classes = np.unique(y_train)
    if len(classes) == 2:
        w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        return {int(c): float(wi) for c, wi in zip(classes, w)}
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Epoch Timing Callback
# ─────────────────────────────────────────────────────────────────────────────
class EpochTiming(callbacks.Callback):
    """Display timing info for each epoch."""
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.epoch_times = []
        print("\n[Training started]\n")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start
        self.epoch_times.append(elapsed)
        avg_time = np.mean(self.epoch_times)
        remaining_epochs = self.params["epochs"] - (epoch + 1)
        remaining = avg_time * remaining_epochs
        total_elapsed = time.time() - self.start_time
        print(
            f"[Epoch {epoch + 1}] took {elapsed:.1f}s | "
            f"ETA: {remaining/60:.1f} min | "
            f"Total elapsed: {total_elapsed/60:.1f} min"
        )

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time
        avg_epoch = np.mean(self.epoch_times)
        print(
            f"\n[Training complete] Total time: {total_time/60:.2f} min "
            f"({total_time:.1f}s) | Avg per epoch: {avg_epoch:.1f}s\n"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main Training
# ─────────────────────────────────────────────────────────────────────────────
def main(
    feature_dir: str = DEFAULT_FEATURE_DIR,
    artifact_dir: str = DEFAULT_ARTIFACT_DIR,
    maxlen: int = DEFAULT_MAXLEN,
    epochs: int = 20,
    batch_size: int = 96,
    use_class_weight: bool = True,
    out_model_path: str | None = None,
    use_date: bool = USE_DATE_FEATURE,
):
    os.makedirs(artifact_dir, exist_ok=True)

    (X_train, y_train, id_train, date_train), \
    (X_val, y_val, id_val, date_val), \
    (X_test, y_test, id_test, date_test) = load_splits(feature_dir)

    tok = load_tokenizer(feature_dir)
    vocab_size = len(tok.word_index) + 1

    if X_train.shape[1] != maxlen:
        print(f"[warn] maxlen mismatch: data={X_train.shape[1]} vs arg={maxlen}. Using data shape.")
        maxlen = int(X_train.shape[1])

    if use_date:
        mean, std = np.mean(date_train), np.std(date_train)
        date_train = ((date_train - mean) / (std + 1e-6)).reshape(-1, 1)
        date_val = ((date_val - mean) / (std + 1e-6)).reshape(-1, 1)
        date_test = ((date_test - mean) / (std + 1e-6)).reshape(-1, 1)

    model = build_bilstm_with_attention(vocab_size=vocab_size, maxlen=maxlen, use_date=use_date)

    ckpt_path = out_model_path or os.path.join(artifact_dir, "bilstm_best.keras")
    cbs = [
        EpochTiming(),
        callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, verbose=1),
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    class_weight = compute_weights_if_binary(y_train) if use_class_weight else None

    if use_date:
        train_inputs = {"text_input": X_train, "date_input": date_train}
        val_inputs = {"text_input": X_val, "date_input": date_val}
    else:
        train_inputs = X_train
        val_inputs = X_val

    history = model.fit(
        train_inputs,
        y_train,
        validation_data=(val_inputs, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        class_weight=class_weight,
        verbose=1,
    )

    test_inputs = {"text_input": X_test, "date_input": date_test} if use_date else X_test
    test_loss, test_acc, test_auc = model.evaluate(test_inputs, y_test, verbose=0)
    print({"test_loss": float(test_loss), "test_acc": float(test_acc), "test_auc": float(test_auc)})

    final_path = os.path.join(artifact_dir, "bilstm_final.keras")
    hist_path = os.path.join(artifact_dir, "bilstm_history.json")
    eval_id_path = os.path.join(artifact_dir, "eval_ids.npz")

    model.save(final_path)
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)
    np.savez_compressed(eval_id_path, id_train=id_train, id_val=id_val, id_test=id_test)

    print(f"\nSaved artifacts to → {os.path.abspath(artifact_dir)}")
    print(f" ├─ Best model:  bilstm_best.keras")
    print(f" ├─ Final model: bilstm_final.keras")
    print(f" ├─ History:     bilstm_history.json")
    print(f" └─ Eval IDs:    eval_ids.npz\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BiLSTM + Attention on processed misinformation dataset.")
    parser.add_argument("--features", default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--artifacts", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--maxlen", type=int, default=DEFAULT_MAXLEN)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--out", dest="out_model_path", default=None)
    parser.add_argument("--use-date", action="store_true", help="Include normalized date feature in training")
    args = parser.parse_args()

    main(
        feature_dir=args.features,
        artifact_dir=args.artifacts,
        maxlen=args.maxlen,
        epochs=args.epochs,
        batch_size=args.batch,
        use_class_weight=not args.no_class_weight,
        out_model_path=args.out_model_path,
        use_date=args.use_date,
    )
