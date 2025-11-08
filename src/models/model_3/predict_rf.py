#!/usr/bin/env python
import os, re, json, joblib, numpy as np, warnings
from textblob import TextBlob
import textstat, spacy
from empath import Empath

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATH = "./src/models/model_3/artifacts/rf_linguistic_model.pkl"
FEATURE_NAMES_PATH = "./src/models/model_3/features/feature_names.pkl"

nlp = spacy.load("en_core_web_sm")
lexicon = Empath()

def extract_single_features(text: str) -> dict:
    doc = nlp(text)
    blob = TextBlob(text)
    words = [w.text for w in doc if w.is_alpha]
    n_words = len(words)
    n_chars = len(text)
    n_sents = max(1, len(list(doc.sents)))

    exclam = text.count("!")
    caps_ratio = sum(1 for c in text if c.isupper()) / max(1, n_chars)
    punct_ratio = sum(1 for c in text if re.match(r"[^\w\s]", c)) / max(1, n_chars)
    unique_ratio = len(set(words)) / max(1, n_words)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    flesch = textstat.flesch_reading_ease(text)
    fog = textstat.gunning_fog(text)
    pos_counts = doc.count_by(spacy.attrs.POS)
    noun_ratio = pos_counts.get(92, 0) / max(1, n_words)
    adj_ratio = pos_counts.get(84, 0) / max(1, n_words)
    adv_ratio = pos_counts.get(85, 0) / max(1, n_words)
    verb_ratio = pos_counts.get(100, 0) / max(1, n_words)
    ner_count = len(doc.ents)
    empath_scores = lexicon.analyze(
        text,
        categories=[
            "politics", "government", "violence", "deception",
            "trust", "law", "money", "emotion", "hate"
        ],
        normalize=True
    )
    bias_intensity = sum(empath_scores.values())

    feats = {
        "n_chars": n_chars, "n_words": n_words, "n_sents": n_sents, "exclam": exclam,
        "caps_ratio": caps_ratio, "punct_ratio": punct_ratio, "unique_ratio": unique_ratio,
        "avg_word_len": avg_word_len, "polarity": polarity, "subjectivity": subjectivity,
        "flesch": flesch, "fog": fog, "noun_ratio": noun_ratio, "adj_ratio": adj_ratio,
        "adv_ratio": adv_ratio, "verb_ratio": verb_ratio, "ner_count": ner_count,
        "bias_intensity": bias_intensity, "date_days": 0.0
    }
    return feats

def predict_from_json(json_input: str):
    if os.path.isfile(json_input):
        data = json.load(open(json_input, "r", encoding="utf-8"))
    else:
        data = json.loads(json_input)

    text = data["text"]
    feats = extract_single_features(text)

    # ── NEW: feature alignment guard ─────────────────────────────────────
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    expected = set(feature_names)
    actual = set(feats.keys())
    missing = list(expected - actual)
    extra = list(actual - expected)
    if missing:
        print(f"[WARN] Missing {len(missing)} features not in extractor: {missing}")
        for m in missing:
            feats[m] = 0.0
    if extra:
        print(f"[INFO] Ignoring {len(extra)} unexpected features: {extra}")

    X = np.array([[feats[col] for col in feature_names]], dtype=np.float32)
    # ─────────────────────────────────────────────────────────────────────

    model = joblib.load(MODEL_PATH)
    prob = float(model.predict_proba(X)[0, 1])
    label = int(prob >= 0.5)
    confidence = prob if label == 1 else (1 - prob)

    summary_keys = ["n_words", "caps_ratio", "polarity", "bias_intensity"]
    feature_summary = {k: float(feats[k]) for k in summary_keys}

    return {
        "id": data.get("id"), "date": data.get("date"), "text": text,
        "predicted_label": label, "rf_prob": prob,
        "confidence": confidence, "feature_summary": feature_summary
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RandomForest stylistic model on JSON input.")
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    result = predict_from_json(args.json)
    print(json.dumps(result, indent=2))

