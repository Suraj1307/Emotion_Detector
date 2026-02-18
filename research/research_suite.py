import json
import os
import time
import ast
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import AttentionLayer
DATA_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "saved_models" / "emotion_model_final.keras"
OUT_DIR = ROOT / "research" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]


def load_json_if_available(path: Path):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    if text.startswith("version https://git-lfs.github.com/spec/v1"):
        return None
    return json.loads(text)


def load_data():
    x_train = np.load(DATA_DIR / "X_train.npy")
    x_test = np.load(DATA_DIR / "X_test.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    y_test = np.load(DATA_DIR / "y_test.npy")
    return x_train, x_test, y_train, y_test


def load_labels() -> List[str]:
    payload = load_json_if_available(DATA_DIR / "label_encoder.json")
    if isinstance(payload, dict) and "classes" in payload:
        return payload["classes"]
    if isinstance(payload, list):
        return payload
    return DEFAULT_LABELS


def load_model():
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"AttentionLayer": AttentionLayer},
        compile=False,
    )
    return model


def class_distribution(y: np.ndarray, labels: List[str]) -> List[Dict]:
    counts = Counter(y.tolist())
    total = len(y)
    out = []
    for i, label in enumerate(labels):
        c = int(counts.get(i, 0))
        out.append(
            {"class_id": i, "label": label, "count": c, "pct": round(100.0 * c / total, 2)}
        )
    return out


def dataset_stats(x_train, x_test):
    x_all = np.vstack([x_train, x_test])
    lengths = (x_all != 0).sum(axis=1)
    non_zero = x_all[x_all != 0]
    vocab_from_sequences = int(non_zero.max()) if non_zero.size > 0 else 0
    oov_ratio = float((x_all == 1).sum() / np.prod(x_all.shape))
    return {
        "num_train": int(x_train.shape[0]),
        "num_test": int(x_test.shape[0]),
        "max_len": int(x_all.shape[1]),
        "avg_len": round(float(lengths.mean()), 3),
        "median_len": float(np.median(lengths)),
        "p95_len": float(np.percentile(lengths, 95)),
        "vocab_size_estimate": vocab_from_sequences,
        "oov_token_ratio": round(oov_ratio, 6),
    }


def evaluate_main_model(model, x_test, y_test, labels):
    target_len = int(model.input_shape[1])
    if x_test.shape[1] < target_len:
        pad = np.zeros((x_test.shape[0], target_len - x_test.shape[1]), dtype=x_test.dtype)
        x_eval = np.concatenate([x_test, pad], axis=1)
    elif x_test.shape[1] > target_len:
        x_eval = x_test[:, :target_len]
    else:
        x_eval = x_test

    t0 = time.perf_counter()
    probs = model.predict(x_eval, verbose=0)
    elapsed = time.perf_counter() - t0
    preds = probs.argmax(axis=1)
    report = classification_report(
        y_test,
        preds,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, preds, labels=list(range(len(labels))))
    return {
        "preds": preds.tolist(),
        "probs": probs.tolist(),
        "report": report,
        "confusion_matrix": cm.tolist(),
        "inference_ms_per_sample": round((elapsed * 1000.0) / len(x_test), 4),
    }


def top_misclassified(x_test, y_test, probs, labels, top_n=5):
    preds = np.argmax(probs, axis=1)
    rows = []
    for i in range(len(y_test)):
        if preds[i] != y_test[i]:
            rows.append(
                {
                    "idx": int(i),
                    "true_label": labels[int(y_test[i])],
                    "pred_label": labels[int(preds[i])],
                    "pred_conf": round(float(np.max(probs[i])), 4),
                    "token_ids_excerpt": x_test[i][:20].tolist(),
                }
            )
    rows.sort(key=lambda r: r["pred_conf"], reverse=True)
    return rows[:top_n]


def attention_validation(model, x_test, y_test, labels):
    keyword_map = {
        "anger": {"hate", "angry", "furious", "worst", "terrible"},
        "disgust": {"disgust", "gross", "nasty", "sick"},
        "fear": {"fear", "scared", "afraid", "panic", "anxious"},
        "joy": {"happy", "joy", "thrilled", "excited", "grateful"},
        "neutral": {"okay", "fine", "normal"},
        "sadness": {"sad", "depressed", "down", "heartbroken", "lonely"},
        "surprise": {"wow", "surprised", "unexpected", "suddenly"},
    }

    # Without a non-LFS tokenizer file we cannot decode token ids to words;
    # keep method explicit and return a validity flag.
    tokenizer_available = load_json_if_available(DATA_DIR / "tokenizer.json") is not None
    if not tokenizer_available:
        return {
            "tokenizer_available": False,
            "note": "Tokenizer JSON unavailable (LFS pointer), lexical attention validation skipped.",
            "keyword_lexicon_size": sum(len(v) for v in keyword_map.values()),
        }

    return {
        "tokenizer_available": True,
        "note": "Tokenizer available. Extend this block to decode tokens and compute overlap scores.",
        "keyword_lexicon_size": sum(len(v) for v in keyword_map.values()),
    }


def run_ablation(x_train, y_train, x_test, y_test, labels, variant: str):
    inp = tf.keras.Input(shape=(x_train.shape[1],))
    x = tf.keras.layers.Embedding(50000, 64)(inp)

    if variant != "no_bilstm":
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32, return_sequences=True)
        )(x)

    if variant != "no_cnn":
        x = tf.keras.layers.Conv1D(64, 3, activation="relu", padding="same")(x)

    if variant != "no_attention":
        x = AttentionLayer()(x)
    else:
        x = tf.keras.layers.GlobalMaxPooling1D()(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(len(labels), activation="softmax")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(x_train, y_train, epochs=1, batch_size=128, verbose=0)
    preds = np.argmax(m.predict(x_test, verbose=0), axis=1)
    rep = classification_report(
        y_test,
        preds,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "variant": variant,
        "accuracy": round(rep["accuracy"], 4),
        "macro_f1": round(rep["macro avg"]["f1-score"], 4),
    }


def baseline_svm(x_train, y_train, x_test, y_test, labels):
    clf = make_pipeline(StandardScaler(with_mean=False), LinearSVC())
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    rep = classification_report(
        y_test,
        preds,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "model": "LinearSVM",
        "accuracy": round(rep["accuracy"], 4),
        "macro_f1": round(rep["macro avg"]["f1-score"], 4),
    }


def baseline_lstm(x_train, y_train, x_test, y_test, labels):
    inp = tf.keras.Input(shape=(x_train.shape[1],))
    x = tf.keras.layers.Embedding(50000, 64)(inp)
    x = tf.keras.layers.LSTM(64)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    out = tf.keras.layers.Dense(len(labels), activation="softmax")(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(x_train, y_train, epochs=1, batch_size=128, verbose=0)
    preds = np.argmax(m.predict(x_test, verbose=0), axis=1)
    rep = classification_report(
        y_test,
        preds,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return {
        "model": "LSTM",
        "accuracy": round(rep["accuracy"], 4),
        "macro_f1": round(rep["macro avg"]["f1-score"], 4),
    }


def robustness_tests(model, x_test, y_test, labels):
    lengths = (x_test != 0).sum(axis=1)
    buckets = {
        "short_0_10": np.where(lengths <= 10)[0],
        "medium_11_30": np.where((lengths > 10) & (lengths <= 30))[0],
        "long_31_plus": np.where(lengths > 30)[0],
    }
    results = []
    for name, idxs in buckets.items():
        if len(idxs) == 0:
            continue
        probs = model.predict(x_test[idxs], verbose=0)
        preds = probs.argmax(axis=1)
        rep = classification_report(
            y_test[idxs],
            preds,
            labels=list(range(len(labels))),
            target_names=labels,
            output_dict=True,
            zero_division=0,
        )
        results.append(
            {
                "bucket": name,
                "n": int(len(idxs)),
                "accuracy": round(rep["accuracy"], 4),
                "macro_f1": round(rep["macro avg"]["f1-score"], 4),
            }
        )
    return results


def co_occurrence_analysis():
    raw_path = ROOT / "data" / "raw" / "emotion_data.csv"
    if not raw_path.exists():
        return {
            "available": False,
            "note": "Raw multi-label file not present; co-occurrence not computed.",
            "top_pairs": [],
        }
    df = pd.read_csv(raw_path)
    if "labels" not in df.columns:
        return {"available": False, "note": "No labels column in raw data.", "top_pairs": []}

    pairs = Counter()
    for raw in df["labels"].fillna("[]").astype(str):
        try:
            ids = ast.literal_eval(raw)
        except Exception:
            ids = []
        if len(ids) < 2:
            continue
        for p in combinations(sorted(set(ids)), 2):
            pairs[p] += 1
    top = [{"pair": list(k), "count": int(v)} for k, v in pairs.most_common(20)]
    return {"available": True, "top_pairs": top}


def model_size_mb(path: Path):
    return round(path.stat().st_size / (1024 * 1024), 3)


def build_markdown_summary(res: Dict):
    rep = res["main_metrics"]["report"]
    macro = rep["macro avg"]
    weighted = rep["weighted avg"]

    lines = []
    lines.append("# Research Evaluation Summary")
    lines.append("")
    lines.append("## Core Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---:|")
    lines.append(f"| Accuracy | {rep['accuracy']:.4f} |")
    lines.append(f"| Macro Precision | {macro['precision']:.4f} |")
    lines.append(f"| Macro Recall | {macro['recall']:.4f} |")
    lines.append(f"| Macro F1 | {macro['f1-score']:.4f} |")
    lines.append(f"| Weighted F1 | {weighted['f1-score']:.4f} |")
    lines.append(f"| Inference ms/sample | {res['main_metrics']['inference_ms_per_sample']:.4f} |")
    lines.append(f"| Model Size (MB) | {res['model_size_mb']:.3f} |")
    lines.append("")

    lines.append("## Baseline Comparison")
    lines.append("")
    lines.append("| Model | Accuracy | Macro-F1 |")
    lines.append("|---|---:|---:|")
    for b in res["baselines"]:
        lines.append(f"| {b['model']} | {b['accuracy']:.4f} | {b['macro_f1']:.4f} |")
    lines.append("")

    lines.append("## Ablation Study")
    lines.append("")
    lines.append("| Variant | Accuracy | Macro-F1 |")
    lines.append("|---|---:|---:|")
    for a in res["ablations"]:
        lines.append(f"| {a['variant']} | {a['accuracy']:.4f} | {a['macro_f1']:.4f} |")
    lines.append("")

    lines.append("## Robustness by Length")
    lines.append("")
    lines.append("| Bucket | N | Accuracy | Macro-F1 |")
    lines.append("|---|---:|---:|---:|")
    for r in res["robustness"]:
        lines.append(f"| {r['bucket']} | {r['n']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} |")
    lines.append("")

    lines.append("## Top 5 Misclassified Examples")
    lines.append("")
    lines.append("| Idx | True | Pred | Confidence | Token Id Excerpt |")
    lines.append("|---:|---|---|---:|---|")
    for m in res["misclassified"]:
        lines.append(
            f"| {m['idx']} | {m['true_label']} | {m['pred_label']} | {m['pred_conf']:.4f} | `{m['token_ids_excerpt']}` |"
        )
    lines.append("")

    return "\n".join(lines)


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    labels = load_labels()
    x_train, x_test, y_train, y_test = load_data()
    model = load_model()

    main_metrics = evaluate_main_model(model, x_test, y_test, labels)
    miscls = top_misclassified(x_test, y_test, np.array(main_metrics["probs"]), labels, top_n=5)
    attn_val = attention_validation(model, x_test, y_test, labels)

    # Baselines and ablations use a smaller subset to keep runtime manageable.
    n_train = min(12000, len(x_train))
    n_test = min(3000, len(x_test))
    idx_tr = np.random.choice(len(x_train), n_train, replace=False)
    idx_te = np.random.choice(len(x_test), n_test, replace=False)
    xtr, ytr = x_train[idx_tr], y_train[idx_tr]
    xte, yte = x_test[idx_te], y_test[idx_te]

    baselines = [
        baseline_svm(xtr, ytr, xte, yte, labels),
        baseline_lstm(xtr, ytr, xte, yte, labels),
        {
            "model": "Attention-BiLSTM-CNN (Main)",
            "accuracy": round(main_metrics["report"]["accuracy"], 4),
            "macro_f1": round(main_metrics["report"]["macro avg"]["f1-score"], 4),
        },
    ]

    ablations = [
        run_ablation(xtr, ytr, xte, yte, labels, "no_cnn"),
        run_ablation(xtr, ytr, xte, yte, labels, "no_attention"),
        run_ablation(xtr, ytr, xte, yte, labels, "no_bilstm"),
    ]

    robustness = robustness_tests(model, x_test, y_test, labels)

    results = {
        "labels": labels,
        "dataset_stats": dataset_stats(x_train, x_test),
        "train_distribution": class_distribution(y_train, labels),
        "test_distribution": class_distribution(y_test, labels),
        "co_occurrence": co_occurrence_analysis(),
        "main_metrics": main_metrics,
        "misclassified": miscls,
        "attention_validation": attn_val,
        "ablations": ablations,
        "baselines": baselines,
        "robustness": robustness,
        "model_size_mb": model_size_mb(MODEL_PATH),
        "hyperparameters": {
            "max_len": int(x_train.shape[1]),
            "batch_size": 256,
            "learning_rate": 2e-4,
            "dropout_lstm": 0.4,
            "dropout_dense": 0.5,
            "embedding_dim": 100,
            "bilstm_hidden_units": 64,
            "cnn_filters": 64,
            "cnn_kernel_size": 3,
            "epochs": 8,
        },
    }

    (OUT_DIR / "research_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "research_summary.md").write_text(
        build_markdown_summary(results), encoding="utf-8"
    )
    print("Saved:")
    print(OUT_DIR / "research_results.json")
    print(OUT_DIR / "research_summary.md")


if __name__ == "__main__":
    main()
