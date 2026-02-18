import json
import os
import sys
import re
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import AttentionLayer
OUT_DIR = ROOT / "research" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ROOT / "emotion-model" / "emotion_model_final.keras"
LABELS_PATH = ROOT / "emotion-model" / "label_classes.json"
X_TRAIN_PATH = ROOT / "data" / "processed" / "X_train.npy"
Y_TRAIN_PATH = ROOT / "data" / "processed" / "y_train.npy"
X_TEST_PATH = ROOT / "data" / "processed" / "X_test.npy"
Y_TEST_PATH = ROOT / "data" / "processed" / "y_test.npy"
TRAIN_CSV = ROOT / "data_train.csv"
VAL_CSV = ROOT / "data_validation.csv"
TOKENIZER_JSON = ROOT / "emotion-model" / "tokenizer.json"

EPOCHS = int(os.getenv("GRAPH_EPOCHS", "6"))
BATCH_SIZE = int(os.getenv("GRAPH_BATCH_SIZE", "64"))
TRAIN_LIMIT = int(os.getenv("GRAPH_TRAIN_LIMIT", "12000"))
VAL_LIMIT = int(os.getenv("GRAPH_VAL_LIMIT", "3000"))

GO_EMOTIONS_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy", "love",
    "nervousness", "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

GO_TO_7 = {
    "admiration": "joy",
    "amusement": "joy",
    "anger": "anger",
    "annoyance": "anger",
    "approval": "joy",
    "caring": "joy",
    "confusion": "neutral",
    "curiosity": "neutral",
    "desire": "joy",
    "disappointment": "sadness",
    "disapproval": "anger",
    "disgust": "disgust",
    "embarrassment": "sadness",
    "excitement": "joy",
    "fear": "fear",
    "gratitude": "joy",
    "grief": "sadness",
    "joy": "joy",
    "love": "joy",
    "nervousness": "fear",
    "optimism": "joy",
    "pride": "joy",
    "realization": "neutral",
    "relief": "joy",
    "remorse": "sadness",
    "sadness": "sadness",
    "surprise": "surprise",
    "neutral": "neutral",
}


def load_model(path: Path):
    model = tf.keras.models.load_model(
        path,
        custom_objects={"AttentionLayer": AttentionLayer},
        compile=False,
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def as_model_inputs(model, x_batch):
    # Support both single-input and dict-input models.
    if isinstance(model.input_shape, list) and len(model.input_shape) >= 2:
        return {
            "input_ids": x_batch.astype("int32"),
            "attention_mask": (x_batch != 0).astype("int32"),
        }
    return x_batch.astype("int32")


def normalize_social_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\.\\S+", " URL ", text)
    text = re.sub(r"@\\w+", " USER ", text)
    text = re.sub(r"#(\\w+)", r"\\1", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def parse_labels(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            v = ast.literal_eval(value)
            if isinstance(v, (list, tuple, np.ndarray)):
                return [int(x) for x in v]
        except Exception:
            pass
        nums = re.findall(r"\\d+", value)
        return [int(x) for x in nums]
    return []


def load_xy_from_csvs(labels):
    label_to_id = {l: i for i, l in enumerate(labels)}
    tok_payload = json.loads(TOKENIZER_JSON.read_text(encoding="utf-8"))
    tokenizer = tokenizer_from_json(json.dumps(tok_payload))

    max_len = int(tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"AttentionLayer": AttentionLayer},
        compile=False,
    ).input_shape[1])

    def build(csv_path):
        df = pd.read_csv(csv_path)
        texts, ys = [], []
        for _, row in df.iterrows():
            ids = parse_labels(row.get("labels", ""))
            if not ids:
                continue
            base = GO_EMOTIONS_LABELS[int(ids[0])]
            coarse = GO_TO_7.get(base)
            if coarse not in label_to_id:
                continue
            texts.append(normalize_social_text(row.get("text", "")))
            ys.append(label_to_id[coarse])
        x = pad_sequences(
            tokenizer.texts_to_sequences(texts),
            maxlen=max_len,
            padding="post",
            truncating="post",
        ).astype("int32")
        y = np.array(ys, dtype="int32")
        return x, y

    return build(TRAIN_CSV), build(VAL_CSV)


def plot_train_curves(history):
    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])
    epochs = range(1, len(acc) + 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, acc, marker="o", label="Train")
    ax[0].plot(epochs, val_acc, marker="o", label="Validation")
    ax[0].set_title("Model Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].grid(alpha=0.25)

    ax[1].plot(epochs, loss, marker="o", label="Train")
    ax[1].plot(epochs, val_loss, marker="o", label="Validation")
    ax[1].set_title("Model Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid(alpha=0.25)

    fig.suptitle("Training Curves (Current Model)")
    fig.tight_layout()
    out = OUT_DIR / "fig7_accuracy_loss.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_confusion(cm, labels):
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True",
        xlabel="Predicted",
        title="Confusion Matrix (Current Model)",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )
    fig.tight_layout()
    out = OUT_DIR / "fig8_confusion_matrix.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_per_class_metrics(report, labels):
    p = [report[l]["precision"] for l in labels]
    r = [report[l]["recall"] for l in labels]
    f = [report[l]["f1-score"] for l in labels]
    x = np.arange(len(labels))
    w = 0.26

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar(x - w, p, width=w, label="Precision")
    ax.bar(x, r, width=w, label="Recall")
    ax.bar(x + w, f, width=w, label="F1-score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title("Per-Class Performance Metrics")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "fig9_per_class_metrics.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main():
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    (x_train, y_train), (x_test, y_test) = load_xy_from_csvs(labels)

    if TRAIN_LIMIT > 0 and len(x_train) > TRAIN_LIMIT:
        idx = np.random.RandomState(42).choice(len(x_train), TRAIN_LIMIT, replace=False)
        x_train = x_train[idx]
        y_train = y_train[idx]
    if VAL_LIMIT > 0 and len(x_test) > VAL_LIMIT:
        idx = np.random.RandomState(43).choice(len(x_test), VAL_LIMIT, replace=False)
        x_test = x_test[idx]
        y_test = y_test[idx]

    model = load_model(MODEL_PATH)
    history = model.fit(
        as_model_inputs(model, x_train),
        y_train,
        validation_data=(as_model_inputs(model, x_test), y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    probs = model.predict(as_model_inputs(model, x_test), verbose=0)
    preds = np.argmax(probs, axis=1)

    cm = confusion_matrix(y_test, preds, labels=list(range(len(labels))))
    report = classification_report(
        y_test,
        preds,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    fig7 = plot_train_curves(history)
    fig8 = plot_confusion(cm, labels)
    fig9 = plot_per_class_metrics(report, labels)

    summary = {
        "fig7_accuracy_loss": str(fig7),
        "fig8_confusion_matrix": str(fig8),
        "fig9_per_class_metrics": str(fig9),
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
    }
    (OUT_DIR / "model_graphs_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
