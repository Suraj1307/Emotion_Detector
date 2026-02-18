import ast
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import AttentionLayer

OUT_DIR = ROOT / "research" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ROOT / "emotion-model" / "emotion_model_final.keras"
TOKENIZER_PATH = ROOT / "emotion-model" / "tokenizer.json"
LABELS_PATH = ROOT / "emotion-model" / "label_classes.json"
VAL_CSV = ROOT / "data_validation.csv"

GO_EMOTIONS_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
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
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [int(x) for x in parsed]
        except Exception:
            nums = re.findall(r"\\d+", value)
            return [int(n) for n in nums]
    return []


def load_eval_data(labels):
    label_to_id = {lbl: i for i, lbl in enumerate(labels)}
    df = pd.read_csv(VAL_CSV)

    texts = []
    y_true = []
    for _, row in df.iterrows():
        ids = parse_labels(row.get("labels", ""))
        if not ids:
            continue
        fine_lbl = GO_EMOTIONS_LABELS[int(ids[0])]
        coarse = GO_TO_7.get(fine_lbl)
        if coarse not in label_to_id:
            continue
        texts.append(normalize_social_text(row.get("text", "")))
        y_true.append(label_to_id[coarse])
    return texts, np.array(y_true, dtype=np.int32)


def plot_confusion(cm, labels):
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True",
        xlabel="Predicted",
        title="Final Model Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
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
    out = OUT_DIR / "final_model_confusion_matrix.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_markdown_table(report, labels):
    lines = []
    lines.append("| Metric | Value | Derived From |")
    lines.append("|---|---:|---|")
    lines.append(f"| Accuracy | {report['accuracy']:.4f} | Sum of diagonal / Total samples |")
    lines.append(f"| Macro F1 | {report['macro avg']['f1-score']:.4f} | Mean of per-class F1 scores |")
    lines.append("")
    lines.append("| Class | Precision | Recall | F1-score |")
    lines.append("|---|---:|---:|---:|")
    for lbl in labels:
        row = report[lbl]
        lines.append(
            f"| {lbl} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} |"
        )

    out = OUT_DIR / "final_model_metrics_table.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main():
    labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    tok_payload = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))
    tokenizer = tokenizer_from_json(json.dumps(tok_payload))

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"AttentionLayer": AttentionLayer},
        compile=False,
    )

    texts, y_true = load_eval_data(labels)
    max_len = int(model.input_shape[1])
    x_val = pad_sequences(
        tokenizer.texts_to_sequences(texts),
        maxlen=max_len,
        padding="post",
        truncating="post",
    ).astype("int32")

    probs = model.predict(x_val, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    cm_path = plot_confusion(cm, labels)
    md_path = write_markdown_table(report, labels)

    out_json = {
        "labels": labels,
        "accuracy": float(report["accuracy"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_path": str(cm_path),
        "metrics_table_path": str(md_path),
    }
    json_path = OUT_DIR / "final_model_metrics.json"
    json_path.write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    print(json.dumps(
        {
            "accuracy": out_json["accuracy"],
            "macro_f1": out_json["macro_f1"],
            "confusion_matrix_path": str(cm_path),
            "metrics_table_path": str(md_path),
            "json_path": str(json_path),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
