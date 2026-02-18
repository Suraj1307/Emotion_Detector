import json
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from src.model import AttentionLayer

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "research" / "outputs" / "research_results.json"
MODEL_PATH = ROOT / "saved_models" / "emotion_model_final.keras"
TOKENIZER_PATH = ROOT / "data" / "processed" / "tokenizer.json"
DEFAULT_LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]


def _load_json(path: Path):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore")
    if text.startswith("version https://git-lfs.github.com/spec/v1"):
        return None
    return json.loads(text)


def load_artifacts():
    results = _load_json(RESULTS_PATH)

    model = None
    tokenizer = None
    labels = list(DEFAULT_LABELS)

    if MODEL_PATH.exists():
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"AttentionLayer": AttentionLayer},
            compile=False,
        )

    tok_payload = _load_json(TOKENIZER_PATH)
    if tok_payload is not None:
        tokenizer = tokenizer_from_json(json.dumps(tok_payload))

    if results and isinstance(results.get("labels"), list) and results.get("labels"):
        labels = results["labels"]

    return results, model, tokenizer, labels


RESULTS, MODEL, TOKENIZER, LABELS = load_artifacts()


def predict_text(text):
    if MODEL is None:
        return "Model not loaded.", ""
    if TOKENIZER is None:
        return (
            "Tokenizer not available (likely LFS pointer). Prediction disabled.",
            "",
        )
    if not text or not text.strip():
        return "Please enter text.", ""

    max_len = int(MODEL.input_shape[1])
    seq = TOKENIZER.texts_to_sequences([text.lower()])
    x = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    probs = MODEL.predict(x, verbose=0)[0]
    pred_id = int(np.argmax(probs))
    conf = float(np.max(probs))

    if LABELS and pred_id < len(LABELS):
        pred_label = LABELS[pred_id]
    else:
        pred_label = f"class_{pred_id}"

    top_idx = np.argsort(probs)[-5:][::-1]
    lines = []
    for i in top_idx:
        lbl = LABELS[i] if LABELS and i < len(LABELS) else f"class_{i}"
        lines.append(f"- {lbl}: {probs[i]*100:.2f}%")

    summary = f"Primary Emotion: {pred_label} ({conf*100:.2f}%)"
    details = "\n".join(lines)
    return summary, details


def plot_distribution():
    fig, ax = plt.subplots(figsize=(8, 4))
    if not RESULTS:
        ax.text(0.5, 0.5, "research_results.json not found", ha="center", va="center")
        ax.axis("off")
        return fig

    dist = RESULTS.get("test_distribution", [])
    labels = [d["label"] for d in dist]
    counts = [d["count"] for d in dist]
    ax.bar(labels, counts)
    ax.set_title("Test Class Distribution")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig


def plot_confusion_matrix():
    fig, ax = plt.subplots(figsize=(7, 6))
    if not RESULTS:
        ax.text(0.5, 0.5, "research_results.json not found", ha="center", va="center")
        ax.axis("off")
        return fig

    cm = np.array(RESULTS.get("main_metrics", {}).get("confusion_matrix", []))
    labels = RESULTS.get("labels", [str(i) for i in range(cm.shape[0])])
    if cm.size == 0:
        ax.text(0.5, 0.5, "Confusion matrix missing", ha="center", va="center")
        ax.axis("off")
        return fig

    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def metrics_table_md():
    if not RESULTS:
        return "research_results.json not available."
    rep = RESULTS.get("main_metrics", {}).get("report", {})
    macro = rep.get("macro avg", {})
    return (
        "| Metric | Value |\n"
        "|---|---:|\n"
        f"| Accuracy | {rep.get('accuracy', 0):.4f} |\n"
        f"| Macro Precision | {macro.get('precision', 0):.4f} |\n"
        f"| Macro Recall | {macro.get('recall', 0):.4f} |\n"
        f"| Macro F1 | {macro.get('f1-score', 0):.4f} |\n"
        f"| Inference ms/sample | {RESULTS.get('main_metrics', {}).get('inference_ms_per_sample', 0):.4f} |\n"
        f"| Model size (MB) | {RESULTS.get('model_size_mb', 0):.3f} |"
    )


with gr.Blocks(title="Emotion Classifier + Research Graphs") as demo:
    gr.Markdown("# Emotion Classification Dashboard")
    gr.Markdown("Prediction + research evaluation graphs from `research/outputs/research_results.json`.")

    with gr.Row():
        with gr.Column(scale=1):
            txt = gr.Textbox(label="Input Text", lines=4)
            btn = gr.Button("Predict")
            out_main = gr.Textbox(label="Primary Prediction")
            out_top = gr.Textbox(label="Top-5 Probabilities")
            btn.click(predict_text, inputs=txt, outputs=[out_main, out_top])

            gr.Markdown("## Core Metrics")
            gr.Markdown(metrics_table_md())

        with gr.Column(scale=1):
            dist_plot = gr.Plot(value=plot_distribution)
            cm_plot = gr.Plot(value=plot_confusion_matrix)

if __name__ == "__main__":
    demo.launch()
