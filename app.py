import json
import os
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from src.model import AttentionLayer

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "research" / "outputs" / "research_results.json"

MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "SurajAI2025/emotion-model-7")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "emotion_model_final.keras")

DEFAULT_LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
        return first_line.strip() == "version https://git-lfs.github.com/spec/v1"
    except Exception:
        return False


def _load_json(path: Path):
    if not path.exists() or _is_lfs_pointer(path):
        return None
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def _resolve_local_or_hf(candidates):
    for candidate in candidates:
        p = Path(candidate)
        if p.exists() and not _is_lfs_pointer(p):
            return str(p)

    for candidate in candidates:
        try:
            return hf_hub_download(repo_id=MODEL_REPO_ID, filename=candidate)
        except Exception:
            continue

    return None


def _load_results():
    return _load_json(RESULTS_PATH)


def _load_labels(results):
    candidates = [
        ROOT / "data" / "processed" / "label_encoder.json",
        ROOT / "label_encoder.json",
        ROOT / "label_classes.json",
    ]
    for c in candidates:
        payload = _load_json(c)
        if isinstance(payload, dict) and "classes" in payload:
            return payload["classes"]
        if isinstance(payload, list) and payload:
            return payload

    if isinstance(results, dict) and isinstance(results.get("labels"), list) and results.get("labels"):
        return results["labels"]

    return list(DEFAULT_LABELS)


def _load_model():
    model_candidates = [
        str(ROOT / "saved_models" / MODEL_FILENAME),
        str(ROOT / MODEL_FILENAME),
        MODEL_FILENAME,
        f"saved_models/{MODEL_FILENAME}",
    ]
    model_file = _resolve_local_or_hf(model_candidates)
    if not model_file:
        return None, "Model artifact not found locally or on HF repo."
    try:
        model = tf.keras.models.load_model(
            model_file,
            custom_objects={"AttentionLayer": AttentionLayer},
            compile=False,
        )
        return model, f"Model loaded from: {model_file}"
    except Exception as e:
        return None, f"Model load error: {type(e).__name__}: {e}"


def _load_tokenizer():
    tok_candidates = [
        str(ROOT / "data" / "processed" / "tokenizer.json"),
        str(ROOT / "tokenizer.json"),
        "tokenizer.json",
        "data/processed/tokenizer.json",
    ]
    tok_file = _resolve_local_or_hf(tok_candidates)
    if not tok_file:
        return None, "Tokenizer artifact not found locally or on HF repo."
    try:
        payload = _load_json(Path(tok_file))
        if payload is None:
            return None, "Tokenizer file exists but is unreadable or LFS pointer."
        tok = tokenizer_from_json(json.dumps(payload))
        return tok, f"Tokenizer loaded from: {tok_file}"
    except Exception as e:
        return None, f"Tokenizer load error: {type(e).__name__}: {e}"


def load_artifacts():
    results = _load_results()
    labels = _load_labels(results)

    model, model_msg = _load_model()
    tokenizer, tok_msg = _load_tokenizer()

    status = [
        f"MODEL_REPO_ID: {MODEL_REPO_ID}",
        model_msg,
        tok_msg,
        f"Labels source: {'research_results' if results and results.get('labels') else 'default/local'}",
    ]

    return results, model, tokenizer, labels, "\n".join(status)


RESULTS, MODEL, TOKENIZER, LABELS, INIT_STATUS = load_artifacts()


def predict_text(text):
    if MODEL is None:
        return "Model not loaded.", INIT_STATUS
    if TOKENIZER is None:
        return "Tokenizer not loaded.", INIT_STATUS
    if not text or not text.strip():
        return "Please enter text.", ""

    max_len = int(MODEL.input_shape[1])
    seq = TOKENIZER.texts_to_sequences([text.lower()])
    x = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    probs = MODEL.predict(x, verbose=0)[0]
    pred_id = int(np.argmax(probs))
    conf = float(np.max(probs))

    pred_label = LABELS[pred_id] if pred_id < len(LABELS) else f"class_{pred_id}"

    top_idx = np.argsort(probs)[-5:][::-1]
    lines = []
    for i in top_idx:
        lbl = LABELS[i] if i < len(LABELS) else f"class_{i}"
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
    labels = [d.get("label", "") for d in dist]
    counts = [d.get("count", 0) for d in dist]

    if not labels:
        ax.text(0.5, 0.5, "Distribution data missing", ha="center", va="center")
        ax.axis("off")
        return fig

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


with gr.Blocks(title="AffectLens AI: Emotion Intelligence Studio") as demo:
    gr.Markdown("# AffectLens AI: Emotion Intelligence Studio")
    gr.Markdown(
        "Prediction + research evaluation graphs from `research/outputs/research_results.json`."
    )
    gr.Markdown("## Initialization Status")
    gr.Code(INIT_STATUS, language="text")

    with gr.Row():
        with gr.Column(scale=1):
            txt = gr.Textbox(label="Input Text", lines=4)
            btn = gr.Button("Predict")
            out_main = gr.Textbox(label="Primary Prediction")
            out_top = gr.Textbox(label="Top-5 Probabilities")
            btn.click(predict_text, inputs=txt, outputs=[out_main, out_top])

            gr.Examples(
                examples=[
                    ["I am absolutely thrilled and grateful for this wonderful news."],
                    ["This is terrible and unacceptable. I am furious and angry."],
                    ["I feel sad, hopeless, and heartbroken today."],
                ],
                inputs=txt,
                label="Quick Test Examples",
            )

            gr.Markdown("## Core Metrics")
            gr.Markdown(metrics_table_md())

        with gr.Column(scale=1):
            gr.Plot(value=plot_distribution)
            gr.Plot(value=plot_confusion_matrix)

if __name__ == "__main__":
    demo.launch()
