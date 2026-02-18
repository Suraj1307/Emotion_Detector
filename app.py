import json
import os
import time
import tempfile
import zipfile
import h5py
import html
import re
from pathlib import Path
from typing import Tuple

# Disable Spaces hot-reload watcher (can crash on some runtimes)
os.environ.setdefault("SPACES_DISABLE_RELOAD", "1")
os.environ.setdefault("GRADIO_WATCHFN_SPACES", "0")
# ❌ Removed: TF_USE_LEGACY_KERAS (causes Keras3 conflict)
if os.environ.get("TF_USE_LEGACY_KERAS") == "1":
    os.environ["TF_USE_LEGACY_KERAS"] = "0"
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import gradio as gr
import numpy as np
import tensorflow as tf
import keras  # ✅ Use standalone keras (v3)
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from transformers import AutoTokenizer

from src.model import AttentionLayer


# =====================================================
# CONFIG
# =====================================================

MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "SurajAI2025/Emotion")
MODEL_REPO_TYPE = os.getenv("MODEL_REPO_TYPE", "space")
MODEL_REPO_FALLBACK_ID = os.getenv("MODEL_REPO_FALLBACK_ID", "SurajAI2025/emotion-model-7")
MODEL_REPO_FALLBACK_TYPE = os.getenv("MODEL_REPO_FALLBACK_TYPE", "model")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "emotion_model_final.keras")
MODEL_LOCAL_DIR = os.getenv("MODEL_LOCAL_DIR", "emotion-model")
TOKENIZER_PATH_OVERRIDE = os.getenv("TOKENIZER_PATH", "").strip()
LABELS_PATH_OVERRIDE = os.getenv("LABELS_PATH", "").strip()

MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "280"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
METRICS_TEXT = "Test Accuracy: 0.661 | Macro F1: 0.322 (n=4590)"

LABEL_CANDIDATES = [
    "label_classes.json",
    "data/processed/label_classes.json",
    "label_encoder.json",
    "data/processed/label_encoder.json",
    os.path.join(MODEL_LOCAL_DIR, "label_classes.json"),
    os.path.join(MODEL_LOCAL_DIR, "label_encoder.json"),
    os.path.join(MODEL_LOCAL_DIR, "data", "processed", "label_classes.json"),
    os.path.join(MODEL_LOCAL_DIR, "data", "processed", "label_encoder.json"),
]

EMOTION_COLORS = {
    "admiration": "#4f7df3",
    "amusement": "#e6a700",
    "anger": "#e53e3e",
    "annoyance": "#d97706",
    "approval": "#16a34a",
    "caring": "#22c55e",
    "confusion": "#7c3aed",
    "curiosity": "#0ea5e9",
    "desire": "#db2777",
    "disappointment": "#b45309",
    "disapproval": "#a16207",
    "disgust": "#15803d",
    "embarrassment": "#c026d3",
    "excitement": "#ea580c",
    "fear": "#dc2626",
    "gratitude": "#10b981",
    "grief": "#7f1d1d",
    "joy": "#16a34a",
    "love": "#f43f5e",
    "nervousness": "#f59e0b",
    "optimism": "#22c55e",
    "pride": "#0f766e",
    "realization": "#2563eb",
    "relief": "#22c55e",
    "remorse": "#92400e",
    "sadness": "#2563eb",
    "surprise": "#8b5cf6",
    "neutral": "#6b7280",
}

os.environ["TF_DETERMINISTIC_OPS"] = "1"
np.random.seed(42)
tf.random.set_seed(42)


# =====================================================
# HELPERS
# =====================================================

def is_lfs_pointer(file_path: str) -> bool:
    path = Path(file_path)
    if not path.exists():
        return False
    try:
        first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
    except Exception:
        return False
    return first_line.strip() == "version https://git-lfs.github.com/spec/v1"


def normalize_social_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\.\\S+", " URL ", text)
    text = re.sub(r"@\\w+", " USER ", text)
    text = re.sub(r"#(\\w+)", r"\\1", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def repo_specs():
    specs = [(MODEL_REPO_ID, MODEL_REPO_TYPE)]
    if MODEL_REPO_FALLBACK_ID and MODEL_REPO_FALLBACK_ID != MODEL_REPO_ID:
        specs.append((MODEL_REPO_FALLBACK_ID, MODEL_REPO_FALLBACK_TYPE))
    return specs


def resolve_artifact(candidates, override_path=""):
    if override_path:
        override = Path(override_path)
        if override.exists() and not is_lfs_pointer(str(override)):
            return str(override)
    for candidate in candidates:
        local_path = Path(candidate)

        if local_path.exists() and not is_lfs_pointer(str(local_path)):
            return str(local_path)

        for repo_id, repo_type in repo_specs():
            try:
                return hf_hub_download(
                    repo_id=repo_id,
                    filename=candidate,
                    repo_type=repo_type,
                )
            except Exception:
                continue
    return None


def load_label_classes(path: str):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        if "classes" in payload:
            return payload["classes"]
        if "classes_" in payload:
            return payload["classes_"]

    raise ValueError(f"Unsupported label file format: {path}")


def _looks_like_keras_zip(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(2) == b"PK"
    except Exception:
        return False


def _sanitize_keras_config(value):
    if isinstance(value, dict):
        value.pop("quantization_config", None)
        value.pop("shared_object_id", None)
        for k in list(value.keys()):
            _sanitize_keras_config(value[k])
    elif isinstance(value, list):
        for item in value:
            _sanitize_keras_config(item)


def _sanitize_keras_archive(src_path: str):
    src = Path(src_path)
    if not src.exists() or not _looks_like_keras_zip(src):
        return None
    try:
        with zipfile.ZipFile(src, "r") as zin:
            if "config.json" not in zin.namelist():
                return None
            config = json.loads(zin.read("config.json").decode("utf-8"))
            _sanitize_keras_config(config)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
                out_path = tmp.name
            with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
                for info in zin.infolist():
                    if info.filename == "config.json":
                        data = json.dumps(config).encode("utf-8")
                    else:
                        data = zin.read(info.filename)
                    zout.writestr(info, data)
            return out_path
    except Exception:
        return None


def _rebuild_model_from_keras_archive(archive_path: str):
    try:
        with zipfile.ZipFile(archive_path, "r") as zin:
            if "config.json" not in zin.namelist() or "model.weights.h5" not in zin.namelist():
                return None
            config = json.loads(zin.read("config.json").decode("utf-8"))
            _sanitize_keras_config(config)
            config_text = json.dumps(config)
            weights = zin.read("model.weights.h5")

        rebuilt = keras.models.model_from_json(
            config_text,
            custom_objects={"AttentionLayer": AttentionLayer},
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=".weights.h5") as tmp_w:
            tmp_w.write(weights)
            w_path = tmp_w.name
        rebuilt.load_weights(w_path)
        return rebuilt
    except Exception:
        return None


def _rebuild_model_from_weights_only(archive_path: str):
    """Last-resort fallback for heavily incompatible Keras config payloads."""
    try:
        with zipfile.ZipFile(archive_path, "r") as zin:
            if "model.weights.h5" not in zin.namelist():
                return None
            weights = zin.read("model.weights.h5")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".weights.h5") as tmp_w:
            tmp_w.write(weights)
            w_path = tmp_w.name

        with h5py.File(w_path, "r") as f:
            emb_shape = tuple(f["layers"]["embedding"]["vars"]["0"].shape)
            dense0_shape = tuple(f["layers"]["dense"]["vars"]["0"].shape)
            dense1_shape = tuple(f["layers"]["dense_1"]["vars"]["0"].shape)
            fw_rec_shape = tuple(
                f["layers"]["bidirectional"]["forward_layer"]["cell"]["vars"]["1"].shape
            )
            lstm_units = int(fw_rec_shape[0])

        vocab_size, emb_dim = int(emb_shape[0]), int(emb_shape[1])
        dense_units = int(dense0_shape[1])
        num_classes = int(dense1_shape[1])
        max_len = int(os.getenv("MAX_LEN", "50"))

        inputs = keras.Input(shape=(max_len,), name="input_layer")
        x = keras.layers.Embedding(vocab_size, emb_dim, name="embedding")(inputs)
        x = keras.layers.Bidirectional(
            keras.layers.LSTM(lstm_units, return_sequences=True),
            name="bidirectional",
        )(x)
        x = AttentionLayer(name="attention_layer")(x)
        x = keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
        x = keras.layers.Dropout(0.3, name="dropout")(x)
        outputs = keras.layers.Dense(num_classes, activation="softmax", name="dense_1")(x)
        rebuilt = keras.Model(inputs=inputs, outputs=outputs, name="functional")
        rebuilt.load_weights(w_path)
        return rebuilt
    except Exception:
        return None


def _load_model_compat(model_path: str):
    try:
        return keras.models.load_model(
            model_path,
            custom_objects={"AttentionLayer": AttentionLayer},
            compile=False,
        )
    except Exception:
        pass

    sanitized = _sanitize_keras_archive(model_path)
    if sanitized:
        try:
            return keras.models.load_model(
                sanitized,
                custom_objects={"AttentionLayer": AttentionLayer},
                compile=False,
                safe_mode=False,
            )
        except Exception:
            rebuilt = _rebuild_model_from_keras_archive(sanitized)
            if rebuilt is not None:
                return rebuilt
            rebuilt_wo_cfg = _rebuild_model_from_weights_only(sanitized)
            if rebuilt_wo_cfg is not None:
                return rebuilt_wo_cfg

    rebuilt = _rebuild_model_from_keras_archive(model_path)
    if rebuilt is not None:
        return rebuilt
    return _rebuild_model_from_weights_only(model_path)


# =====================================================
# INITIALIZATION
# =====================================================

def load_tokenizer():
    if TOKENIZER_PATH_OVERRIDE and Path(TOKENIZER_PATH_OVERRIDE).exists():
        return AutoTokenizer.from_pretrained(TOKENIZER_PATH_OVERRIDE)

    tok_json = resolve_artifact(
        [
            "tokenizer.json",
            f"{MODEL_LOCAL_DIR}/tokenizer.json",
            f"{MODEL_LOCAL_DIR}/tokenizer/tokenizer.json",
            "data/processed/tokenizer.json",
        ]
    )
    if tok_json:
        try:
            payload = json.loads(Path(tok_json).read_text(encoding="utf-8"))
            return tokenizer_from_json(json.dumps(payload))
        except Exception:
            pass

    local_dir = Path(MODEL_LOCAL_DIR) / "tokenizer"
    if local_dir.exists():
        return AutoTokenizer.from_pretrained(str(local_dir))

    for repo_id, repo_type in repo_specs():
        if repo_type != "model":
            continue
        try:
            return AutoTokenizer.from_pretrained(repo_id)
        except Exception:
            continue

    return None


def _build_model_inputs(texts, model, tokenizer, max_len):
    if hasattr(tokenizer, "texts_to_sequences"):
        seqs = tokenizer.texts_to_sequences(texts)
        x = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
        return x.astype("int32")

    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="np",
    )

    if isinstance(model.input_shape, list) and len(model.input_shape) >= 2:
        return {
            "input_ids": enc["input_ids"].astype("int32"),
            "attention_mask": enc["attention_mask"].astype("int32"),
        }

    return enc["input_ids"].astype("int32")


def initialize_pipeline() -> Tuple[keras.Model, any, LabelEncoder, int]:
    print("Downloading model from:", MODEL_REPO_ID)

    local_model_path = Path(MODEL_LOCAL_DIR) / MODEL_FILENAME
    if local_model_path.exists() and not is_lfs_pointer(str(local_model_path)):
        model_path = str(local_model_path)
    else:
        model_path = resolve_artifact(
            [
                MODEL_FILENAME,
                f"{MODEL_LOCAL_DIR}/{MODEL_FILENAME}",
                f"saved_models/{MODEL_FILENAME}",
            ]
        )
        if not model_path:
            raise FileNotFoundError(
                f"Could not locate {MODEL_FILENAME} in configured repos."
            )

    model = _load_model_compat(model_path)
    if model is None:
        raise RuntimeError("Model deserialization failed after compatibility fallbacks.")

    tokenizer = load_tokenizer()
    if tokenizer is None:
        raise FileNotFoundError("Tokenizer artifact not found in configured repos.")

    labels_path = resolve_artifact(LABEL_CANDIDATES, LABELS_PATH_OVERRIDE)
    if not labels_path:
        raise FileNotFoundError("Label classes JSON not found in model repo.")

    classes = load_label_classes(labels_path)

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(classes)

    model_input_shape = model.input_shape
    if isinstance(model_input_shape, list):
        max_len = int(model_input_shape[0][1])
    else:
        max_len = int(model_input_shape[1])

    warmup_inputs = _build_model_inputs(["warmup"], model, tokenizer, max_len)
    model(warmup_inputs, training=False)

    print("Model loaded successfully.")
    return model, tokenizer, label_encoder, max_len


INIT_ERROR = None
model = None
tokenizer = None
label_encoder = None
MAX_LEN = 50

try:
    model, tokenizer, label_encoder, MAX_LEN = initialize_pipeline()
except Exception as exc:
    INIT_ERROR = str(exc)
    print("Initialization error:", INIT_ERROR)


# =====================================================
# ATTENTION + PREDICTION (UNCHANGED)
# =====================================================

ATTN_AVAILABLE = False
attention_layer = None
attention_model = None

if INIT_ERROR is None:
    try:
        attention_layer = None
        seq_layer = None

        for lyr in model.layers:
            if isinstance(lyr, AttentionLayer):
                attention_layer = lyr
                break
        if attention_layer is None:
            attention_layer = model.get_layer("attention_layer")

        for lyr in model.layers:
            if isinstance(lyr, keras.layers.Bidirectional):
                seq_layer = lyr
                break
        if seq_layer is None:
            seq_layer = model.get_layer("bilstm_layer")

        attention_model = keras.Model(model.input, seq_layer.output)
        ATTN_AVAILABLE = True
    except Exception:
        ATTN_AVAILABLE = False


def compute_attention_weights(model_inputs):
    seq = attention_model(model_inputs, training=False).numpy()[0]
    W, b, u = attention_layer.get_weights()
    uit = np.tanh(np.dot(seq, W) + b)
    ait = np.dot(uit, u).squeeze(-1)
    exp = np.exp(ait - np.max(ait))
    return exp / (np.sum(exp) + 1e-9)


def extract_tokens_for_attention(text: str, model_inputs, tokenizer):
    if hasattr(tokenizer, "texts_to_sequences"):
        seq = model_inputs[0].tolist() if isinstance(model_inputs, np.ndarray) else []
        idx_to_word = getattr(tokenizer, "index_word", {})
        tokens = [idx_to_word.get(int(i), "<OOV>") for i in seq if int(i) != 0]
        return tokens

    if isinstance(model_inputs, dict) and "input_ids" in model_inputs:
        ids = model_inputs["input_ids"][0].tolist()
        mask = model_inputs.get("attention_mask", np.ones_like(model_inputs["input_ids"]))[0]
        tokens = []
        for tid, m in zip(ids, mask.tolist()):
            if int(m) == 0:
                continue
            tok = tokenizer.convert_ids_to_tokens([int(tid)])[0]
            if tok in {"[CLS]", "[SEP]", "[PAD]"}:
                continue
            tokens.append(tok.replace("##", ""))
        return tokens

    return normalize_social_text(text).split()


def build_attention_heatmap_html(tokens, weights):
    if not tokens or len(weights) == 0:
        return "<div style='color:#6b7280'>Attention not available for this input.</div>"

    n = min(len(tokens), len(weights))
    tokens = tokens[:n]
    w = np.array(weights[:n], dtype=np.float32)
    w = w / (w.max() + 1e-9)

    parts = []
    for tok, score in zip(tokens, w.tolist()):
        alpha = 0.12 + 0.58 * float(score)
        safe_tok = html.escape(tok)
        parts.append(
            f"<span style='background:rgba(245,158,11,{alpha:.3f}); "
            f"padding:2px 4px; margin:2px; border-radius:4px; display:inline-block'>{safe_tok}</span>"
        )
    return "<div style='line-height:2'>" + "".join(parts) + "</div>"


def predict_and_explain(text):

    if INIT_ERROR:
        return "Setup Error:\n" + INIT_ERROR, ""

    if not text or not text.strip():
        return "Please enter some text.", ""

    if len(text) > MAX_INPUT_CHARS:
        return f"Please keep input under {MAX_INPUT_CHARS} characters.", ""

    normalized = normalize_social_text(text.strip())
    model_inputs = _build_model_inputs([normalized], model, tokenizer, MAX_LEN)

    t0 = time.perf_counter()
    pred = model(model_inputs, training=False).numpy()[0]
    dt_ms = (time.perf_counter() - t0) * 1000.0

    pred_id = int(np.argmax(pred))
    emotion = label_encoder.inverse_transform([pred_id])[0]
    confidence = float(np.max(pred)) * 100

    primary_color = EMOTION_COLORS.get(emotion.lower(), "#111827")

    result = (
        f"<div style='font-size:16px'>"
        f"<div><b>Primary Emotion:</b> "
        f"<span style='color:{primary_color}; font-weight:700'>{emotion.title()}</span> "
        f"({round(confidence,2)}%)</div>"
        f"<div style='margin-top:8px; color:#6b7280; font-size:12px'>"
        f"Inference: {dt_ms:.1f} ms</div>"
        f"</div>"
    )

    heatmap = "<div style='color:#6b7280'>Attention layer unavailable for this model.</div>"
    if ATTN_AVAILABLE:
        try:
            weights = compute_attention_weights(model_inputs)
            tokens = extract_tokens_for_attention(normalized, model_inputs, tokenizer)
            heatmap = build_attention_heatmap_html(tokens, weights)
        except Exception:
            heatmap = "<div style='color:#6b7280'>Attention could not be rendered for this input.</div>"

    return result, heatmap


# =====================================================
# UI (UNCHANGED)
# =====================================================

demo = gr.Interface(
    fn=predict_and_explain,
    inputs=gr.Textbox(
        lines=4,
        label="Social Media Text",
        placeholder="Type a tweet or Reddit comment...",
        max_lines=8,
    ),
    outputs=[
        gr.HTML(label="Emotion Prediction"),
        gr.HTML(label="Attention Heatmap"),
    ],
    title="Emotion Classification in Social Media Using Attention-Based BiLSTM",
    description=(
        "Classifies short, noisy social media text and highlights emotionally relevant cues "
        "using an attention-enabled BiLSTM pipeline."
    ),
)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
