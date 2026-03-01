import json
import os
import time
import tempfile
import zipfile
import h5py
import html
import re
import ast
from pathlib import Path
from typing import Tuple

# Disable Spaces hot-reload watcher (can crash on some runtimes)
os.environ.setdefault("SPACES_DISABLE_RELOAD", "1")
os.environ.setdefault("GRADIO_WATCHFN_SPACES", "0")
# Reduce TensorFlow runtime noise on CPU hosts.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
# ? Removed: TF_USE_LEGACY_KERAS (causes Keras3 conflict)
if os.environ.get("TF_USE_LEGACY_KERAS") == "1":
    os.environ["TF_USE_LEGACY_KERAS"] = "0"
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras  # ? Use standalone keras (v3)
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from src.model import AttentionLayer

ROOT = Path(__file__).resolve().parent


# =====================================================
# CONFIG
# =====================================================

MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "SurajAI2025/emotion-model-7")
MODEL_REPO_TYPE = os.getenv("MODEL_REPO_TYPE", "model")
MODEL_REPO_FALLBACK_ID = os.getenv("MODEL_REPO_FALLBACK_ID", "SurajAI2025/Emotion")
MODEL_REPO_FALLBACK_TYPE = os.getenv("MODEL_REPO_FALLBACK_TYPE", "space")
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "emotion_model_final.keras")
MODEL_LOCAL_DIR = os.getenv("MODEL_LOCAL_DIR", "emotion-model")
TOKENIZER_PATH_OVERRIDE = os.getenv("TOKENIZER_PATH", "").strip()
LABELS_PATH_OVERRIDE = os.getenv("LABELS_PATH", "").strip()

MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "280"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
METRICS_TEXT = "Test Accuracy: 0.661 | Macro F1: 0.322 (n=4590)"
LOW_CONF_WARN_THRESHOLD = float(os.getenv("LOW_CONF_WARN_THRESHOLD", "40"))
UNCERTAIN_THRESHOLD = float(os.getenv("UNCERTAIN_THRESHOLD", "35"))
FULL_EXAMPLE_TEXTS = [
    ["I am incredibly happy and blessed, life is wonderful."],
    ["I love this so much, it makes me so happy and excited."],
    ["I am so depressed and heartbroken, everything feels so sad."],
    ["This makes me so angry, I hate it."],
    ["I am very grateful and genuinely joyful today."],
    ["This is amazing and I feel so thankful and excited."],
    ["The update is okay, nothing special, just normal."],
    ["I am so depressed and heartbroken, everything feels so sad."],
]
DEMO_EXAMPLE_TEXTS = [
    ["I am incredibly happy and blessed, life is wonderful."],
    ["I love this so much, it makes me so happy and excited."],
    ["I am very grateful and genuinely joyful today."],
    ["This is amazing and I feel so thankful and excited."],
]

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
    "admiration": "#4CAF50",
    "amusement": "#9C27B0",
    "anger": "#F44336",
    "annoyance": "#F44336",
    "approval": "#4CAF50",
    "caring": "#4CAF50",
    "confusion": "#757575",
    "curiosity": "#2196F3",
    "desire": "#9C27B0",
    "disappointment": "#2196F3",
    "disapproval": "#F44336",
    "disgust": "#8B7355",
    "embarrassment": "#757575",
    "excitement": "#4CAF50",
    "fear": "#FF9800",
    "gratitude": "#4CAF50",
    "grief": "#2196F3",
    "joy": "#4CAF50",
    "love": "#4CAF50",
    "nervousness": "#FF9800",
    "optimism": "#4CAF50",
    "pride": "#4CAF50",
    "realization": "#757575",
    "relief": "#4CAF50",
    "remorse": "#2196F3",
    "sadness": "#2196F3",
    "surprise": "#9C27B0",
    "neutral": "#757575",
}

EMOTION_ICONS = {
    "joy": "&#128522;",
    "anger": "&#128544;",
    "sadness": "&#128546;",
    "neutral": "&#128528;",
    "surprise": "&#128562;",
    "disgust": "&#129314;",
    "fear": "&#128552;",
    "love": "&#10084;&#65039;",
    "gratitude": "&#128591;",
    "amusement": "&#128516;",
}

UI_CSS = """
:root {
  --primary: #2563EB;
  --success: #16A34A;
  --teal: #0D9488;
  --text-dark: #111827;
  --text-medium: #374151;
  --bg-light: #F9FAFB;
  --card-bg: #FFFFFF;
  --border: #E5E7EB;
}

body {
  background-color: var(--bg-light);
}

.gradio-container {
  background: var(--bg-light) !important;
  color: var(--text-dark) !important;
  font-size: 15px;
}

.gradio-container,
.gradio-container * {
  color: var(--text-dark) !important;
}

h2, h3 {
  color: var(--text-dark) !important;
  font-weight: 700 !important;
  margin-top: 24px !important;
  margin-bottom: 12px !important;
}

.gradio-container .block,
.gradio-container .panel,
.gradio-container .gr-box,
.gradio-container .gr-form,
.gradio-container .gr-panel,
.gradio-container .gr-accordion,
.gradio-container .gr-dataframe,
.gradio-container .gr-plot,
.gradio-container .gr-file,
.gradio-container .gr-radio,
.gradio-container .gr-dropdown,
.gradio-container .gr-textbox {
  background: var(--card-bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select,
.gradio-container option {
  background: #ffffff !important;
  color: var(--text-dark) !important;
  border-color: var(--border) !important;
}

.gradio-container .gr-textbox textarea::placeholder,
.gradio-container .gr-textbox input::placeholder {
  color: var(--text-medium) !important;
  opacity: 1 !important;
}

.gradio-container .label-wrap,
.gradio-container .label-wrap *,
.gradio-container .block-title,
.gradio-container .block-title *,
.gradio-container .block-info,
.gradio-container .block-info * {
  color: var(--text-dark) !important;
  background: transparent !important;
}

/* Buttons */
.gradio-container button,
.gradio-container .gr-button {
  border-radius: 10px !important;
  font-weight: 600 !important;
  border: 1px solid var(--border) !important;
  background: #E5E7EB !important;
  color: var(--text-dark) !important;
}

.primary-btn button,
.primary-btn .gr-button {
  background: var(--primary) !important;
  color: #ffffff !important;
  border-color: var(--primary) !important;
}

.secondary-btn button,
.secondary-btn .gr-button {
  background: #E5E7EB !important;
  color: var(--text-dark) !important;
}

/* Tabs */
.gradio-container [role="tab"] {
  background: #ffffff !important;
  color: var(--text-medium) !important;
  border: 1px solid transparent !important;
}

.gradio-container [role="tab"][aria-selected="true"] {
  color: var(--primary) !important;
  border-bottom: 2px solid var(--primary) !important;
  font-weight: 700 !important;
}

/* DataFrame + HTML table styling */
.gradio-container table,
.gradio-container .gr-dataframe table,
.gradio-container [data-testid="dataframe"] table,
.gradio-container .handsontable .htCore {
  width: 100% !important;
  border-collapse: collapse !important;
  background: #ffffff !important;
  border-radius: 12px !important;
  overflow: hidden !important;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
}

.gradio-container thead,
.gradio-container .gr-dataframe thead,
.gradio-container .handsontable thead {
  background: var(--primary) !important;
}

.gradio-container th,
.gradio-container .gr-dataframe th,
.gradio-container .handsontable th {
  background: var(--primary) !important;
  color: #ffffff !important;
  padding: 12px !important;
  font-weight: 700 !important;
  text-align: left !important;
  border: 1px solid #1E40AF !important;
}

.gradio-container td,
.gradio-container .gr-dataframe td,
.gradio-container .handsontable td {
  background: #ffffff !important;
  color: var(--text-dark) !important;
  padding: 11px !important;
  border: 1px solid var(--border) !important;
  font-weight: 500 !important;
}

.gradio-container tbody tr:nth-child(even) td,
.gradio-container .gr-dataframe tbody tr:nth-child(even) td,
.gradio-container .handsontable tbody tr:nth-child(even) td {
  background: #F3F4F6 !important;
}

.gradio-container tbody tr:hover td,
.gradio-container .gr-dataframe tbody tr:hover td,
.gradio-container .handsontable tbody tr:hover td {
  background: #EFF6FF !important;
}

/* Metric cards */
.result-card {
  border: 1px solid var(--border);
  border-left: 6px solid var(--teal);
  border-radius: 12px;
  padding: 14px;
  background: #ffffff;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.04);
}

.result-title {
  font-size: 13px;
  color: var(--text-medium) !important;
  margin-bottom: 6px;
  font-weight: 600;
}

.result-main {
  font-size: 32px;
  font-weight: 800;
  color: var(--text-dark) !important;
  line-height: 1.15;
}

.result-meta {
  margin-top: 8px;
  color: var(--text-medium) !important;
  font-size: 13px;
}

.result-band {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  margin-top: 8px;
  background: #ECFDF5;
  color: var(--success) !important;
  border: 1px solid #86EFAC;
}

.conf-meter {
  margin-top: 10px;
  border-radius: 10px;
  border: 1px solid var(--border);
  padding: 8px;
  background: #ffffff;
}

.conf-meter-track {
  height: 10px;
  background: #E5E7EB;
  border-radius: 999px;
  overflow: hidden;
}

.conf-meter-fill {
  height: 10px;
  border-radius: 999px;
  transition: width 320ms ease-out;
}

.alt-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 9px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
}

.help-chip {
  display: inline-block;
  margin-left: 6px;
  padding: 1px 6px;
  border-radius: 999px;
  background: #EFF6FF;
  border: 1px solid #BFDBFE;
  color: #1E40AF !important;
  font-size: 11px;
}
"""

os.environ["TF_DETERMINISTIC_OPS"] = "1"
np.random.seed(42)
tf.random.set_seed(42)
tf.get_logger().setLevel("ERROR")


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
    sarcasm_patterns = ["yeah right", "as if", "sure...", "/s", "lol sure", "totally great"]
    fear_words = {"terrified", "scared", "horrifying", "frightening", "panic", "afraid"}
    disgust_words = {"disgusting", "repulsive", "revolting", "gross", "nasty"}
    surprise_words = {"unexpected", "shocking", "unbelievable", "omg", "wow"}
    negative_words = {
        "garbage", "waste", "useless", "awful", "horrible", "worst", "pathetic",
        "trash", "broken", "bad", "disappointing", "refund", "scam", "terrible",
    }

    text = re.sub(r"http\\S+|www\\.\\S+", " URL ", text)
    text = re.sub(r"@\\w+", " USER ", text)
    text = re.sub(r"#(\\w+)", r"\\1", text)
    text = re.sub(r"\\s+", " ", text).strip()
    if any(p in text for p in sarcasm_patterns):
        text += " sarcasm_cue"
    if any(w in text for w in fear_words):
        text += " fear_cue"
    if any(w in text for w in disgust_words):
        text += " disgust_cue"
    if any(w in text for w in surprise_words):
        text += " surprise_cue"
    if any(w in text for w in negative_words):
        text += " anger_cue disgust_cue sadness_cue"
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
        try:
            payload = json.loads(Path(TOKENIZER_PATH_OVERRIDE).read_text(encoding="utf-8"))
            return tokenizer_from_json(json.dumps(payload))
        except Exception:
            pass

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

    return None


def _build_model_inputs(texts, model, tokenizer, max_len):
    seqs = tokenizer.texts_to_sequences(texts)
    x = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
    return x.astype("int32")


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
    seq = model_inputs[0].tolist() if isinstance(model_inputs, np.ndarray) else []
    idx_to_word = getattr(tokenizer, "index_word", {})
    tokens = [idx_to_word.get(int(i), "<OOV>") for i in seq if int(i) != 0]
    return tokens


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
            f"<span style='background:rgba(14,165,233,{alpha:.3f}); "
            f"padding:2px 4px; margin:2px; border-radius:4px; display:inline-block'>{safe_tok}</span>"
        )
    return "<div style='line-height:2'>" + "".join(parts) + "</div>"


def apply_emotion_cue_adjustment(normalized_text: str, probs: np.ndarray, classes: np.ndarray) -> np.ndarray:
    fear_words = {"terrified", "scared", "horrifying", "frightening", "panic", "afraid", "fear"}
    disgust_words = {"disgusting", "repulsive", "revolting", "gross", "nasty", "disgust"}
    surprise_words = {"unexpected", "shocking", "unbelievable", "omg", "wow", "surprising", "surprise"}
    anger_words = {"furious", "angry", "hate", "rage", "unacceptable", "outrage"}

    words = set(normalized_text.split())
    boost = np.ones_like(probs, dtype=np.float32)
    class_to_idx = {str(c).lower(): i for i, c in enumerate(classes.tolist())}

    def _boost(label: str, value: float):
        idx = class_to_idx.get(label)
        if idx is not None:
            boost[idx] = max(boost[idx], value)

    if words & fear_words:
        _boost("fear", 1.35)
    if words & disgust_words:
        _boost("disgust", 1.35)
    if words & surprise_words:
        _boost("surprise", 1.30)
    if words & anger_words:
        _boost("anger", 1.25)

    adjusted = probs.astype(np.float32) * boost
    adjusted_sum = float(np.sum(adjusted))
    if adjusted_sum <= 0:
        return probs
    return adjusted / adjusted_sum


def confidence_band(conf_pct: float) -> str:
    if conf_pct < UNCERTAIN_THRESHOLD:
        return "Uncertain"
    if conf_pct >= 90:
        return "High"
    if conf_pct >= 70:
        return "Moderate"
    if conf_pct >= 50:
        return "Low"
    return "Very Low"


def build_rationale_html(emotion: str, confidence: float, prob_df: pd.DataFrame, attn_df: pd.DataFrame):
    top2 = prob_df.head(2).to_dict("records")
    alt = ""
    if len(top2) > 1:
        alt = f"{top2[1]['emotion'].title()} ({top2[1]['probability_%']}%)"
    top_tokens = ", ".join(attn_df["token"].head(5).astype(str).tolist()) if len(attn_df) else "N/A"
    band = confidence_band(confidence)
    return (
        f"<div style='font-size:14px'>"
        f"<b>Confidence Band:</b> {band}<span class='help-chip' title='Band thresholds are based on top-class probability.'>?</span><br>"
        f"<b>Primary vs Next:</b> {emotion.title()} ({round(confidence,2)}%)"
        + (f" vs {alt}<br>" if alt else "<br>")
        + f"<b>Top Attention Tokens:</b> {html.escape(top_tokens)}"
        + "<span class='help-chip' title='These tokens had the highest attention weights and influenced the model most.'>?</span>"
        + "<br><span style='color:#6b7280'>Confidence bands: High (>=90), Moderate (70-89), Low (50-69), Very Low (35-49), Uncertain (<35).</span>"
        + f"</div>"
    )


def build_primary_result_card(emotion: str, confidence: float, dt_ms: float) -> str:
    e = emotion.lower()
    color = EMOTION_COLORS.get(e, "#111827")
    icon = EMOTION_ICONS.get(e, "&#128578;")
    band = confidence_band(confidence)
    width = min(100.0, max(2.0, float(confidence)))
    intensity = "Strong" if confidence >= 80 else "Moderate" if confidence >= 60 else "Weak"
    band_bg = "#dcfce7" if band == "High" else "#fef3c7" if band in {"Moderate", "Low"} else "#fee2e2"
    band_fg = "#166534" if band == "High" else "#92400e" if band in {"Moderate", "Low"} else "#991b1b"

    warning_html = ""
    if confidence < LOW_CONF_WARN_THRESHOLD:
        warning_html = (
            "<div style='margin-top:10px;padding:8px;border-radius:8px;"
            "background:#fff7ed;color:#9a3412;font-size:13px;'>"
            "Low-confidence prediction. Consider the alternative emotions below."
            "</div>"
        )

    return (
        "<div class='result-card'>"
        "<div class='result-title'>Primary Prediction</div>"
        f"<div class='result-main' style='color:{color}'>{icon} {emotion.title()}</div>"
        f"<div class='result-meta'>Confidence: <b>{confidence:.2f}%</b> | Intensity: <b>{intensity}</b></div>"
        "<div class='conf-meter'>"
        f"<div style='font-size:12px;color:#475569;margin-bottom:6px;'>Confidence Meter</div>"
        "<div class='conf-meter-track'>"
        f"<div class='conf-meter-fill' style='width:{width:.2f}%;background:{color};'></div>"
        "</div>"
        "</div>"
        f"<div class='result-band' style='background:{band_bg};color:{band_fg};'>{band}</div>"
        f"<div class='result-meta'>Inference: {dt_ms:.1f} ms</div>"
        f"{warning_html}"
        "</div>"
    )


def build_alternative_emotions_html(prob_df: pd.DataFrame) -> str:
    top3 = prob_df.head(3).to_dict("records")
    if not top3:
        return ""
    rows = []
    for i, row in enumerate(top3, start=1):
        emo = str(row["emotion"]).lower()
        color = EMOTION_COLORS.get(emo, "#6b7280")
        icon = EMOTION_ICONS.get(emo, "&#128578;")
        rows.append(
            f"<div style='margin:6px 0;'>"
            f"<span style='display:inline-block;width:20px;color:#6b7280;'>{i}.</span> "
            f"<span class='alt-chip' style='background:{color}1a;color:{color};border:1px solid {color}55;'>"
            f"{icon} {emo.title()}</span> "
            f"<span style='color:#374151'>({row['probability_%']}%)</span></div>"
        )
    return (
        "<div class='result-card'>"
        "<div class='result-title'>Alternative Emotions (Top 3)</div>"
        + "".join(rows)
        + "</div>"
    )


def estimate_token_count(text: str) -> int:
    txt = (text or "").strip()
    if not txt:
        return 0
    normalized = normalize_social_text(txt)
    try:
        if hasattr(tokenizer, "texts_to_sequences"):
            return len(tokenizer.texts_to_sequences([normalized])[0])
        enc = tokenizer(
            [normalized],
            truncation=False,
            padding=False,
            return_tensors=None,
        )
        ids = enc.get("input_ids", [[]])[0]
        return len(ids)
    except Exception:
        return len(normalized.split())


def input_feedback(text: str) -> str:
    txt = text or ""
    chars = len(txt)
    toks = estimate_token_count(txt)
    char_color = "#059669"
    if chars > int(MAX_INPUT_CHARS * 0.85):
        char_color = "#b45309"
    if chars > MAX_INPUT_CHARS:
        char_color = "#b91c1c"
    warn = ""
    if chars > MAX_INPUT_CHARS:
        warn = f"<span style='color:#b91c1c'>Input exceeds {MAX_INPUT_CHARS} characters.</span>"
    elif toks > MAX_LEN:
        warn = f"<span style='color:#b45309'>Input exceeds {MAX_LEN} tokens; it will be truncated.</span>"
    else:
        warn = "<span style='color:#6b7280'>Input length is within model limits.</span>"
    return f"<div style='font-size:13px'>Chars: <b style='color:{char_color}'>{chars}</b> | Tokens: <b>{toks}</b> | {warn}</div>"


def ensure_sample_batch_file() -> str:
    out_path = ROOT / "research" / "outputs" / "sample_batch_input.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        sample = pd.DataFrame(
            {
                "text": [
                    "I am so happy and excited about this wonderful day!",
                    "I hate this so much and I am furious.",
                    "I feel really sad and heartbroken today.",
                    "This was unexpected and shocking wow!",
                    "The update is fine, nothing special.",
                ]
            }
        )
        sample.to_csv(out_path, index=False)
    return str(out_path)


def compute_dataset_stats_md():
    stats = []
    for name, fp in [("train", ROOT / "data_train.csv"), ("validation", ROOT / "data_validation.csv"), ("test", ROOT / "data_test.csv")]:
        if fp.exists():
            try:
                n = sum(1 for _ in fp.open("r", encoding="utf-8", errors="ignore")) - 1
                stats.append(f"- {name}: {max(n,0)} samples")
            except Exception:
                pass
    if not stats:
        return "- dataset files not found locally"
    return "\n".join(stats)


def load_eval_metrics_md():
    candidates = [
        ROOT / "research" / "outputs" / "final_model_metrics.json",
        ROOT / "research" / "outputs" / "research_results.json",
    ]
    payload = None
    for p in candidates:
        if p.exists():
            try:
                payload = json.loads(p.read_text(encoding="utf-8"))
                break
            except Exception:
                continue
    if not payload:
        return f"- Runtime Metrics: {METRICS_TEXT}"

    acc = payload.get("accuracy")
    mp = payload.get("macro_precision")
    mr = payload.get("macro_recall")
    mf1 = payload.get("macro_f1")
    if acc is None:
        return f"- Runtime Metrics: {METRICS_TEXT}"

    parts = [f"- Accuracy: `{float(acc):.4f}`"]
    if mp is not None:
        parts.append(f"- Macro Precision: `{float(mp):.4f}`")
    if mr is not None:
        parts.append(f"- Macro Recall: `{float(mr):.4f}`")
    if mf1 is not None:
        parts.append(f"- Macro F1: `{float(mf1):.4f}`")
    return "\n".join(parts)


def load_class_metrics_table():
    p = ROOT / "research" / "outputs" / "final_model_metrics.json"
    if not p.exists():
        return pd.DataFrame(columns=["emotion", "precision", "recall", "f1_score", "support"])
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        report = payload.get("report", {})
        rows = []
        for cls in payload.get("labels", []):
            m = report.get(cls, {})
            rows.append(
                {
                    "emotion": cls,
                    "precision": round(float(m.get("precision", 0.0)), 4),
                    "recall": round(float(m.get("recall", 0.0)), 4),
                    "f1_score": round(float(m.get("f1-score", 0.0)), 4),
                    "support": int(float(m.get("support", 0.0))),
                }
            )
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame(columns=["emotion", "precision", "recall", "f1_score", "support"])


def get_example_choices(mode: str):
    src = DEMO_EXAMPLE_TEXTS if mode == "Demo Mode" else FULL_EXAMPLE_TEXTS
    return [row[0] for row in src]


def update_example_choices(mode: str):
    choices = get_example_choices(mode)
    value = choices[0] if choices else None
    return gr.update(choices=choices, value=value)


def load_selected_example(example_text: str):
    return example_text or ""


def build_benchmark_summary_md():
    p = ROOT / "research" / "outputs" / "final_model_metrics.json"
    if not p.exists():
        return "### Benchmark Summary\n- Metrics file not found."
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        labels = payload.get("labels", [])
        cm = np.array(payload.get("confusion_matrix", []), dtype=np.int32)
        report = payload.get("report", {})
        acc = float(payload.get("accuracy", 0.0))
        macro_f1 = float(payload.get("macro_f1", 0.0))

        weakest = []
        for lbl in labels:
            f1 = float(report.get(lbl, {}).get("f1-score", 0.0))
            weakest.append((lbl, f1))
        weakest.sort(key=lambda x: x[1])
        weakest_text = ", ".join([f"{k} ({v:.3f})" for k, v in weakest[:3]])

        conf_pairs = []
        if cm.ndim == 2 and cm.size > 0 and len(labels) == cm.shape[0]:
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    if i == j:
                        continue
                    conf_pairs.append((int(cm[i, j]), labels[i], labels[j]))
            conf_pairs.sort(reverse=True)
        top_conf = conf_pairs[:3]
        conf_text = ", ".join([f"{a}->{b} ({n})" for n, a, b in top_conf]) if top_conf else "N/A"

        return (
            "### Benchmark Summary\n"
            f"- 7-class Accuracy: `{acc:.4f}`\n"
            f"- Macro F1: `{macro_f1:.4f}`\n"
            f"- Weakest classes by F1: `{weakest_text}`\n"
            f"- Top confusion pairs: `{conf_text}`\n"
        )
    except Exception:
        return "### Benchmark Summary\n- Could not parse benchmark metrics."


def build_class_imbalance_plot(metrics_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    if metrics_df is None or metrics_df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "Class support not available.", ha="center", va="center")
        fig.tight_layout()
        return fig

    df = metrics_df.copy()
    df = df.sort_values("support", ascending=False).reset_index(drop=True)
    colors = [EMOTION_COLORS.get(str(e).lower(), "#6b7280") for e in df["emotion"].tolist()]
    ax.bar(df["emotion"].tolist(), df["support"].astype(int).tolist(), color=colors, alpha=0.9)
    ax.set_title("Validation Class Support (Imbalance View)")
    ax.set_ylabel("Support")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def build_improvement_summary_md(metrics_df: pd.DataFrame) -> str:
    if metrics_df is None or metrics_df.empty:
        return "- Per-class metrics not found."
    ordered = metrics_df.sort_values("f1_score", ascending=False).reset_index(drop=True)
    best = ordered.iloc[0]
    worst = ordered.iloc[-1]
    low = metrics_df[metrics_df["f1_score"] < 0.45].sort_values("f1_score")
    low_classes = ", ".join(low["emotion"].astype(str).tolist()) if len(low) else "None"
    imbalance_ratio = float(metrics_df["support"].max()) / max(float(metrics_df["support"].min()), 1.0)
    return (
        "### Performance Insights\n"
        f"- Best class: `{best['emotion']}` (F1 `{best['f1_score']:.4f}`)\n"
        f"- Lowest class: `{worst['emotion']}` (F1 `{worst['f1_score']:.4f}`)\n"
        f"- Priority classes (F1 < 0.45): `{low_classes}`\n"
        f"- Class imbalance ratio (max/min support): `{imbalance_ratio:.2f}x`\n"
    )


def build_model_info_md():
    if model is None:
        return "Model unavailable."
    input_desc = model.input_shape
    try:
        params = f"{model.count_params():,}"
    except Exception:
        params = "N/A"
    classes = ", ".join(list(label_encoder.classes_)) if label_encoder is not None else "N/A"
    return (
        "### Model Information\n"
        f"- Repo: `{MODEL_REPO_ID}` (`{MODEL_REPO_TYPE}`)\n"
        f"- Fallback Repo: `{MODEL_REPO_FALLBACK_ID}` (`{MODEL_REPO_FALLBACK_TYPE}`)\n"
        f"- Input Shape: `{input_desc}`\n"
        f"- Parameters: `{params}`\n"
        f"- Classes: {classes}\n"
        f"{load_eval_metrics_md()}\n\n"
        "### Architecture (Current)\n"
        "- Embedding -> BiLSTM -> Attention -> Dense classifier\n"
        "- Attention is visualized at token level in predictions\n\n"
        "### Dataset Snapshot\n"
        f"{compute_dataset_stats_md()}"
    )


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
    pred = apply_emotion_cue_adjustment(normalized, pred, label_encoder.classes_)
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


def build_confidence_plot(labels, probs):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    probs_pct = np.array(probs, dtype=np.float32) * 100.0
    colors = [EMOTION_COLORS.get(lbl.lower(), "#6b7280") for lbl in labels]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, probs_pct, color=colors, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([l.title() for l in labels], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Emotion Probability Distribution")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    fig.tight_layout()
    return fig


def predict_full(text):
    if INIT_ERROR:
        msg = "Setup Error:\n" + INIT_ERROR
        return msg, "", "", "", pd.DataFrame(columns=["token", "attention_weight"]), pd.DataFrame(columns=["emotion", "probability_%"]), None, ""

    if not text or not text.strip():
        return "Please enter some text.", "", "", "", pd.DataFrame(columns=["token", "attention_weight"]), pd.DataFrame(columns=["emotion", "probability_%"]), None, ""

    if len(text) > MAX_INPUT_CHARS:
        return f"Please keep input under {MAX_INPUT_CHARS} characters.", "", "", "", pd.DataFrame(columns=["token", "attention_weight"]), pd.DataFrame(columns=["emotion", "probability_%"]), None, ""

    normalized = normalize_social_text(text.strip())
    token_len = estimate_token_count(text.strip())
    model_inputs = _build_model_inputs([normalized], model, tokenizer, MAX_LEN)

    t0 = time.perf_counter()
    pred = model(model_inputs, training=False).numpy()[0]
    pred = apply_emotion_cue_adjustment(normalized, pred, label_encoder.classes_)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    pred_id = int(np.argmax(pred))
    emotion = label_encoder.inverse_transform([pred_id])[0]
    confidence = float(np.max(pred)) * 100
    result = build_primary_result_card(emotion, confidence, dt_ms)

    heatmap = "<div style='color:#6b7280'>Attention layer unavailable for this model.</div>"
    attn_df = pd.DataFrame(columns=["token", "attention_weight"])
    if ATTN_AVAILABLE:
        try:
            weights = compute_attention_weights(model_inputs)
            tokens = extract_tokens_for_attention(normalized, model_inputs, tokenizer)
            heatmap = build_attention_heatmap_html(tokens, weights)
            n = min(len(tokens), len(weights))
            attn_df = pd.DataFrame(
                {
                    "token": tokens[:n],
                    "attention_weight": np.round(np.array(weights[:n], dtype=np.float32), 4),
                }
            ).sort_values("attention_weight", ascending=False)
        except Exception:
            heatmap = "<div style='color:#6b7280'>Attention could not be rendered for this input.</div>"

    labels = list(label_encoder.classes_)
    prob_df = pd.DataFrame(
        {"emotion": labels, "probability_%": np.round(pred * 100.0, 2)}
    ).sort_values("probability_%", ascending=False)
    alternatives_html = build_alternative_emotions_html(prob_df)
    rationale_html = build_rationale_html(emotion, confidence, prob_df, attn_df)
    if token_len > MAX_LEN:
        rationale_html += (
            f"<div style='margin-top:8px;color:#92400e;font-size:13px'>"
            f"Note: input has {token_len} tokens; model uses first {MAX_LEN} tokens."
            f"</div>"
        )
    conf_fig = build_confidence_plot(labels, pred)
    copy_summary = f"Primary Emotion: {emotion.title()} ({confidence:.2f}%) | Confidence Band: {confidence_band(confidence)} | Inference: {dt_ms:.1f} ms"
    return result, rationale_html, alternatives_html, heatmap, attn_df.head(12), prob_df, conf_fig, copy_summary


def write_batch_exports(out_df: pd.DataFrame):
    if out_df is None or out_df.empty:
        return None, None
    out_dir = ROOT / "research" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"batch_predictions_{stamp}.csv"
    json_path = out_dir / f"batch_predictions_{stamp}.json"
    out_df.to_csv(csv_path, index=False)
    out_df.to_json(json_path, orient="records", indent=2)
    return str(csv_path), str(json_path)


def load_training_history():
    for repo_id, repo_type in repo_specs():
        try:
            fp = hf_hub_download(
                repo_id=repo_id,
                filename="training_history.json",
                repo_type=repo_type,
                revision="main",
                force_download=True,
                local_files_only=False,
            )
            return json.loads(Path(fp).read_text(encoding="utf-8"))
        except Exception:
            continue

    candidates = [
        ROOT / "training_history.json",
        ROOT / "emotion-model" / "training_history.json",
        ROOT / "research" / "outputs" / "training_history.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
    return None


def build_training_curves_plot():
    history = load_training_history()
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))

    if not history:
        for a in ax:
            a.axis("off")
            a.text(0.5, 0.5, "Training history not found.\nRetrain once to generate graphs.", ha="center", va="center")
        fig.tight_layout()
        return fig

    acc = history.get("accuracy", [])
    val_acc = history.get("val_accuracy", [])
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    n = max(len(acc), len(loss), len(val_acc), len(val_loss))
    epochs = list(range(1, n + 1))

    if acc:
        ax[0].plot(range(1, len(acc) + 1), acc, marker="o", label="Train")
    if val_acc:
        ax[0].plot(range(1, len(val_acc) + 1), val_acc, marker="o", label="Validation")
    ax[0].set_title("Model Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].grid(alpha=0.25)
    ax[0].legend()

    if loss:
        ax[1].plot(range(1, len(loss) + 1), loss, marker="o", label="Train")
    if val_loss:
        ax[1].plot(range(1, len(val_loss) + 1), val_loss, marker="o", label="Validation")
    ax[1].set_title("Model Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].grid(alpha=0.25)
    ax[1].legend()

    fig.tight_layout()
    return fig


def analyze_batch(file_obj, progress=gr.Progress()):
    if INIT_ERROR:
        return pd.DataFrame({"error": [INIT_ERROR]}), None, None, None, "<div style='color:#b91c1c'>Batch analysis unavailable due to setup error.</div>"
    if file_obj is None:
        return pd.DataFrame(columns=["text", "prediction", "confidence_%"]), None, None, None, "<div style='color:#6b7280'>Upload a batch file to begin.</div>"

    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    path = Path(file_path)
    if not path.exists():
        return pd.DataFrame({"error": ["Uploaded file not found."]}), None, None, None, "<div style='color:#b91c1c'>Uploaded file not found.</div>"

    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            text_col = "text" if "text" in df.columns else df.columns[0]
            texts = df[text_col].astype(str).tolist()
        else:
            texts = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    except Exception as e:
        return pd.DataFrame({"error": [f"Failed to read file: {e}"]}), None, None, None, f"<div style='color:#b91c1c'>Failed to read file: {html.escape(str(e))}</div>"

    rows = []
    label_counts = {}
    timings_ms = []
    total = max(len(texts), 1)
    progress(0, desc="Starting batch inference...")
    for i, t in enumerate(texts):
        t0 = time.perf_counter()
        normalized = normalize_social_text(t)
        model_inputs = _build_model_inputs([normalized], model, tokenizer, MAX_LEN)
        pred = model(model_inputs, training=False).numpy()[0]
        pred = apply_emotion_cue_adjustment(normalized, pred, label_encoder.classes_)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        pred_id = int(np.argmax(pred))
        lbl = label_encoder.inverse_transform([pred_id])[0]
        conf = float(np.max(pred)) * 100.0
        rows.append({"text": t, "prediction": lbl, "confidence_%": round(conf, 2)})
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        timings_ms.append(dt_ms)
        progress((i + 1) / total, desc=f"Analyzed {i + 1}/{total} texts")

    out_df = pd.DataFrame(rows)
    if not label_counts:
        csv_path, json_path = write_batch_exports(out_df)
        return out_df, None, csv_path, json_path, "<div style='color:#6b7280'>No valid rows were found in this file.</div>"

    labels = list(label_counts.keys())
    values = [label_counts[k] for k in labels]
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 3.8))
    colors = [EMOTION_COLORS.get(lbl.lower(), "#6b7280") for lbl in labels]
    ax[0].bar(labels, values, color=colors, alpha=0.9)
    ax[0].set_title("Batch Emotion Distribution")
    ax[0].set_ylabel("Count")
    ax[0].tick_params(axis="x", rotation=30)
    ax[0].grid(axis="y", alpha=0.2, linestyle="--")
    conf_vals = out_df["confidence_%"].astype(float).to_numpy()
    ax[1].hist(conf_vals, bins=[0, 20, 40, 60, 80, 100], color="#2563eb", alpha=0.85, edgecolor="#ffffff")
    ax[1].set_title("Confidence Distribution")
    ax[1].set_xlabel("Confidence (%)")
    ax[1].set_ylabel("Rows")
    ax[1].set_xlim(0, 100)
    ax[1].grid(axis="y", alpha=0.2, linestyle="--")
    fig.tight_layout()

    avg_ms = float(np.mean(timings_ms)) if timings_ms else 0.0
    top_label = max(label_counts.items(), key=lambda kv: kv[1])[0]
    summary_html = (
        "<div class='result-card'>"
        "<div class='result-title'>Batch Summary</div>"
        f"<div style='font-size:14px;color:#334155'>Rows analyzed: <b>{len(out_df)}</b></div>"
        f"<div style='font-size:14px;color:#334155'>Average inference time: <b>{avg_ms:.1f} ms/text</b></div>"
        f"<div style='font-size:14px;color:#334155'>Most frequent emotion: "
        f"<b style='color:{EMOTION_COLORS.get(top_label.lower(), '#111827')}'>{top_label.title()}</b></div>"
        "</div>"
    )

    csv_path, json_path = write_batch_exports(out_df)
    return out_df, fig, csv_path, json_path, summary_html


GOEMOTIONS_28 = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise",
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


def parse_label_ids(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def pick_coarse_label(label_ids):
    mapped = []
    for idx in label_ids:
        if not isinstance(idx, (int, np.integer)):
            continue
        if idx < 0 or idx >= len(GOEMOTIONS_28):
            continue
        coarse = GO_TO_7.get(GOEMOTIONS_28[int(idx)])
        if coarse and coarse not in mapped:
            mapped.append(coarse)
    if not mapped:
        return None
    if len(mapped) == 1:
        return mapped[0]
    mapped_wo_neutral = [m for m in mapped if m != "neutral"]
    if len(mapped_wo_neutral) == 1:
        return mapped_wo_neutral[0]
    priority = ["anger", "disgust", "fear", "sadness", "surprise", "joy", "neutral"]
    for p in priority:
        if p in mapped:
            return p
    return mapped[0]


def compute_core_metrics_df():
    cache_path = ROOT / "core_classification_metrics.json"
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            return pd.DataFrame(payload)
        except Exception:
            pass

    if INIT_ERROR or model is None or tokenizer is None or label_encoder is None:
        return pd.DataFrame(
            [{"Metric": "Error", "Value": "Model initialization failed; metrics unavailable"}]
        )

    test_path = ROOT / "data_test.csv"
    if not test_path.exists():
        return pd.DataFrame([{"Metric": "Error", "Value": "data_test.csv not found"}])

    try:
        df = pd.read_csv(test_path)
        if "text" not in df.columns or "labels" not in df.columns:
            return pd.DataFrame([{"Metric": "Error", "Value": "data_test.csv missing required columns"}])

        df["labels_parsed"] = df["labels"].apply(parse_label_ids)
        df["coarse_label"] = df["labels_parsed"].apply(pick_coarse_label)
        df = df[df["coarse_label"].notna()].copy()
        df["text"] = df["text"].astype(str).map(normalize_social_text)
        df = df[df["text"].str.len() > 0]
        if df.empty:
            return pd.DataFrame([{"Metric": "Error", "Value": "No valid evaluation samples"}])

        y_true_names = df["coarse_label"].tolist()
        texts = df["text"].tolist()
        x = _build_model_inputs(texts, model, tokenizer, MAX_LEN)
        probs = model.predict(x, batch_size=128, verbose=0)
        if probs.ndim != 2:
            return pd.DataFrame([{"Metric": "Error", "Value": "Unexpected prediction output shape"}])

        adjusted_probs = []
        for t, p in zip(texts, probs):
            adjusted_probs.append(apply_emotion_cue_adjustment(t, p, label_encoder.classes_))
        adjusted_probs = np.asarray(adjusted_probs, dtype=np.float32)

        y_pred_idx = np.argmax(adjusted_probs, axis=1)
        y_pred_names = label_encoder.inverse_transform(y_pred_idx).tolist()

        y_true = np.asarray(y_true_names, dtype=object)
        y_pred = np.asarray(y_pred_names, dtype=object)

        acc = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        recall = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        support = int(y_true.shape[0])
        num_classes = int(len(label_encoder.classes_))

        metrics_df = pd.DataFrame(
            [
                {"Metric": "Accuracy", "Value": round(acc, 4)},
                {"Metric": "Precision", "Value": round(precision, 4)},
                {"Metric": "Recall", "Value": round(recall, 4)},
                {"Metric": "F1-Score", "Value": round(f1, 4)},
                {"Metric": "Macro F1", "Value": round(macro_f1, 4)},
                {"Metric": "Weighted F1", "Value": round(weighted_f1, 4)},
                {"Metric": "Micro F1", "Value": round(micro_f1, 4)},
                {"Metric": "Support", "Value": support},
                {"Metric": "Emotion Classes", "Value": num_classes},
            ]
        )
        try:
            cache_path.write_text(metrics_df.to_json(orient="records", indent=2), encoding="utf-8")
        except Exception:
            pass
        return metrics_df
    except Exception as exc:
        return pd.DataFrame([{"Metric": "Error", "Value": f"Failed to compute metrics: {exc}"}])


def compute_showcase_metrics_df():
    core = compute_core_metrics_df()
    if core is None or core.empty:
        return pd.DataFrame([{"Metric": "Error", "Value": "No metrics available"}])
    keep = {"Accuracy", "Precision", "Weighted F1", "Support", "Emotion Classes"}
    if "Metric" not in core.columns:
        return core
    filtered = core[core["Metric"].astype(str).isin(keep)].copy()
    if filtered.empty:
        return core
    ordered = ["Accuracy", "Precision", "Weighted F1", "Support", "Emotion Classes"]
    filtered["__order"] = filtered["Metric"].map({k: i for i, k in enumerate(ordered)})
    filtered = filtered.sort_values("__order").drop(columns=["__order"])
    return filtered.reset_index(drop=True)


def metric_band(score: float):
    if score >= 0.80:
        return "EXCELLENT", "#16a34a", "*****"
    if score >= 0.70:
        return "GOOD", "#2563eb", "****"
    if score >= 0.60:
        return "ACCEPTABLE", "#ca8a04", "***"
    return "DEVELOPING", "#dc2626", "**"


def metric_bar(score: float, slots: int = 16) -> str:
    s = max(0.0, min(1.0, float(score)))
    n = int(round(s * slots))
    return ("#" * n) + ("-" * (slots - n))


def build_metrics_dashboard_html(metrics_df: pd.DataFrame) -> str:
    if metrics_df is None or metrics_df.empty or "Metric" not in metrics_df.columns:
        return "<div class='result-card'><b>Metrics unavailable.</b></div>"

    value_map = {}
    for _, row in metrics_df.iterrows():
        try:
            value_map[str(row["Metric"])] = float(row["Value"])
        except Exception:
            value_map[str(row["Metric"])] = row["Value"]

    primary_keys = ["Precision", "Weighted F1"]
    info_keys = ["Support", "Emotion Classes"]

    def render_line(k):
        if k not in value_map:
            return ""
        v = value_map[k]
        if not isinstance(v, float):
            return f"<div><b>{k}:</b> {v}</div>"
        band, color, stars = metric_band(v)
        bar = metric_bar(v)
        return (
            "<div style='margin:4px 0;'>"
            f"<b>{k}:</b> {v:.4f} ({v*100:.2f}%) "
            f"<code style='color:{color};'>{bar}</code> "
            f"<span style='color:{color};font-weight:700'>{band}</span> "
            f"<span>{stars}</span>"
            "</div>"
        )

    html_out = ["<div class='result-card'>"]
    html_out.append("<div class='result-title'>MODEL PERFORMANCE HIGHLIGHTS</div>")
    html_out.append("<div><b>PREDICTION RELIABILITY</b></div>")
    for k in primary_keys:
        html_out.append(render_line(k))
    html_out.append("<div style='margin-top:8px;'><b>DATASET SCALE</b></div>")
    for k in info_keys:
        html_out.append(render_line(k))
    html_out.append(
        "<div style='margin-top:10px;font-size:12px;color:#4b5563;'>"
        "<b>Interpretation:</b> Precision indicates prediction correctness. "
        "Weighted F1 measures balanced performance under class imbalance. "
        "Support and class count show evaluation scale and task complexity."
        "</div>"
    )
    html_out.append(
        "<table style='margin-top:10px;width:100%;border-collapse:collapse;font-size:12px;'>"
        "<tr style='background:#f8fafc;'><th style='text-align:left;border:1px solid #dbe3ea;padding:6px;'>Metric</th>"
        "<th style='text-align:left;border:1px solid #dbe3ea;padding:6px;'>Your Score</th>"
        "<th style='text-align:left;border:1px solid #dbe3ea;padding:6px;'>What It Means</th></tr>"
        f"<tr><td style='border:1px solid #dbe3ea;padding:6px;'>Precision</td><td style='border:1px solid #dbe3ea;padding:6px;'>{value_map.get('Precision','N/A')}</td>"
        "<td style='border:1px solid #dbe3ea;padding:6px;'>When model predicts an emotion, this is how often it is correct.</td></tr>"
        f"<tr><td style='border:1px solid #dbe3ea;padding:6px;'>Weighted F1</td><td style='border:1px solid #dbe3ea;padding:6px;'>{value_map.get('Weighted F1','N/A')}</td>"
        "<td style='border:1px solid #dbe3ea;padding:6px;'>F1 adjusted by class frequency under imbalance.</td></tr>"
        f"<tr><td style='border:1px solid #dbe3ea;padding:6px;'>Support</td><td style='border:1px solid #dbe3ea;padding:6px;'>{value_map.get('Support','N/A')}</td>"
        "<td style='border:1px solid #dbe3ea;padding:6px;'>Total evaluated samples in the held-out test split.</td></tr>"
        f"<tr><td style='border:1px solid #dbe3ea;padding:6px;'>Emotion Classes</td><td style='border:1px solid #dbe3ea;padding:6px;'>{value_map.get('Emotion Classes','N/A')}</td>"
        "<td style='border:1px solid #dbe3ea;padding:6px;'>Seven-way emotion detection with attention-based sequence modeling.</td></tr>"
        "</table>"
    )
    html_out.append("</div>")
    return "".join(html_out)


def build_metrics_bar_plot(metrics_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    if metrics_df is None or metrics_df.empty or "Metric" not in metrics_df.columns:
        ax.axis("off")
        ax.text(0.5, 0.5, "Metrics unavailable.", ha="center", va="center")
        fig.tight_layout()
        return fig

    metric_names = ["Precision", "Weighted F1"]
    rows = []
    for name in metric_names:
        m = metrics_df[metrics_df["Metric"] == name]
        if m.empty:
            continue
        try:
            rows.append((name, float(m.iloc[0]["Value"])))
        except Exception:
            continue

    if not rows:
        ax.axis("off")
        ax.text(0.5, 0.5, "Numeric metrics unavailable.", ha="center", va="center")
        fig.tight_layout()
        return fig

    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = []
    for v in vals:
        if v >= 0.80:
            colors.append("#16a34a")
        elif v >= 0.70:
            colors.append("#2563eb")
        elif v >= 0.60:
            colors.append("#ca8a04")
        else:
            colors.append("#dc2626")

    bars = ax.barh(names, vals, color=colors, alpha=0.95)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Score")
    ax.set_title("Strength Metrics")
    ax.grid(axis="x", alpha=0.2)
    for b, v in zip(bars, vals):
        ax.text(min(v + 0.01, 0.98), b.get_y() + b.get_height() / 2, f"{v:.2%}", va="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    return fig


def build_per_class_strengths_df(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame(columns=["Emotion", "Precision", "F1", "Support", "Status"])

    df = metrics_df.copy()
    required = {"emotion", "precision", "f1_score", "support"}
    if not required.issubset(set(df.columns)):
        return pd.DataFrame(columns=["Emotion", "Precision", "F1", "Support", "Status"])

    df = df.sort_values(["precision", "f1_score"], ascending=False).head(5).copy()

    def status(v: float) -> str:
        if v >= 0.80:
            return "EXCELLENT"
        if v >= 0.70:
            return "STRONG"
        if v >= 0.60:
            return "GOOD"
        return "DEVELOPING"

    return pd.DataFrame(
        {
            "Emotion": df["emotion"].astype(str).str.title(),
            "Precision": df["precision"].astype(float).round(4),
            "F1": df["f1_score"].astype(float).round(4),
            "Support": df["support"].astype(int),
            "Status": df["precision"].astype(float).map(status),
        }
    ).reset_index(drop=True)


# =====================================================
# UI
# =====================================================

SAMPLE_BATCH_FILE = ensure_sample_batch_file()

with gr.Blocks(theme=None, css=UI_CSS) as demo:
    gr.Markdown("## Emotion Classification in Social Media Using Attention-Based BiLSTM")
    with gr.Accordion("Model Diagnostics & Documentation", open=False):
        class_metrics_df = load_class_metrics_table()
        gr.Markdown(build_model_info_md())
        gr.Markdown(build_benchmark_summary_md())
        gr.Dataframe(
            value=class_metrics_df,
            label="Per-Class Validation Metrics",
            interactive=False,
        )
        gr.Markdown(build_improvement_summary_md(class_metrics_df))
        gr.Plot(
            label="Class Support Distribution",
            value=build_class_imbalance_plot(class_metrics_df),
        )

    with gr.Tab("Single Prediction"):
        gr.Markdown("### Model Predictions")
        text_input = gr.Textbox(
            lines=4,
            label="Social Media Text",
            placeholder="Enter a tweet, comment, or social media post...",
            max_lines=8,
        )
        input_stats_html = gr.HTML(value=input_feedback(""), label="Input Stats")
        example_mode = gr.Radio(
            choices=["Demo Mode", "Full Mode"],
            value="Demo Mode",
            label="Example Set",
        )
        example_selector = gr.Dropdown(
            choices=get_example_choices("Demo Mode"),
            value=get_example_choices("Demo Mode")[0],
            label="Quick Examples",
        )
        load_example_btn = gr.Button("Load Example", elem_classes="secondary-btn")
        with gr.Row():
            predict_btn = gr.Button("Submit", elem_classes="primary-btn")
            clear_btn = gr.Button("Clear", elem_classes="secondary-btn")
            copy_btn = gr.Button("Copy Result", elem_classes="secondary-btn")
        with gr.Row():
            with gr.Column(scale=1):
                pred_html = gr.HTML(label="Primary Emotion")
                rationale_html = gr.HTML(label="Prediction Rationale")
                alternatives_html = gr.HTML(label="Alternative Emotions")
                copy_summary_text = gr.Textbox(label="Copy-Ready Summary", interactive=False)
            with gr.Column(scale=1):
                conf_plot = gr.Plot(label="Confidence Chart")
                prob_table = gr.Dataframe(
                    headers=["emotion", "probability_%"],
                    label="All Emotion Probabilities",
                    interactive=False,
                )

        with gr.Accordion("Detailed Analytics", open=False):
            gr.Markdown("### Attention Analysis")
            heatmap_html = gr.HTML(label="Attention Heatmap")
            attn_table = gr.Dataframe(
                headers=["token", "attention_weight"],
                label="Top Attention Weights",
                interactive=False,
            )

        predict_btn.click(
            fn=predict_full,
            inputs=[text_input],
            outputs=[pred_html, rationale_html, alternatives_html, heatmap_html, attn_table, prob_table, conf_plot, copy_summary_text],
        )
        text_input.change(
            fn=input_feedback,
            inputs=[text_input],
            outputs=[input_stats_html],
        )
        example_mode.change(
            fn=update_example_choices,
            inputs=[example_mode],
            outputs=[example_selector],
        )
        load_example_btn.click(
            fn=load_selected_example,
            inputs=[example_selector],
            outputs=[text_input],
        )
        clear_btn.click(
            lambda: ("", "", "", "", pd.DataFrame(columns=["token", "attention_weight"]), pd.DataFrame(columns=["emotion", "probability_%"]), None, "", input_feedback("")),
            outputs=[pred_html, rationale_html, alternatives_html, heatmap_html, attn_table, prob_table, conf_plot, copy_summary_text, input_stats_html],
        )
        copy_btn.click(
            fn=lambda _: None,
            inputs=[copy_summary_text],
            outputs=[],
            js="""(summary) => {
                if (!summary || !summary.trim()) {
                    alert("No prediction to copy yet.");
                    return;
                }
                navigator.clipboard.writeText(summary);
            }""",
        )

    with gr.Tab("Batch Analysis"):
        gr.Markdown("Download a sample CSV to test batch mode quickly.")
        gr.DownloadButton("Download Sample Batch CSV", value=SAMPLE_BATCH_FILE)
        batch_file = gr.File(
            label="Upload .csv (text column) or .txt (one text per line)",
            file_types=[".csv", ".txt"],
        )
        batch_btn = gr.Button("Run Batch Analysis", elem_classes="primary-btn")
        batch_summary = gr.HTML(label="Batch Summary")
        batch_table = gr.Dataframe(label="Batch Predictions", interactive=False)
        batch_plot = gr.Plot(label="Batch Comparison Charts")
        batch_csv_file = gr.File(label="Batch CSV Export", interactive=False)
        batch_json_file = gr.File(label="Batch JSON Export", interactive=False)
        batch_btn.click(
            fn=analyze_batch,
            inputs=[batch_file],
            outputs=[batch_table, batch_plot, batch_csv_file, batch_json_file, batch_summary],
        )

    showcase_metrics_df = compute_showcase_metrics_df()

    gr.Markdown("### Training Performance")
    train_curves_plot = gr.Plot(
        label="Model Accuracy and Model Loss",
        value=build_training_curves_plot(),
    )
    gr.Dataframe(
        value=showcase_metrics_df,
        label="Model Performance Highlights",
        interactive=False,
    )
    gr.Markdown(
        "### Project Strengths\n"
        "- Attention-based BiLSTM architecture for social media emotion detection\n"
        "- Interpretable attention visualization for token-level rationale\n"
        "- Evaluated on a large held-out test set with real noisy text\n"
        "- Real-time inference interface with batch analysis support"
    )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)

