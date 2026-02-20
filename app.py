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
# Reduce TensorFlow runtime noise on CPU hosts.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
# ❌ Removed: TF_USE_LEGACY_KERAS (causes Keras3 conflict)
if os.environ.get("TF_USE_LEGACY_KERAS") == "1":
    os.environ["TF_USE_LEGACY_KERAS"] = "0"
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras  # ✅ Use standalone keras (v3)
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from src.model import AttentionLayer

ROOT = Path(__file__).resolve().parent


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
LOW_CONF_WARN_THRESHOLD = float(os.getenv("LOW_CONF_WARN_THRESHOLD", "40"))
UNCERTAIN_THRESHOLD = float(os.getenv("UNCERTAIN_THRESHOLD", "35"))
EXAMPLE_TEXTS = [
    ["I am incredibly happy and blessed, life is wonderful."],
    ["I love this so much, it makes me so happy and excited."],
    ["I am so depressed and heartbroken, everything feels so sad."],
    ["This makes me so angry, I hate it."],
    ["I am very grateful and genuinely joyful today."],
    ["This is amazing and I feel so thankful and excited."],
    ["The update is okay, nothing special, just normal."],
    ["I am so depressed and heartbroken, everything feels so sad."],
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
    "joy": "[JOY]",
    "anger": "[ANGER]",
    "sadness": "[SADNESS]",
    "neutral": "[NEUTRAL]",
    "surprise": "[SURPRISE]",
    "disgust": "[DISGUST]",
    "fear": "[FEAR]",
}

UI_CSS = """
.gradio-container { font-size: 15px; }
.result-card {
  border: 1px solid #dbe3ea;
  border-radius: 10px;
  padding: 14px;
  background: #ffffff;
}
.result-title { font-size: 13px; color: #4b5563; margin-bottom: 6px; }
.result-main { font-size: 26px; font-weight: 700; line-height: 1.2; }
.result-meta { margin-top: 8px; color: #6b7280; font-size: 13px; }
.result-band {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  margin-top: 6px;
  background: #eef2f7;
  color: #374151;
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
    return (
        f"<div style='font-size:14px'>"
        f"<b>Confidence Band:</b> {confidence_band(confidence)}<br>"
        f"<b>Primary vs Next:</b> {emotion.title()} ({round(confidence,2)}%)"
        + (f" vs {alt}<br>" if alt else "<br>")
        + f"<b>Top Attention Tokens:</b> {html.escape(top_tokens)}"
        "<br><span style='color:#6b7280'>Confidence bands: High (>=90), Moderate (70-89), Low (50-69), Very Low (35-49), Uncertain (<35).</span>"
        f"</div>"
    )


def build_primary_result_card(emotion: str, confidence: float, dt_ms: float) -> str:
    e = emotion.lower()
    color = EMOTION_COLORS.get(e, "#111827")
    icon = EMOTION_ICONS.get(e, "[EMOTION]")
    band = confidence_band(confidence)
    width = min(100.0, max(2.0, float(confidence)))
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
        f"<div class='result-meta'>Confidence: <b>{confidence:.2f}%</b></div>"
        f"<div style='height:9px;background:#e5e7eb;border-radius:999px;margin-top:8px;'>"
        f"<div style='height:9px;width:{width:.2f}%;background:{color};border-radius:999px;'></div>"
        "</div>"
        f"<div class='result-band'>{band}</div>"
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
        rows.append(
            f"<div style='margin:6px 0;'>"
            f"<span style='display:inline-block;width:20px;color:#6b7280;'>{i}.</span> "
            f"<span style='color:{color};font-weight:700'>{emo.title()}</span> "
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
    warn = ""
    if chars > MAX_INPUT_CHARS:
        warn = f"<span style='color:#b91c1c'>Input exceeds {MAX_INPUT_CHARS} characters.</span>"
    elif toks > MAX_LEN:
        warn = f"<span style='color:#b45309'>Input exceeds {MAX_LEN} tokens; it will be truncated.</span>"
    else:
        warn = "<span style='color:#6b7280'>Input length is within model limits.</span>"
    return f"<div style='font-size:13px'>Chars: <b>{chars}</b> | Tokens: <b>{toks}</b> | {warn}</div>"


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
        return msg, "", "", "", pd.DataFrame(columns=["token", "attention_weight"]), pd.DataFrame(columns=["emotion", "probability_%"]), None

    if not text or not text.strip():
        return "Please enter some text.", "", "", "", pd.DataFrame(columns=["token", "attention_weight"]), pd.DataFrame(columns=["emotion", "probability_%"]), None

    if len(text) > MAX_INPUT_CHARS:
        return f"Please keep input under {MAX_INPUT_CHARS} characters.", "", "", "", pd.DataFrame(columns=["token", "attention_weight"]), pd.DataFrame(columns=["emotion", "probability_%"]), None

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
    return result, rationale_html, alternatives_html, heatmap, attn_df.head(12), prob_df, conf_fig


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
    for repo_id, repo_type in repo_specs():
        try:
            fp = hf_hub_download(
                repo_id=repo_id,
                filename="training_history.json",
                repo_type=repo_type,
            )
            return json.loads(Path(fp).read_text(encoding="utf-8"))
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


def analyze_batch(file_obj):
    if INIT_ERROR:
        return pd.DataFrame({"error": [INIT_ERROR]}), None, None, None
    if file_obj is None:
        return pd.DataFrame(columns=["text", "prediction", "confidence_%"]), None, None, None

    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
    path = Path(file_path)
    if not path.exists():
        return pd.DataFrame({"error": ["Uploaded file not found."]}), None, None, None

    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
            text_col = "text" if "text" in df.columns else df.columns[0]
            texts = df[text_col].astype(str).tolist()
        else:
            texts = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    except Exception as e:
        return pd.DataFrame({"error": [f"Failed to read file: {e}"]}), None, None, None

    rows = []
    label_counts = {}
    for t in texts:
        normalized = normalize_social_text(t)
        model_inputs = _build_model_inputs([normalized], model, tokenizer, MAX_LEN)
        pred = model(model_inputs, training=False).numpy()[0]
        pred = apply_emotion_cue_adjustment(normalized, pred, label_encoder.classes_)
        pred_id = int(np.argmax(pred))
        lbl = label_encoder.inverse_transform([pred_id])[0]
        conf = float(np.max(pred)) * 100.0
        rows.append({"text": t, "prediction": lbl, "confidence_%": round(conf, 2)})
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

    out_df = pd.DataFrame(rows)
    if not label_counts:
        csv_path, json_path = write_batch_exports(out_df)
        return out_df, None, csv_path, json_path

    labels = list(label_counts.keys())
    values = [label_counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    colors = [EMOTION_COLORS.get(lbl.lower(), "#6b7280") for lbl in labels]
    ax.bar(labels, values, color=colors, alpha=0.9)
    ax.set_title("Batch Emotion Distribution")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    csv_path, json_path = write_batch_exports(out_df)
    return out_df, fig, csv_path, json_path


# =====================================================
# UI
# =====================================================

SAMPLE_BATCH_FILE = ensure_sample_batch_file()

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=UI_CSS) as demo:
    gr.Markdown("## Emotion Classification in Social Media Using Attention-Based BiLSTM")
    with gr.Accordion("Model Diagnostics & Documentation", open=False):
        class_metrics_df = load_class_metrics_table()
        gr.Markdown(build_model_info_md())
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
        gr.Examples(
            examples=EXAMPLE_TEXTS,
            inputs=[text_input],
            label="Quick Examples",
        )
        with gr.Row():
            predict_btn = gr.Button("Submit")
            clear_btn = gr.Button("Clear")
        with gr.Row():
            with gr.Column(scale=1):
                pred_html = gr.HTML(label="Primary Emotion")
                rationale_html = gr.HTML(label="Prediction Rationale")
                alternatives_html = gr.HTML(label="Alternative Emotions")
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
            outputs=[pred_html, rationale_html, alternatives_html, heatmap_html, attn_table, prob_table, conf_plot],
        )
        text_input.change(
            fn=input_feedback,
            inputs=[text_input],
            outputs=[input_stats_html],
        )
        clear_btn.click(
            lambda: ("", "", "", "", pd.DataFrame(columns=["token", "attention_weight"]), pd.DataFrame(columns=["emotion", "probability_%"]), None, input_feedback("")),
            outputs=[pred_html, rationale_html, alternatives_html, heatmap_html, attn_table, prob_table, conf_plot, input_stats_html],
        )

    with gr.Tab("Batch Analysis"):
        gr.Markdown("Download a sample CSV to test batch mode quickly.")
        gr.DownloadButton("Download Sample Batch CSV", value=SAMPLE_BATCH_FILE)
        batch_file = gr.File(
            label="Upload .csv (text column) or .txt (one text per line)",
            file_types=[".csv", ".txt"],
        )
        batch_btn = gr.Button("Run Batch Analysis")
        batch_table = gr.Dataframe(label="Batch Predictions", interactive=False)
        batch_plot = gr.Plot(label="Batch Distribution")
        batch_csv_file = gr.File(label="Batch CSV Export", interactive=False)
        batch_json_file = gr.File(label="Batch JSON Export", interactive=False)
        batch_btn.click(
            fn=analyze_batch,
            inputs=[batch_file],
            outputs=[batch_table, batch_plot, batch_csv_file, batch_json_file],
        )

    gr.Markdown("### Training Performance")
    train_curves_plot = gr.Plot(
        label="Model Accuracy and Model Loss",
        value=build_training_curves_plot(),
    )

if __name__ == "__main__":
    demo.launch(ssr_mode=False)
