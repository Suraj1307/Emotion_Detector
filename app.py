import json
import os
import tempfile
import zipfile
import h5py

# Reduce TensorFlow log noise on CPU-only hosts (e.g., HF Spaces CPU runtime).
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
# Force Keras 3 loader path to avoid tf_keras (legacy) deserialization mismatches.
if os.environ.get("TF_USE_LEGACY_KERAS") == "1":
    os.environ["TF_USE_LEGACY_KERAS"] = "0"
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import pickle
import time
from pathlib import Path

import gradio as gr
import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from src.model import AttentionLayer

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "research" / "outputs" / "research_results.json"

MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "SurajAI2025/emotion-model-7")
MODEL_REPO_TYPE = os.getenv("MODEL_REPO_TYPE", "model")
MODEL_REPO_FALLBACK = os.getenv("MODEL_REPO_FALLBACK", "SurajAI2025/Emotion")
MODEL_REPO_FALLBACK_TYPE = os.getenv("MODEL_REPO_FALLBACK_TYPE", "space")
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

EMOTION_COLORS = {
    "joy": "#FFD700",
    "sadness": "#5B7C99",
    "anger": "#DC143C",
    "fear": "#9932CC",
    "disgust": "#8B7355",
    "surprise": "#FF69B4",
    "neutral": "#808080",
}

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


def _repo_ids():
    repos = [MODEL_REPO_ID]
    if MODEL_REPO_FALLBACK and MODEL_REPO_FALLBACK not in repos:
        repos.append(MODEL_REPO_FALLBACK)
    return repos


def _repo_specs():
    specs = [(MODEL_REPO_ID, MODEL_REPO_TYPE)]
    if MODEL_REPO_FALLBACK and all(MODEL_REPO_FALLBACK != rid for rid, _ in specs):
        specs.append((MODEL_REPO_FALLBACK, MODEL_REPO_FALLBACK_TYPE))
    return specs


class EmbeddingCompat(tf.keras.layers.Embedding):
    """Compatibility layer for legacy .keras files that include extra embedding keys."""

    @classmethod
    def from_config(cls, config):
        cfg = dict(config)
        cfg.pop("quantization_config", None)
        return super().from_config(cfg)


def _is_lfs_pointer(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        first_line = path.read_text(encoding="utf-8", errors="ignore").splitlines()[0]
        return first_line.strip() == "version https://git-lfs.github.com/spec/v1"
    except Exception:
        return False


def _looks_like_keras_zip(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(2) == b"PK"
    except Exception:
        return False


def _sanitize_keras_config(value):
    def _initializer_to_string(init_obj):
        if not isinstance(init_obj, dict):
            return init_obj
        class_name = str(init_obj.get("class_name", "")).strip()
        mapping = {
            "Orthogonal": "orthogonal",
            "GlorotUniform": "glorot_uniform",
            "GlorotNormal": "glorot_normal",
            "HeUniform": "he_uniform",
            "HeNormal": "he_normal",
            "RandomUniform": "random_uniform",
            "RandomNormal": "random_normal",
            "Zeros": "zeros",
            "Ones": "ones",
            "TruncatedNormal": "truncated_normal",
        }
        return mapping.get(class_name, class_name.lower())

    if isinstance(value, dict):
        value.pop("quantization_config", None)
        value.pop("shared_object_id", None)

        for key in list(value.keys()):
            v = value[key]
            if key.endswith("_initializer") and isinstance(v, dict) and "class_name" in v:
                # Normalize initializer objects to simple identifiers for robust Keras parsing.
                value[key] = _initializer_to_string(v)
            else:
                _sanitize_keras_config(v)
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


def _rebuild_model_from_keras_archive(archive_path: str, custom_objects):
    """Fallback loader: rebuild from config.json + model.weights.h5 inside .keras."""
    try:
        with zipfile.ZipFile(archive_path, "r") as zin:
            if "config.json" not in zin.namelist() or "model.weights.h5" not in zin.namelist():
                return None
            config = json.loads(zin.read("config.json").decode("utf-8"))
            # Ensure legacy keys such as quantization_config/shared_object_id are removed
            # before deserializing through model_from_json.
            _sanitize_keras_config(config)
            config_text = json.dumps(config)
            weights_bytes = zin.read("model.weights.h5")

        model = keras.models.model_from_json(config_text, custom_objects=custom_objects)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".weights.h5") as tmp_w:
            tmp_w.write(weights_bytes)
            weights_path = tmp_w.name
        model.load_weights(weights_path)
        return model
    except Exception:
        return None


def _rebuild_model_from_weights_only(archive_path: str):
    """Last-resort fallback for Keras 2.x vs 3.x config incompatibilities."""
    try:
        with zipfile.ZipFile(archive_path, "r") as zin:
            if "model.weights.h5" not in zin.namelist():
                return None
            weights_bytes = zin.read("model.weights.h5")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".weights.h5") as tmp_w:
            tmp_w.write(weights_bytes)
            weights_path = tmp_w.name

        with h5py.File(weights_path, "r") as f:
            emb_shape = tuple(f["layers"]["embedding"]["vars"]["0"].shape)  # (vocab, emb_dim)
            dense0_shape = tuple(f["layers"]["dense"]["vars"]["0"].shape)    # (att_dim, dense_units)
            dense1_shape = tuple(f["layers"]["dense_1"]["vars"]["0"].shape)  # (dense_units, num_classes)
            fw_rec_shape = tuple(f["layers"]["bidirectional"]["forward_layer"]["cell"]["vars"]["1"].shape)
            lstm_units = int(fw_rec_shape[0])

        vocab_size, emb_dim = int(emb_shape[0]), int(emb_shape[1])
        dense_units = int(dense0_shape[1])
        num_classes = int(dense1_shape[1])
        max_len = int(os.getenv("MAX_LEN", "50"))

        inputs = tf.keras.Input(shape=(max_len,), name="input_layer")
        x = tf.keras.layers.Embedding(vocab_size, emb_dim, name="embedding")(inputs)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units, return_sequences=True),
            name="bidirectional",
        )(x)
        x = AttentionLayer(name="attention_layer")(x)
        x = tf.keras.layers.Dense(dense_units, activation="relu", name="dense")(x)
        x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="dense_1")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="functional")
        model.load_weights(weights_path)
        return model
    except Exception:
        return None


def _load_json(path: Path):
    if not path.exists() or _is_lfs_pointer(path):
        return None
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return None
    return json.loads(text)


def _resolve_local_or_hf(candidates):
    for candidate in candidates:
        p = Path(candidate)
        if p.exists() and not _is_lfs_pointer(p):
            return str(p)

    for candidate in candidates:
        for repo_id, repo_type in _repo_specs():
            try:
                return hf_hub_download(
                    repo_id=repo_id,
                    filename=candidate,
                    repo_type=repo_type,
                )
            except Exception:
                continue

    return None


def _load_results():
    return _load_json(RESULTS_PATH)


def _load_labels(results):
    candidates = [
        ROOT / "emotion-model-7" / "label_classes.json",
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

    hf_label_candidates = [
        "label_classes.json",
        "emotion-model-7/label_classes.json",
        "data/processed/label_classes.json",
    ]
    for candidate in hf_label_candidates:
        for repo_id, repo_type in _repo_specs():
            try:
                f = hf_hub_download(
                    repo_id=repo_id,
                    filename=candidate,
                    repo_type=repo_type,
                )
                payload = _load_json(Path(f))
                if isinstance(payload, dict) and "classes" in payload:
                    return payload["classes"]
                if isinstance(payload, list) and payload:
                    return payload
            except Exception:
                continue

    label_pickle_candidates = [
        ROOT / "data" / "processed" / "label_encoder.pickle",
        ROOT / "label_encoder.pickle",
    ]
    for c in label_pickle_candidates:
        if c.exists() and not _is_lfs_pointer(c):
            try:
                with c.open("rb") as f:
                    enc = pickle.load(f)
                classes = getattr(enc, "classes_", None)
                if classes is not None and len(classes):
                    return [str(x) for x in classes]
            except Exception:
                continue

    if isinstance(results, dict) and isinstance(results.get("labels"), list) and results.get("labels"):
        return results["labels"]

    return list(DEFAULT_LABELS)


def _load_model():
    local_model_candidates = [
        str(ROOT / "emotion-model-7" / MODEL_FILENAME),
        str(ROOT / "saved_models" / MODEL_FILENAME),
        str(ROOT / MODEL_FILENAME),
    ]
    hf_model_candidates = [
        MODEL_FILENAME,
        f"saved_models/{MODEL_FILENAME}",
        f"emotion-model-7/{MODEL_FILENAME}",
    ]

    tried = []

    def _attempt_load(model_path: str):
        path_obj = Path(model_path)
        if not _looks_like_keras_zip(path_obj):
            tried.append(f"{model_path} -> invalid_keras_archive")
            return None, None

        custom_base = {
            "AttentionLayer": AttentionLayer,
            "Custom>AttentionLayer": AttentionLayer,
            "src.model.AttentionLayer": AttentionLayer,
        }
        custom_compat = {
            "AttentionLayer": AttentionLayer,
            "Custom>AttentionLayer": AttentionLayer,
            "src.model.AttentionLayer": AttentionLayer,
            "Embedding": EmbeddingCompat,
            "EmbeddingCompat": EmbeddingCompat,
        }
        err1 = ""
        err2 = ""
        sanitized_path = _sanitize_keras_archive(model_path)
        primary_path = sanitized_path if sanitized_path else model_path

        # Try progressively more permissive loading modes for cross-version artifacts.
        try:
            m = keras.models.load_model(
                primary_path,
                custom_objects=custom_base,
                compile=False,
            )
            src = "sanitized archive" if sanitized_path else "original archive"
            return m, f"Model loaded from {src}: {model_path}"
        except Exception as e1:
            err1 = f"{type(e1).__name__}:{e1}"

        try:
            m = keras.models.load_model(
                primary_path,
                custom_objects=custom_base,
                compile=False,
                safe_mode=False,
            )
            src = "sanitized archive" if sanitized_path else "original archive"
            return m, f"Model loaded from {src} (safe_mode=False): {model_path}"
        except Exception as e2:
            err2 = f"{type(e2).__name__}:{e2}"

        try:
            if hasattr(keras, "config") and hasattr(keras.config, "enable_unsafe_deserialization"):
                keras.config.enable_unsafe_deserialization()
            m = keras.models.load_model(
                primary_path,
                custom_objects=custom_compat,
                compile=False,
                safe_mode=False,
            )
            src = "sanitized archive" if sanitized_path else "original archive"
            return m, f"Model loaded from {src} with compatibility mode: {model_path}"
        except Exception as e3:
            combined = f"{err1} | {err2} | {type(e3).__name__}:{e3}"
            # Final fallback: rebuild model from sanitized config + weights.
            rebuilt = _rebuild_model_from_keras_archive(primary_path, custom_compat)
            if rebuilt is not None:
                src = "sanitized archive" if sanitized_path else "original archive"
                return rebuilt, f"Model rebuilt from {src} config+weights: {model_path}"

            # Ultimate fallback: build architecture from weight shapes only.
            rebuilt_wo_cfg = _rebuild_model_from_weights_only(primary_path)
            if rebuilt_wo_cfg is not None:
                src = "sanitized archive" if sanitized_path else "original archive"
                return rebuilt_wo_cfg, f"Model rebuilt from {src} weights-only fallback: {model_path}"

            if (not sanitized_path) and ("quantization_config" in combined):
                patched_path = _sanitize_keras_archive(model_path)
                if patched_path:
                    try:
                        m = keras.models.load_model(
                            patched_path,
                            custom_objects=custom_compat,
                            compile=False,
                            safe_mode=False,
                        )
                        return m, f"Model loaded from sanitized archive: {model_path}"
                    except Exception as e4:
                        rebuilt = _rebuild_model_from_keras_archive(patched_path, custom_compat)
                        if rebuilt is not None:
                            return rebuilt, f"Model rebuilt from sanitized archive config+weights: {model_path}"
                        combined = f"{combined} | sanitized:{type(e4).__name__}:{e4}"
            tried.append(f"{model_path} -> {combined}")
            return None, None

    for candidate in local_model_candidates:
        p = Path(candidate)
        if p.exists() and not _is_lfs_pointer(p):
            model, msg = _attempt_load(str(p))
            if model is not None:
                return model, msg

    for candidate in hf_model_candidates:
        for repo_id, repo_type in _repo_specs():
            try:
                remote_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=candidate,
                    repo_type=repo_type,
                )
                model, msg = _attempt_load(remote_file)
                if model is not None:
                    return model, msg
            except Exception as e:
                tried.append(f"hf:{repo_id}:{candidate} -> {type(e).__name__}")

    return None, "Model artifact not found/invalid. Tried: " + " | ".join(tried)

def _load_tokenizer():
    local_tok_candidates = [
        str(ROOT / "emotion-model-7" / "tokenizer.json"),
        str(ROOT / "data" / "processed" / "tokenizer.json"),
        str(ROOT / "tokenizer.json"),
    ]
    hf_tok_candidates = [
        "tokenizer.json",
        "data/processed/tokenizer.json",
        "emotion-model-7/tokenizer.json",
    ]

    tried = []

    for candidate in local_tok_candidates:
        p = Path(candidate)
        if p.exists() and not _is_lfs_pointer(p):
            try:
                payload = _load_json(p)
                if payload is None:
                    raise ValueError("empty_or_invalid_json")
                tok = tokenizer_from_json(json.dumps(payload))
                return tok, f"Tokenizer loaded from: {p}"
            except Exception as e:
                tried.append(f"local:{p} -> {type(e).__name__}")

    tok_pickle_candidates = [
        ROOT / "data" / "processed" / "tokenizer.pickle",
        ROOT / "tokenizer.pickle",
    ]
    for p in tok_pickle_candidates:
        if p.exists() and not _is_lfs_pointer(p):
            try:
                with p.open("rb") as f:
                    tok = pickle.load(f)
                if hasattr(tok, "texts_to_sequences"):
                    return tok, f"Tokenizer loaded from pickle: {p}"
            except Exception as e:
                tried.append(f"local_pickle:{p} -> {type(e).__name__}")

    for candidate in hf_tok_candidates:
        for repo_id, repo_type in _repo_specs():
            try:
                tok_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=candidate,
                    repo_type=repo_type,
                )
                payload = _load_json(Path(tok_file))
                if payload is None:
                    raise ValueError("empty_or_invalid_json")
                tok = tokenizer_from_json(json.dumps(payload))
                return tok, f"Tokenizer loaded from HF ({repo_id}): {candidate}"
            except Exception as e:
                tried.append(f"hf:{repo_id}:{candidate} -> {type(e).__name__}")

    return None, "Tokenizer artifact not found/invalid. Tried: " + " | ".join(tried)


def _validate_runtime(model, tokenizer, labels):
    if model is None:
        return "Runtime validation skipped: model unavailable."
    if tokenizer is None:
        return "Runtime validation skipped: tokenizer unavailable."

    try:
        sample = "i feel happy today"
        max_len = int(model.input_shape[1])
        seq = tokenizer.texts_to_sequences([sample])
        x = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

        probs = model.predict(x, verbose=0)
        if probs.ndim != 2 or probs.shape[0] != 1:
            return f"Runtime validation warning: unexpected output shape {probs.shape}."

        class_count = int(probs.shape[1])
        if labels and class_count != len(labels):
            return (
                "Runtime validation warning: model output classes "
                f"({class_count}) != label count ({len(labels)})."
            )

        prob_sum = float(np.sum(probs[0]))
        return f"Runtime validation OK: output shape={probs.shape}, prob_sum={prob_sum:.4f}."
    except Exception as e:
        return f"Runtime validation failed: {type(e).__name__}: {e}"


def load_artifacts():
    results = _load_results()
    labels = _load_labels(results)

    model, model_msg = _load_model()
    tokenizer, tok_msg = _load_tokenizer()
    validation_msg = _validate_runtime(model, tokenizer, labels)

    health = "READY" if model is not None and tokenizer is not None else "ERROR"
    status = [
        f"System status: {health}",
        f"MODEL_REPO_ID: {MODEL_REPO_ID}",
        f"MODEL_REPO_FALLBACK: {MODEL_REPO_FALLBACK}",
        f"MODEL_REPO_TYPE: {MODEL_REPO_TYPE}",
        f"MODEL_REPO_FALLBACK_TYPE: {MODEL_REPO_FALLBACK_TYPE}",
        model_msg,
        tok_msg,
        validation_msg,
        f"Labels source: {'research_results' if results and results.get('labels') else 'default/local'}",
    ]

    return results, model, tokenizer, labels, "\n".join(status), health


RESULTS, MODEL, TOKENIZER, LABELS, INIT_STATUS, APP_HEALTH = load_artifacts()


def _evaluate_live(model, tokenizer, labels):
    if model is None or tokenizer is None:
        return None, "Live evaluation skipped: model/tokenizer unavailable."

    try:
        eval_samples = int(os.getenv("EVAL_SAMPLES", "2000"))
        ds = load_dataset("go_emotions", split="validation")
        label_names = ds.features["labels"].feature.names
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

        texts = []
        y_true = []
        for row in ds:
            labs = row.get("labels", [])
            if not labs:
                continue
            first = int(labs[0])
            coarse = GO_TO_7.get(label_names[first])
            if coarse is None or coarse not in label_to_idx:
                continue
            texts.append(str(row.get("text", "")))
            y_true.append(label_to_idx[coarse])
            if eval_samples > 0 and len(texts) >= eval_samples:
                break

        if not texts:
            return None, "Live evaluation skipped: no valid validation samples."

        max_len = int(model.input_shape[1])
        x = pad_sequences(
            tokenizer.texts_to_sequences([t.lower() for t in texts]),
            maxlen=max_len,
            padding="post",
            truncating="post",
        )

        t0 = time.perf_counter()
        probs = model.predict(x, verbose=0)
        infer_ms_per_sample = ((time.perf_counter() - t0) * 1000.0) / max(1, len(texts))
        y_pred = np.argmax(probs, axis=1)

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(labels))),
            target_names=labels,
            output_dict=True,
            zero_division=0,
        )
        dist = []
        true_counts = np.bincount(np.array(y_true), minlength=len(labels))
        for i, lbl in enumerate(labels):
            dist.append({"label": lbl, "count": int(true_counts[i])})

        payload = {
            "labels": labels,
            "test_distribution": dist,
            "main_metrics": {
                "confusion_matrix": cm.tolist(),
                "report": report,
                "inference_ms_per_sample": float(infer_ms_per_sample),
            },
            "num_eval_samples": len(texts),
        }
        return payload, f"Live evaluation computed on {len(texts)} validation samples."
    except Exception as e:
        return None, f"Live evaluation failed: {type(e).__name__}: {e}"


LIVE_RESULTS, LIVE_EVAL_STATUS = _evaluate_live(MODEL, TOKENIZER, LABELS)


def _confidence_color(percent: float):
    if percent >= 90:
        return "#10B981", "HIGH"
    if percent >= 70:
        return "#F59E0B", "MEDIUM"
    if percent >= 50:
        return "#F97316", "LOW"
    if percent >= 30:
        return "#EF4444", "VERY LOW"
    return "#7C3AED", "UNRELIABLE"


def _build_confidence_html(probs, labels):
    top_idx = np.argsort(probs)[-5:][::-1]
    rows = []
    for i in top_idx:
        pct = float(probs[i] * 100.0)
        label = labels[i] if i < len(labels) else f"class_{i}"
        bar_color, tier = _confidence_color(pct)
        rows.append(
            f"""
            <div class="conf-row">
              <div class="conf-head">
                <span><strong>{label}</strong></span>
                <span>{pct:.2f}% ({tier})</span>
              </div>
              <div class="conf-track"><div class="conf-fill" style="width:{pct:.2f}%;background:{bar_color};"></div></div>
            </div>
            """
        )
    return "<div id='confidence-card'>" + "".join(rows) + "</div>"


def _plot_session_distribution(session_counts):
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    labels = LABELS if LABELS else list(DEFAULT_LABELS)
    if not session_counts or sum(session_counts) == 0:
        ax.text(0.5, 0.5, "No predictions yet. Submit text to build live distribution.", ha="center", va="center")
        ax.axis("off")
        return fig

    counts = [int(x) for x in session_counts[: len(labels)]]
    bar_colors = [EMOTION_COLORS.get(lbl, "#0066FF") for lbl in labels]
    ax.bar(labels, counts, color=bar_colors)
    ax.set_title("Session Prediction Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count", fontsize=12)
    ax.tick_params(axis="x", rotation=45, labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    plt.tight_layout()
    return fig


def _reset_session_distribution():
    counts = [0 for _ in range(len(LABELS))]
    return counts, _plot_session_distribution(counts)


def predict_text(text, session_counts):
    if session_counts is None or len(session_counts) != len(LABELS):
        session_counts = [0 for _ in range(len(LABELS))]

    if MODEL is None:
        return "Model is not ready. Please check Startup Diagnostics.", INIT_STATUS, "", _plot_session_distribution(session_counts), session_counts
    if TOKENIZER is None:
        return "Tokenizer is not ready. Please check Startup Diagnostics.", INIT_STATUS, "", _plot_session_distribution(session_counts), session_counts
    if not text or not text.strip():
        return "Please enter text.", "", "", _plot_session_distribution(session_counts), session_counts

    try:
        start = time.perf_counter()
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
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        lines.append(f"- inference_time_ms: {elapsed_ms:.2f}")

        pred_color = EMOTION_COLORS.get(pred_label, "#808080")
        summary = f"Primary Emotion: <span class='emotion-tag' style='background:{pred_color};'>{pred_label.upper()}</span> ({conf*100:.2f}%)"
        details = "\n".join(lines)
        conf_html = _build_confidence_html(probs, LABELS)
        updated_counts = list(session_counts)
        if pred_id < len(updated_counts):
            updated_counts[pred_id] += 1
        return summary, details, conf_html, _plot_session_distribution(updated_counts), updated_counts
    except Exception as e:
        return "Prediction failed.", f"{type(e).__name__}: {e}", "", _plot_session_distribution(session_counts), session_counts


def plot_distribution():
    fig, ax = plt.subplots(figsize=(8, 4))
    data = LIVE_RESULTS if LIVE_RESULTS else RESULTS
    if not data:
        ax.text(0.5, 0.5, "No evaluation data available", ha="center", va="center")
        ax.axis("off")
        return fig

    dist = data.get("test_distribution", [])
    labels = [d.get("label", "") for d in dist]
    counts = [d.get("count", 0) for d in dist]

    if not labels:
        ax.text(0.5, 0.5, "Distribution data missing", ha="center", va="center")
        ax.axis("off")
        return fig

    bar_colors = [EMOTION_COLORS.get(lbl, "#0066FF") for lbl in labels]
    ax.bar(labels, counts, color=bar_colors)
    ax.set_title("Test Class Distribution")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    return fig


def plot_confusion_matrix():
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    data = LIVE_RESULTS if LIVE_RESULTS else RESULTS
    if not data:
        ax.text(0.5, 0.5, "No evaluation data available", ha="center", va="center")
        ax.axis("off")
        return fig

    cm = np.array(data.get("main_metrics", {}).get("confusion_matrix", []))
    labels = data.get("labels", [str(i) for i in range(cm.shape[0])])
    if cm.size == 0:
        ax.text(0.5, 0.5, "Confusion matrix missing", ha="center", va="center")
        ax.axis("off")
        return fig

    im = ax.imshow(cm, cmap="cividis")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def metrics_table_md():
    data = LIVE_RESULTS if LIVE_RESULTS else RESULTS
    if not data:
        return "No evaluation metrics available."
    rep = data.get("main_metrics", {}).get("report", {})
    macro = rep.get("macro avg", {})
    return (
        "| Metric | Value |\n"
        "|---|---:|\n"
        f"| Accuracy | {rep.get('accuracy', 0):.4f} |\n"
        f"| Macro Precision | {macro.get('precision', 0):.4f} |\n"
        f"| Macro Recall | {macro.get('recall', 0):.4f} |\n"
        f"| Macro F1 | {macro.get('f1-score', 0):.4f} |\n"
        f"| Inference ms/sample | {data.get('main_metrics', {}).get('inference_ms_per_sample', 0):.4f} |\n"
        f"| Eval samples | {data.get('num_eval_samples', 0)} |"
    )


def metrics_cards_html():
    data = LIVE_RESULTS if LIVE_RESULTS else RESULTS
    if not data:
        return "<div class='metric-grid'><div class='metric-box'>No evaluation metrics available.</div></div>"
    rep = data.get("main_metrics", {}).get("report", {})
    macro = rep.get("macro avg", {})
    acc = float(rep.get("accuracy", 0.0)) * 100.0
    f1 = float(macro.get("f1-score", 0.0)) * 100.0
    lat = float(data.get("main_metrics", {}).get("inference_ms_per_sample", 0.0))
    samples = int(data.get("num_eval_samples", 0))
    return f"""
    <div class="metric-grid">
      <div class="metric-box"><div class="metric-title">Accuracy</div><div class="metric-value">{acc:.2f}%</div></div>
      <div class="metric-box"><div class="metric-title">Macro F1</div><div class="metric-value">{f1:.2f}%</div></div>
      <div class="metric-box"><div class="metric-title">Latency</div><div class="metric-value">{lat:.3f} ms</div></div>
      <div class="metric-box"><div class="metric-title">Eval Samples</div><div class="metric-value">{samples}</div></div>
    </div>
    """


def diagnostics_summary():
    if APP_HEALTH == "READY":
        return "Model Status: READY"
    return "Model Status: ERROR"


UI_CSS = """
:root {
  --card-bg: #ffffff;
  --text-main: #111827;
  --accent: #2563eb;
  --accent-2: #1d4ed8;
  --ok-bg: #ecfdf5;
  --ok-text: #065f46;
  --info-bg: #f3f4f6;
  --info-text: #111827;
  --warn-bg: #f9fafb;
  --warn-text: #374151;
  --err-bg: #fef2f2;
  --err-text: #991b1b;
  --border: #d1d5db;
  --confidence-high: #10B981;
  --confidence-medium: #F59E0B;
  --confidence-low: #EF4444;
  --confidence-very-low: #7C3AED;
}

.gradio-container {
  color: var(--text-main);
  background: #f9fafb;
}

#hero {
  border-radius: 12px;
  padding: 18px 20px;
  background: #ffffff;
  color: #111827;
  border: 1px solid var(--border);
  box-shadow: none;
  margin-bottom: 8px;
}

#hero h1 { margin: 0 0 8px 0; font-size: 28px; }
#hero p { margin: 0; opacity: 1; }

#status-chip {
  margin: 8px 0 2px 0;
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  background: var(--info-bg);
  color: var(--info-text);
  font-weight: 600;
}

.chip-ready {
  background: var(--ok-bg) !important;
  color: var(--ok-text) !important;
}

.chip-error {
  background: var(--err-bg) !important;
  color: var(--err-text) !important;
}

.chip-row {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}

.card {
  border-radius: 14px;
  background: var(--card-bg);
  border: 1px solid var(--border);
  box-shadow: 0 1px 4px rgba(17, 24, 39, 0.08);
  padding: 10px;
}

.card button {
  background: var(--accent) !important;
  color: #ffffff !important;
  border: 1px solid #1d4ed8 !important;
  box-shadow: none !important;
}

.section-title {
  margin: 8px 0 6px 0;
  font-weight: 700;
  color: #0f172a;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}

.metric-box {
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px;
}

.metric-title {
  font-size: 13px;
  color: #334155;
  margin-bottom: 4px;
}

.metric-value {
  font-size: 20px;
  font-weight: 700;
  color: #0f172a;
}

/* Improve readability of text boxes and markdown blocks */
.gradio-container textarea,
.gradio-container input,
.gradio-container .prose,
.gradio-container .markdown,
.gradio-container .wrap {
  color: #0f172a !important;
}

.gradio-container .block {
  border-color: var(--border) !important;
}

#confidence-card {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 10px;
  background: #ffffff;
}

.conf-row {
  margin: 8px 0;
}

.conf-head {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  margin-bottom: 3px;
  color: #0f172a;
}

.conf-track {
  width: 100%;
  height: 10px;
  border-radius: 999px;
  background: #e2e8f0;
  overflow: hidden;
}

.conf-fill {
  height: 100%;
  border-radius: 999px;
}

.emotion-tag {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-weight: 700;
  color: #111827;
}

/* Emotion-specific quick test buttons */
.quick-example-joy button {
  background: var(--accent) !important;
  color: #ffffff !important;
}
.quick-example-angry button {
  background: var(--accent) !important;
  color: #ffffff !important;
}
.quick-example-sad button {
  background: var(--accent) !important;
  color: #ffffff !important;
}
"""

with gr.Blocks(
    title="AffectLens: Attention-Based BiLSTM Emotion Classifier",
    css=UI_CSS,
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:
    gr.HTML(
        """
        <section id="hero">
          <h1>AffectLens: Attention-Based BiLSTM Emotion Classifier</h1>
          <p>Paper-aligned model for short, noisy social media text with live validation metrics and confusion analysis.</p>
        </section>
        """
    )
    chip_class = "chip-ready" if APP_HEALTH == "READY" else "chip-error"
    gr.HTML(
        f"""
        <div class="chip-row">
          <div id="status-chip" class="{chip_class}">Startup Health: {APP_HEALTH}</div>
          <div id="status-chip">Live Evaluation: {LIVE_EVAL_STATUS}</div>
        </div>
        """
    )
    gr.Markdown(f"### {diagnostics_summary()}")
    with gr.Accordion("🔧 System Diagnostics (Click to Expand)", open=False):
        gr.Markdown(f"```text\n{INIT_STATUS}\n```")

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.Markdown("<div class='section-title'>Input & Prediction</div>")
            txt = gr.Textbox(
                label="Input Text (short social post)",
                lines=4,
                placeholder="Type a tweet or Reddit-style post with informal/noisy language...",
            )
            btn = gr.Button("Predict Emotion")
            reset_btn = gr.Button("Reset Live Distribution")
            session_state = gr.State([0 for _ in range(len(LABELS))])

            gr.Markdown("**Quick Test Examples**")
            joy_btn = gr.Button(
                "I am absolutely thrilled and grateful for this wonderful news.",
                elem_classes=["quick-example-joy"],
            )
            angry_btn = gr.Button(
                "This is terrible and unacceptable. I am furious and angry.",
                elem_classes=["quick-example-angry"],
            )
            sad_btn = gr.Button(
                "I feel sad, hopeless, and heartbroken today.",
                elem_classes=["quick-example-sad"],
            )

        with gr.Column(scale=1, elem_classes=["card"]):
            gr.Markdown("<div class='section-title'>Prediction Results</div>")
            out_main = gr.HTML()
            out_top = gr.Textbox(label="Top-5 Probabilities + Latency")
            gr.Markdown("**Confidence Indicators**")
            out_conf = gr.HTML()
            gr.Markdown("<div class='section-title'>Live Model Metrics</div>")
            gr.HTML(metrics_cards_html())

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.Markdown("<div class='section-title'>Live Class Distribution (Current Session)</div>")
            live_dist_plot = gr.Plot(value=_plot_session_distribution([0 for _ in range(len(LABELS))]))
        with gr.Column(scale=1, elem_classes=["card"]):
            gr.Markdown("<div class='section-title'>Model Confusion Matrix (Validation)</div>")
            gr.Plot(value=plot_confusion_matrix)

    btn.click(
        predict_text,
        inputs=[txt, session_state],
        outputs=[out_main, out_top, out_conf, live_dist_plot, session_state],
    )
    reset_btn.click(
        _reset_session_distribution,
        inputs=[],
        outputs=[session_state, live_dist_plot],
    )
    joy_btn.click(lambda: "I am absolutely thrilled and grateful for this wonderful news.", outputs=txt)
    angry_btn.click(lambda: "This is terrible and unacceptable. I am furious and angry.", outputs=txt)
    sad_btn.click(lambda: "I feel sad, hopeless, and heartbroken today.", outputs=txt)

if __name__ == "__main__":
    demo.launch()







