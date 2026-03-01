import ast
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.model import AttentionLayer

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_CSV = PROJECT_DIR / "data_train.csv"
VAL_CSV = PROJECT_DIR / "data_validation.csv"
OUT_DIR = PROJECT_DIR / "emotion-model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LEN = int(os.getenv("MAX_LEN", "64"))
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "50000"))
EMBED_DIM = int(os.getenv("EMBED_DIM", "200"))
LSTM_UNITS = int(os.getenv("LSTM_UNITS", "96"))
DENSE_UNITS = int(os.getenv("DENSE_UNITS", "96"))
SPATIAL_DROPOUT = float(os.getenv("SPATIAL_DROPOUT", "0.25"))
DROPOUT_RATE = float(os.getenv("DROPOUT_RATE", "0.5"))
RECURRENT_DROPOUT = float(os.getenv("RECURRENT_DROPOUT", "0.25"))
LSTM_L2 = float(os.getenv("LSTM_L2", "3e-4"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "48"))
EPOCHS = int(os.getenv("EPOCHS", "14"))
SAMPLE_TRAIN = int(os.getenv("SAMPLE_TRAIN", "0"))
SAMPLE_VAL = int(os.getenv("SAMPLE_VAL", "0"))
USE_FOCAL_LOSS = os.getenv("USE_FOCAL_LOSS", "0") == "1"
FOCAL_GAMMA = float(os.getenv("FOCAL_GAMMA", "2.0"))
OVERSAMPLE_MINORITY = os.getenv("OVERSAMPLE_MINORITY", "1") == "1"
MINORITY_TARGET_RATIO = float(os.getenv("MINORITY_TARGET_RATIO", "0.30"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "3e-4"))
LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", "0.04"))
EARLY_STOP_PATIENCE = int(os.getenv("EARLY_STOP_PATIENCE", "3"))
LR_PATIENCE = int(os.getenv("LR_PATIENCE", "1"))
LR_MIN = float(os.getenv("LR_MIN", "5e-6"))
DROP_AMBIGUOUS = os.getenv("DROP_AMBIGUOUS", "0") == "1"
AUGMENT_MINORITY = os.getenv("AUGMENT_MINORITY", "1") == "1"
MINORITY_AUG_MAX_PER_CLASS = int(os.getenv("MINORITY_AUG_MAX_PER_CLASS", "1800"))

TARGET_LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
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

LABEL_TO_ID = {label: i for i, label in enumerate(TARGET_LABELS)}


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
        # Help model separate clearly negative content from neutral class.
        text += " anger_cue disgust_cue sadness_cue"
    return text


def parse_labels(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except Exception:
            return []
    return []


def pick_coarse_label(label_ids, id_to_label):
    mapped = []
    for idx in label_ids:
        fine = id_to_label.get(int(idx))
        if fine is None:
            continue
        coarse = GO_TO_7.get(fine)
        if coarse and coarse not in mapped:
            mapped.append(coarse)

    if not mapped:
        return None
    if len(mapped) == 1:
        return mapped[0]

    # Multi-label rows are noisier for single-label training.
    mapped_wo_neutral = [m for m in mapped if m != "neutral"]
    if len(mapped_wo_neutral) == 1:
        return mapped_wo_neutral[0]
    if DROP_AMBIGUOUS:
        return None

    # Fallback deterministic priority if ambiguity is allowed.
    priority = ["anger", "disgust", "fear", "sadness", "surprise", "joy", "neutral"]
    for p in priority:
        if p in mapped:
            return p
    return mapped[0]


def ensure_training_csvs():
    if TRAIN_CSV.exists() and VAL_CSV.exists():
        return

    ds = load_dataset("go_emotions")
    train_df = ds["train"].to_pandas()[["text", "labels"]]
    val_df = ds["validation"].to_pandas()[["text", "labels"]]
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)


def load_csv(path: Path, id_to_label):
    df = pd.read_csv(path)
    df = df[df["labels"].notna()]
    df["labels_parsed"] = df["labels"].apply(parse_labels)
    df = df[df["labels_parsed"].map(len) > 0]
    df["label"] = df["labels_parsed"].apply(lambda x: pick_coarse_label(x, id_to_label))
    df = df[df["label"].notna()]
    df["text"] = df["text"].astype(str).map(normalize_social_text)
    return df[["text", "label"]]


def oversample_minority(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    if not OVERSAMPLE_MINORITY:
        return df
    counts = df[label_col].value_counts()
    if counts.empty:
        return df
    max_count = int(counts.max())
    target_min = max(1, int(max_count * MINORITY_TARGET_RATIO))

    parts = [df]
    for lbl, cnt in counts.items():
        if cnt >= target_min:
            continue
        need = target_min - int(cnt)
        subset = df[df[label_col] == lbl]
        if subset.empty:
            continue
        parts.append(subset.sample(need, replace=True, random_state=42))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=42).reset_index(drop=True)


def augment_minority_texts(df: pd.DataFrame) -> pd.DataFrame:
    if not AUGMENT_MINORITY or df.empty:
        return df

    cue_suffix = {
        "fear": " fear_cue scared terrified anxious panic",
        "disgust": " disgust_cue disgusting repulsive gross",
        "surprise": " surprise_cue unexpected wow shocking",
    }

    parts = [df]
    rng = np.random.RandomState(42)
    for label, suffix in cue_suffix.items():
        subset = df[df["label"] == label]
        if subset.empty:
            continue
        n = min(len(subset), MINORITY_AUG_MAX_PER_CLASS)
        sampled = subset.sample(n=n, replace=(len(subset) < n), random_state=42).copy()
        sampled["text"] = sampled["text"].astype(str) + suffix
        parts.append(sampled)

    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=rng).reset_index(drop=True)


def build_model(num_classes: int, vocab_size: int):
    inputs = tf.keras.Input(shape=(MAX_LEN,), dtype="int32", name="input_layer")
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBED_DIM,
        name="embedding",
    )(inputs)
    x = tf.keras.layers.SpatialDropout1D(SPATIAL_DROPOUT)(x)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            LSTM_UNITS,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=RECURRENT_DROPOUT,
            kernel_regularizer=tf.keras.regularizers.l2(LSTM_L2),
            recurrent_regularizer=tf.keras.regularizers.l2(LSTM_L2),
        ),
        name="bilstm_layer",
    )(x)
    x = AttentionLayer(name="attention_layer")(x)
    x = tf.keras.layers.Dense(
        DENSE_UNITS,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="dense",
    )(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="dense_1")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="attention_bilstm")
    if USE_FOCAL_LOSS:
        def sparse_focal_loss(y_true, y_pred):
            y_true_i = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
            y_onehot = tf.one_hot(y_true_i, depth=num_classes)
            y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            ce = -tf.reduce_sum(y_onehot * tf.math.log(y_pred_clipped), axis=-1)
            p_t = tf.reduce_sum(y_onehot * y_pred_clipped, axis=-1)
            focal = tf.pow(1.0 - p_t, FOCAL_GAMMA) * ce
            return tf.reduce_mean(focal)
        loss_fn = sparse_focal_loss
    else:
        try:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                label_smoothing=LABEL_SMOOTHING
            )
        except TypeError:
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model


def main():
    tf.get_logger().setLevel("ERROR")
    ensure_training_csvs()

    ds = load_dataset("go_emotions")
    label_names = ds["train"].features["labels"].feature.names
    id_to_label = {i: name for i, name in enumerate(label_names)}

    train_df = load_csv(TRAIN_CSV, id_to_label)
    val_df = load_csv(VAL_CSV, id_to_label)
    train_df = oversample_minority(train_df, "label")
    train_df = augment_minority_texts(train_df)

    if SAMPLE_TRAIN > 0 and len(train_df) > SAMPLE_TRAIN:
        train_df = train_df.sample(SAMPLE_TRAIN, random_state=42).reset_index(drop=True)
    if SAMPLE_VAL > 0 and len(val_df) > SAMPLE_VAL:
        val_df = val_df.sample(SAMPLE_VAL, random_state=42).reset_index(drop=True)

    y_train = train_df["label"].map(LABEL_TO_ID).astype("int32").values
    y_val = val_df["label"].map(LABEL_TO_ID).astype("int32").values

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_df["text"].tolist())

    x_train = pad_sequences(
        tokenizer.texts_to_sequences(train_df["text"].tolist()),
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )
    x_val = pad_sequences(
        tokenizer.texts_to_sequences(val_df["text"].tolist()),
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )

    vocab_size = min(VOCAB_SIZE, len(tokenizer.word_index) + 1)
    model = build_model(num_classes=len(TARGET_LABELS), vocab_size=vocab_size)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(TARGET_LABELS)),
        y=y_train,
    )
    class_weight_map = {int(i): float(w) for i, w in enumerate(class_weights)}
    print("Class weights:", class_weight_map)
    print("Loss:", "SparseCategoricalFocalCrossentropy" if USE_FOCAL_LOSS else "sparse_categorical_crossentropy")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(OUT_DIR / "best_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=LR_PATIENCE,
            min_lr=LR_MIN,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_map,
        verbose=1,
    )

    best_model_path = OUT_DIR / "best_model.keras"
    if best_model_path.exists():
        try:
            model = tf.keras.models.load_model(
                best_model_path, custom_objects={"AttentionLayer": AttentionLayer}
            )
        except Exception:
            pass

    model.save(OUT_DIR / "emotion_model_final.keras")
    best_epoch = int(np.argmax(history.history.get("val_accuracy", [0.0]))) + 1
    history.history["best_epoch"] = [best_epoch]
    (OUT_DIR / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    (OUT_DIR / "label_classes.json").write_text(
        json.dumps(TARGET_LABELS), encoding="utf-8"
    )
    (OUT_DIR / "training_history.json").write_text(
        json.dumps(history.history, indent=2), encoding="utf-8"
    )
    (PROJECT_DIR / "training_history.json").write_text(
        json.dumps(history.history, indent=2), encoding="utf-8"
    )
    print("Saved Attention-based BiLSTM emotion model artifacts to:", OUT_DIR)


if __name__ == "__main__":
    main()
