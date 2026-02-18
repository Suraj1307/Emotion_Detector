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

MAX_LEN = int(os.getenv("MAX_LEN", "50"))
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "30000"))
EMBED_DIM = int(os.getenv("EMBED_DIM", "100"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
EPOCHS = int(os.getenv("EPOCHS", "6"))
SAMPLE_TRAIN = int(os.getenv("SAMPLE_TRAIN", "18000"))
SAMPLE_VAL = int(os.getenv("SAMPLE_VAL", "4000"))

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
            return ast.literal_eval(value)
        except Exception:
            return []
    return []


def ensure_training_csvs():
    if TRAIN_CSV.exists() and VAL_CSV.exists():
        return

    ds = load_dataset("go_emotions")
    train_df = ds["train"].to_pandas()[["text", "labels"]]
    val_df = ds["validation"].to_pandas()[["text", "labels"]]
    train_df.to_csv(TRAIN_CSV, index=False)
    val_df.to_csv(VAL_CSV, index=False)


def load_csv(path: Path):
    df = pd.read_csv(path)
    df = df[df["labels"].notna()]
    df["labels_parsed"] = df["labels"].apply(parse_labels)
    df = df[df["labels_parsed"].map(len) > 0]
    df["label_id"] = df["labels_parsed"].apply(lambda x: x[0])
    df["text"] = df["text"].astype(str).map(normalize_social_text)
    return df[["text", "label_id"]]


def build_model(num_classes: int, vocab_size: int):
    inputs = tf.keras.Input(shape=(MAX_LEN,), dtype="int32", name="input_layer")
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=EMBED_DIM,
        name="embedding",
    )(inputs)
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True),
        name="bilstm_layer",
    )(x)
    x = AttentionLayer(name="attention_layer")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="dense_1")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="attention_bilstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    tf.get_logger().setLevel("ERROR")
    ensure_training_csvs()

    ds = load_dataset("go_emotions")
    label_names = ds["train"].features["labels"].feature.names
    id_to_label = {i: name for i, name in enumerate(label_names)}

    train_df = load_csv(TRAIN_CSV)
    val_df = load_csv(VAL_CSV)

    train_df["label"] = train_df["label_id"].map(lambda i: GO_TO_7[id_to_label[i]])
    val_df["label"] = val_df["label_id"].map(lambda i: GO_TO_7[id_to_label[i]])

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

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
        ),
    ]

    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_map,
        verbose=1,
    )

    model.save(OUT_DIR / "emotion_model_final.keras")
    (OUT_DIR / "tokenizer.json").write_text(tokenizer.to_json(), encoding="utf-8")
    (OUT_DIR / "label_classes.json").write_text(
        json.dumps(TARGET_LABELS), encoding="utf-8"
    )
    print("Saved Attention-based BiLSTM emotion model artifacts to:", OUT_DIR)


if __name__ == "__main__":
    main()
