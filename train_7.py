import ast
import json
import os
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, TFAutoModel

from src.model import AttentionLayer

PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_CSV = PROJECT_DIR / "data_train.csv"
VAL_CSV = PROJECT_DIR / "data_validation.csv"
OUT_DIR = PROJECT_DIR / "emotion-model"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BERT_MODEL_NAME = os.getenv("BERT_MODEL_NAME", "bert-base-uncased")
MAX_LEN = int(os.getenv("MAX_LEN", "64"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
EPOCHS = int(os.getenv("EPOCHS", "2"))
SAMPLE_TRAIN = int(os.getenv("SAMPLE_TRAIN", "16000"))
SAMPLE_VAL = int(os.getenv("SAMPLE_VAL", "4000"))
FREEZE_BERT = os.getenv("FREEZE_BERT", "false").lower() == "true"

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
    df["text"] = df["text"].astype(str)
    return df[["text", "label_id"]]


def encode_texts(tokenizer, texts):
    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="np",
    )
    return enc["input_ids"].astype("int32"), enc["attention_mask"].astype("int32")


def build_model(num_classes: int):
    input_ids = tf.keras.layers.Input(
        shape=(MAX_LEN,), dtype=tf.int32, name="input_ids"
    )
    attention_mask = tf.keras.layers.Input(
        shape=(MAX_LEN,), dtype=tf.int32, name="attention_mask"
    )

    bert = TFAutoModel.from_pretrained(BERT_MODEL_NAME, name="bert_encoder")
    bert.trainable = not FREEZE_BERT
    bert_outputs = bert(input_ids=input_ids, attention_mask=attention_mask)
    token_embeddings = bert_outputs.last_hidden_state

    seq = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(96, return_sequences=True, dropout=0.2),
        name="bilstm_layer",
    )(token_embeddings)

    attn_vec = AttentionLayer(name="attention_layer")(seq)

    cnn3 = tf.keras.layers.Conv1D(
        128, 3, activation="relu", padding="same", name="cnn_k3"
    )(seq)
    cnn5 = tf.keras.layers.Conv1D(
        128, 5, activation="relu", padding="same", name="cnn_k5"
    )(seq)
    pool3 = tf.keras.layers.GlobalMaxPooling1D(name="pool_k3")(cnn3)
    pool5 = tf.keras.layers.GlobalMaxPooling1D(name="pool_k5")(cnn5)

    fused = tf.keras.layers.Concatenate(name="fusion_concat")([attn_vec, pool3, pool5])
    fused = tf.keras.layers.Dense(128, activation="relu", name="fusion_dense")(fused)
    fused = tf.keras.layers.Dropout(0.4, name="fusion_dropout")(fused)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="classifier"
    )(fused)

    model = tf.keras.Model(
        inputs={"input_ids": input_ids, "attention_mask": attention_mask},
        outputs=outputs,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
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

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    x_train_ids, x_train_mask = encode_texts(tokenizer, train_df["text"].tolist())
    x_val_ids, x_val_mask = encode_texts(tokenizer, val_df["text"].tolist())

    model = build_model(num_classes=len(TARGET_LABELS))

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(TARGET_LABELS)),
        y=y_train,
    )
    class_weight_map = {int(i): float(w) for i, w in enumerate(class_weights)}

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=1,
            restore_best_weights=True,
        ),
    ]

    model.fit(
        {"input_ids": x_train_ids, "attention_mask": x_train_mask},
        y_train,
        validation_data=(
            {"input_ids": x_val_ids, "attention_mask": x_val_mask},
            y_val,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_map,
        verbose=0,
    )

    model.save(OUT_DIR / "emotion_model_final.keras")
    tokenizer.save_pretrained(OUT_DIR / "tokenizer")
    (OUT_DIR / "label_classes.json").write_text(
        json.dumps(TARGET_LABELS), encoding="utf-8"
    )

    print("Saved Attention-based BiLSTM emotion model artifacts to:", OUT_DIR)


if __name__ == "__main__":
    main()
