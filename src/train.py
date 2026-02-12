import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

try:
    from src.model import build_model
    from src.utils import load_embedding_matrix
    from src.preprocess import preprocess_text
except ModuleNotFoundError:
    from model import build_model
    from utils import load_embedding_matrix
    from preprocess import preprocess_text


# ==========================================
# CONFIG
# ==========================================
MAX_LEN = 100
EMBEDDING_DIM = 100
VOCAB_SIZE_LIMIT = 50000
TARGET_CLASS_SIZE = 15000

GLOVE_PATH = "embeddings/glove.twitter.27B.100d.txt"
MODEL_PATH = "saved_models/emotion_model_final.keras"
TOKENIZER_PATH = "data/processed/tokenizer.pickle"
LABEL_ENCODER_PATH = "data/processed/label_encoder.pickle"


# ==========================================
# LOAD DATA
# ==========================================
def load_dataset():
    import ast

    df_go = pd.read_csv("data/raw/emotion_data.csv")
    df_txt = pd.read_csv("data/raw/text.csv")

    texts = []
    labels = []

    # GoEmotions "simplified" label IDs:
    # 11=disgust, 14=fear, 15=gratitude, 16=grief, 17=joy, 18=love,
    # 25=sadness, 26=surprise, 27=neutral
    GOEMO_MAP = {
        2: "anger", 3: "anger", 14: "fear",
        17: "joy", 18: "joy", 15: "joy",
        11: "disgust", 25: "sadness",
        26: "surprise", 27: "neutral"
    }

    for _, row in df_go.iterrows():
        try:
            ids_raw = row["labels"]
            if isinstance(ids_raw, list):
                ids = ids_raw
            elif isinstance(ids_raw, str):
                ids = ast.literal_eval(ids_raw)
            else:
                ids = [int(ids_raw)]

            if ids:
                emotion = GOEMO_MAP.get(int(ids[0]))
                if emotion:
                    texts.append(preprocess_text(row["text"]))
                    labels.append(emotion)
        except Exception:
            continue

    TEXT_LABEL_MAP = {
        0: "sadness", 1: "joy", 2: "joy",
        3: "anger", 4: "fear", 5: "surprise"
    }

    for _, row in df_txt.iterrows():
        try:
            emotion = TEXT_LABEL_MAP.get(int(row["label"]))
            if emotion:
                texts.append(preprocess_text(row["text"]))
                labels.append(emotion)
        except Exception:
            continue

    return pd.DataFrame({"text": texts, "label": labels})


# ==========================================
# BALANCING
# ==========================================
def balance_dataset(df):
    target = TARGET_CLASS_SIZE
    frames = []

    for label in df["label"].unique():
        subset = df[df["label"] == label]
        if len(subset) < target:
            subset = resample(
                subset, replace=True, n_samples=target, random_state=42
            )
        else:
            subset = subset.sample(n=target, random_state=42)
        frames.append(subset)

    return pd.concat(frames, ignore_index=True)


# ==========================================
# TRAIN
# ==========================================
def run_training():

    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    df = load_dataset()
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df = balance_dataset(df)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])

    with open(LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    tokenizer = Tokenizer(num_words=VOCAB_SIZE_LIMIT, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["text"])

    X = tokenizer.texts_to_sequences(df["text"])
    X = pad_sequences(X, maxlen=MAX_LEN, padding="post")

    with open(TOKENIZER_PATH, "wb") as f:
        pickle.dump(tokenizer, f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    embedding_matrix = load_embedding_matrix(
        tokenizer, GLOVE_PATH, EMBEDDING_DIM
    )

    model = build_model(
        vocab_size=len(tokenizer.word_index) + 1,
        embedding_dim=EMBEDDING_DIM,
        max_len=MAX_LEN,
        num_classes=len(label_encoder.classes_),
        embedding_matrix=embedding_matrix
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=8,
        batch_size=256,
        callbacks=[early_stop]
    )

    model.save(MODEL_PATH)
    print("Model saved.")


if __name__ == "__main__":
    run_training()
