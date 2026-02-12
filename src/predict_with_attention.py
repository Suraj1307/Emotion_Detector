import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

try:
    from src.model import AttentionLayer
    from src.preprocess import preprocess_text
except ModuleNotFoundError:
    from model import AttentionLayer
    from preprocess import preprocess_text


MODEL_PATH = "saved_models/emotion_model_final.keras"
TOKENIZER_PATH = "data/processed/tokenizer.pickle"
LABEL_ENCODER_PATH = "data/processed/label_encoder.pickle"


model = load_model(
    MODEL_PATH,
    custom_objects={"AttentionLayer": AttentionLayer},
)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

MAX_LEN = model.input_shape[1]


def _find_attention_layer(trained_model):
    for layer in trained_model.layers:
        if isinstance(layer, AttentionLayer):
            return layer
    raise ValueError("No AttentionLayer found in loaded model.")


attention_layer = _find_attention_layer(model)
sequence_model = Model(inputs=model.input, outputs=attention_layer.input)


def _compute_attention_weights(sequence_output):
    w = attention_layer.W
    b = attention_layer.b
    u = attention_layer.u

    score = tf.tanh(tf.tensordot(sequence_output, w, axes=1) + b)
    weights = tf.nn.softmax(tf.tensordot(score, u, axes=1), axis=1)
    return tf.squeeze(weights, axis=-1).numpy()


def _apply_confidence_rule(predicted_label, confidence, threshold):
    if threshold is not None and confidence < threshold:
        return "ambiguous"
    return predicted_label


def get_predictions_with_attention(text, threshold=0.4):
    clean = preprocess_text(text)
    words = clean.split()

    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    predictions = model.predict(padded, verbose=0)[0]
    emotion_id = int(np.argmax(predictions))
    model_label = label_encoder.inverse_transform([emotion_id])[0]
    confidence = float(np.max(predictions))
    final_label = _apply_confidence_rule(model_label, confidence, threshold)

    sequence_output = sequence_model.predict(padded, verbose=0)
    weights = _compute_attention_weights(sequence_output)[0]
    weights = weights[: len(words)]

    if weights.size > 0 and float(np.max(weights)) > 0:
        weights = weights / float(np.max(weights))

    return words, weights, final_label, confidence, model_label


def plot_attention_heatmap(text):
    words, weights, emotion_label, confidence, model_label = get_predictions_with_attention(text)

    if len(words) == 0 or weights.size == 0:
        print("No tokens found after preprocessing.")
        return

    plt.figure(figsize=(max(6, len(words) * 1.2), 2))
    sns.heatmap(
        [weights],
        annot=[words],
        fmt="",
        cmap="YlOrRd",
        cbar=False,
    )

    plt.title(
        f"Attention (final={emotion_label}, model={model_label}, conf={confidence:.3f})"
    )
    plt.yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sample = "I am so incredibly happy for your success"
    words, weights, emotion, conf, model_label = get_predictions_with_attention(sample)
    print("Text:", sample)
    print("Predicted:", emotion, f"(confidence={conf:.3f}, model={model_label})")
    print("Top tokens by attention:")
    ranked = sorted(zip(words, weights), key=lambda x: x[1], reverse=True)[:5]
    for token, wt in ranked:
        print(f"  {token:<15} {wt:.3f}")
