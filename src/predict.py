import numpy as np
import pickle
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
    custom_objects={"AttentionLayer": AttentionLayer}
)

MAX_LEN = model.input_shape[1]

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)


def _find_attention_layer(trained_model):
    for layer in trained_model.layers:
        if isinstance(layer, AttentionLayer):
            return layer
    return None


attention_layer = _find_attention_layer(model)
sequence_model = None
if attention_layer is not None:
    sequence_model = Model(inputs=model.input, outputs=attention_layer.input)


def clean_text(text):
    return preprocess_text(text)


def _apply_confidence_rule(base_label, confidence, threshold):
    if threshold is not None and confidence < threshold:
        return "ambiguous"
    return base_label


def _compute_attention_weights(sequence_output):
    if attention_layer is None:
        return None

    score = tf.tanh(
        tf.tensordot(sequence_output, attention_layer.W, axes=1) + attention_layer.b
    )
    weights = tf.nn.softmax(
        tf.tensordot(score, attention_layer.u, axes=1),
        axis=1,
    )
    return tf.squeeze(weights, axis=-1).numpy()


def _build_attention_tokens(cleaned_text, padded_input, top_k=8):
    if sequence_model is None:
        return []

    words = cleaned_text.split()
    if not words:
        return []

    sequence_output = sequence_model.predict(padded_input, verbose=0)
    weights = _compute_attention_weights(sequence_output)
    if weights is None or len(weights) == 0:
        return []

    token_weights = weights[0][: len(words)]
    if token_weights.size == 0:
        return []

    max_weight = float(np.max(token_weights))
    if max_weight > 0:
        token_weights = token_weights / max_weight

    ranked = sorted(
        zip(words, token_weights),
        key=lambda x: float(x[1]),
        reverse=True,
    )[:top_k]

    return [
        {"token": token, "weight": round(float(weight), 4)}
        for token, weight in ranked
    ]


def predict_detailed(text, threshold=0.4, top_attention_tokens=8):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

    probs = model.predict(padded, verbose=0)[0]
    pred_id = int(np.argmax(probs))
    confidence = float(np.max(probs))
    base_label = label_encoder.inverse_transform([pred_id])[0]
    final_label = _apply_confidence_rule(base_label, confidence, threshold)

    all_probs = [
        {
            "emotion": label_encoder.inverse_transform([i])[0],
            "probability": round(float(p), 6),
        }
        for i, p in enumerate(probs)
    ]
    all_probs.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "input_text": text,
        "cleaned_text": cleaned,
        "final_emotion": final_label,
        "model_emotion": base_label,
        "confidence": round(confidence, 6),
        "all_probabilities": all_probs,
        "attention_tokens": _build_attention_tokens(
            cleaned, padded, top_k=top_attention_tokens
        ),
    }


def predict_emotion_with_probs(text, threshold=0.4, debug=True):
    details = predict_detailed(text, threshold=threshold)
    probs = np.array([x["probability"] for x in details["all_probabilities"]], dtype=float)
    labels = [x["emotion"] for x in details["all_probabilities"]]

    # Convert sorted probabilities back to label-index order
    ordered_probs = np.zeros(len(label_encoder.classes_), dtype=float)
    for label, prob in zip(labels, probs):
        idx = int(np.where(label_encoder.classes_ == label)[0][0])
        ordered_probs[idx] = prob

    if debug:
        print("\nText:", text)
        print("Probabilities:")
        for i, label in enumerate(label_encoder.classes_):
            print(f"  {label:<10}: {round(float(ordered_probs[i]), 3)}")
        print("Final Emotion:", details["final_emotion"])
        print("Confidence:", round(details["confidence"], 3))

    return details["final_emotion"], details["confidence"], ordered_probs


def predict_emotion(text, threshold=0.4, debug=True):
    predicted_emotion, confidence, _ = predict_emotion_with_probs(
        text,
        threshold=threshold,
        debug=debug,
    )
    return predicted_emotion, confidence


if __name__ == "__main__":
    test_texts = [
        "I'm absolutely thrilled about this amazing news!",
        "This is terrible and I hate it",
        "I feel so sad and depressed",
        "Fear about interview",
        "Rejection and frustration",
    ]

    for t in test_texts:
        predict_emotion(t)
