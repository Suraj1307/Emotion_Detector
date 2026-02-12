# src/ekman_mapping.py

# GoEmotions (28) → Ekman (7)

EKMAN_MAPPING = {
    "anger": [
        "anger", "annoyance", "disapproval"
    ],

    "disgust": [
        "disgust"
    ],

    "fear": [
        "fear", "nervousness"
    ],

    "joy": [
        "joy", "amusement", "excitement",
        "love", "gratitude", "optimism",
        "pride", "relief", "approval",
        "admiration", "desire"
    ],

    "sadness": [
        "sadness", "disappointment",
        "grief", "remorse", "embarrassment",
        "shame", "guilt"
    ],

    "surprise": [
        "surprise", "confusion", "curiosity"
    ],

    "neutral": [
        "neutral", "caring"
    ]
}

# Ekman label → ID
EKMAN_LABEL_TO_ID = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "sadness": 4,
    "surprise": 5,
    "neutral": 6
}

# Reverse mapping
ID_TO_EKMAN_LABEL = {v: k for k, v in EKMAN_LABEL_TO_ID.items()}
