# src/preprocess.py
import re

# --------------------------------------------
# JOY SEMANTIC EXPANSION
# --------------------------------------------
JOY_WORDS = [
    "thrilled", "ecstatic", "overjoyed", "delighted",
    "fantastic", "amazing", "awesome", "blessed",
    "grateful", "excited", "promotion", "celebrate",
    "great news", "wonderful", "love", "best",
    "excellent", "cheerful", "glad", "pleased"
]

def joy_augmentation(text):
    for word in JOY_WORDS:
        if word in text:
            text += " happy joy excited"
            break
    return text


# --------------------------------------------
# CLEAN TEXT (TRAINING + INFERENCE SAFE)
# --------------------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------
# FULL PIPELINE
# --------------------------------------------
def preprocess_text(text):
    text = clean_text(text)
    text = joy_augmentation(text)
    return text
