---
title: "Emotion Classification in Social Media Using Attention-Based BiLSTM"
emoji: "💬"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.34.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Emotion Classification in Social Media Using Attention-Based BiLSTM

## Problem Statement
Emotion classification of short, noisy social media text is difficult because key emotional cues are often informal, fragmented, and mixed across emotions. Traditional classifiers miss these subtle cues.

This project uses an attention-based BiLSTM model to:
- detect emotion from short posts (tweets/Reddit-style text),
- prioritize emotionally relevant words through attention,
- improve interpretability and robustness on noisy text.

## Model + Deployment
- Model repository: `SurajAI2025/Emotion`
- Inference app: `app.py` (Gradio)
- Core custom layer: `src/model.py` (`AttentionLayer`)

## Training Pipeline
- Script: `train_7.py`
- Label mapping to 7 target emotions: anger, disgust, fear, joy, neutral, sadness, surprise
- Sequence backbone: BiLSTM + Attention
- Tokenization: Keras tokenizer (`tokenizer.json`)
- Input focus: short/noisy social text normalization (URL, @user, hashtag handling)

## Attention Visualization
- The app displays token-level attention highlighting to show emotionally relevant words.

## Research Alignment
The implementation is aligned to the paper goal of attention-driven sequence modeling for social-media emotion recognition and includes:
- confusion-matrix style evaluation assets (`research/outputs/`)
- report scaffolding for research presentation (`research/RESEARCH_REPORT.md`)

