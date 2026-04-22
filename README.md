---

title: "Emotion Classification in Social Media Using Attention-Based BiLSTM"
emoji: "💬"
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.34.0"
python_version: "3.10"
app_file: app/app.py
pinned: true
------------

# 💬 Emotion Classification in Social Media using Attention-Based BiLSTM

> A production-ready deep learning system for detecting emotions in noisy social media text, enhanced with attention-based interpretability and an interactive UI.

---

## 🚀 Overview

Social media text (tweets, chats, Reddit posts) is:

* noisy
* unstructured
* emotionally ambiguous

Traditional NLP models struggle to capture subtle emotional signals.

This project solves that using an **Attention-Based BiLSTM**, enabling:

* 🎯 Accurate emotion classification
* 🔍 Word-level interpretability (attention)
* ⚡ Real-time predictions via interactive UI

---

## 🧠 Key Features

* ✅ Classifies text into **7 emotions**
  *(anger, disgust, fear, joy, neutral, sadness, surprise)*

* 🔍 **Attention Visualization**
  Highlights emotionally important words

* 📊 **Confidence Insights**

  * Probability distribution
  * Confidence bands (High / Moderate / Low)

* ⚡ **Fast Inference (~200ms)**

* 🧪 Supports:

  * Single prediction
  * Batch analysis

* 🎨 Clean UI with charts & analytics

---

## 🖥️ Demo

👉 **Live App (HuggingFace Space)**
*Add your link here*

---

## 🏗️ System Architecture

```text
User Input
   ↓
Preprocessing (clean text, normalize)
   ↓
Tokenizer (Keras)
   ↓
Embedding Layer
   ↓
BiLSTM (Bidirectional)
   ↓
Attention Layer ⭐
   ↓
Dense + Softmax
   ↓
Emotion Prediction + Confidence
```

### 🔍 Why Attention?

The attention mechanism:

* focuses on emotionally relevant words
* reduces noise impact
* provides explainability

---

## 📂 Project Structure

```text
emotion-classification/
│
├── app/                        # 🚀 UI + Inference
│   ├── app.py
│   ├── templates/
│   ├── static/
│
├── src/                        # 🧠 Core ML logic
│   ├── model.py
│   ├── preprocess.py
│   ├── predict.py
│   ├── predict_with_attention.py
│   ├── utils.py
│   ├── ekman_mapping.py
│
├── training/                   # 🏋️ Training pipeline
│   ├── train.py
│   ├── download_data.py
│   ├── training_history.json
│
├── models/                     # 💾 Saved models
│   ├── emotion_model.keras
│   ├── tokenizer.pkl
│   ├── label_encoder.pkl
│
├── data/                       # 📊 Dataset
│   ├── raw/
│   ├── processed/
│   ├── data_train.csv
│   ├── data_validation.csv
│
├── research/                   # 📄 Research artifacts
├── notebooks/                  # 📓 Experiments
├── deployment/                 # ⚙️ Deployment configs
│
├── requirements.txt
├── runtime.txt
├── README.md
```

---

## ⚙️ Training Pipeline

* 📌 Dataset: Social media text

* 🏷️ Labels: 7 emotion classes

* 🔄 Preprocessing:

  * URL removal
  * @user normalization
  * hashtag handling

* 🧮 Model:

  * Embedding Layer
  * BiLSTM
  * Attention Layer
  * Dense + Softmax

* 🔤 Tokenization:

  * Keras tokenizer

---

## 📊 Sample Output

* **Primary Emotion:** Joy 😊
* **Confidence:** ~81.7%
* **Inference Time:** ~227 ms

Additional outputs:

* Confidence band
* Top attention tokens
* Full probability distribution

---

## 📈 Model Performance

| Metric      | Value  |
| ----------- | ------ |
| Accuracy    | 0.5926 |
| Precision   | 0.6261 |
| Weighted F1 | 0.6032 |
| Classes     | 7      |

---

## 🔬 Research Alignment

This project aligns with research in:

* Attention-based sequence modeling
* Emotion detection in noisy text
* Explainable AI (XAI)

Includes:

* evaluation outputs
* research report scaffold

---

## 🛠️ Tech Stack

* Python 3.10
* TensorFlow / Keras
* Gradio / Flask (UI)
* NumPy / Pandas

---

## ⚡ Running Locally

```bash
git clone <your-repo>
cd emotion-classification

pip install -r requirements.txt
python app/app.py
```

---

## 🚀 Deployment

Supports:

* HuggingFace Spaces
* Render / Docker

Configs available in:

```
deployment/
```

---

## 💡 Future Improvements

* 🔥 Transformer models (BERT / RoBERTa)
* 🌐 Multilingual emotion detection
* 😊 Emoji-aware embeddings
* 📱 Mobile-optimized UI
* ⚡ Real-time streaming predictions

---

## 🤝 Author

**Suraj Kumar**

* Full Stack + AI Developer
* Passionate about building real-world AI systems

---

## ⭐ Why This Project Stands Out

* Combines **Deep Learning + Interpretability**
* End-to-end system (training → deployment)
* Real-world NLP use case
* Clean architecture + UI

---

## 📬 Feedback

Open an issue or contribute to improve the project 🚀

---
