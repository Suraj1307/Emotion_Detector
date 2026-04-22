# 💬 Emotion Classification in Social Media using Attention-Based BiLSTM

> A production-ready deep learning system for detecting emotions in noisy social media text with attention-based interpretability and an interactive UI.

---

## 🚀 Overview

Social media text (tweets, chats, Reddit posts) is:

* noisy
* unstructured
* emotionally ambiguous

Traditional NLP models struggle to capture subtle emotional signals.

This project uses an **Attention-Based BiLSTM** to:

* 🎯 Accurately classify emotions
* 🔍 Highlight important words using attention
* ⚡ Provide real-time predictions via UI

---

## 🧠 Key Features

* ✅ Classifies text into **7 emotions**
  *(anger, disgust, fear, joy, neutral, sadness, surprise)*

* 🔍 **Attention Visualization**
  Highlights emotionally important words

* 📊 **Confidence Insights**

  * Probability distribution
  * Confidence bands (High / Moderate / Low)

* ⚡ Fast inference (~200ms)

* 🧪 Supports:

  * Single prediction
  * Batch analysis

* 🎨 Clean UI with charts & analytics

---

## 🖥️ Demo

👉 Add your HuggingFace Space / Live App link here

---

## 🏗️ System Architecture

```text
Input Text
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

* Focuses on emotionally relevant words
* Reduces noise impact
* Improves interpretability

---

## 📂 Project Structure

```text
emotion-classification/
├── app/                        # UI + Inference
├── src/                        # Core ML logic
├── training/                   # Training pipeline
├── models/                     # Saved models
├── data/                       # Dataset
├── research/                   # Research artifacts
├── notebooks/                  # Experiments
├── deployment/                 # Deployment configs
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## ⚙️ Training Pipeline

* 📌 Dataset: Social media text
* 🏷️ Labels: 7 emotion classes

### Preprocessing:

* URL removal
* @user normalization
* hashtag handling

### Model:

* Embedding Layer
* BiLSTM
* Attention Layer
* Dense + Softmax

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

## 🛠️ Tech Stack

* Python 3.10
* TensorFlow / Keras
* Gradio / Flask
* NumPy / Pandas

---

## ⚡ Run Locally

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

Deployment configs available in:

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

## 👨‍💻 Author

**Suraj Kumar**
Full Stack + AI Developer

---

## ⭐ Why This Project Stands Out

* Combines **Deep Learning + Interpretability**
* End-to-end system (training → deployment)
* Real-world NLP use case
* Clean architecture + UI

---

## 📬 Feedback

Feel free to open issues or contribute 🚀
