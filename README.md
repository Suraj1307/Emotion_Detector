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

> A production-ready deep learning system for detecting emotions in noisy social media text with attention-based interpretability.

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

* 🔍 Attention visualization (word importance)

* 📊 Confidence insights:

  * Probability distribution
  * Confidence bands

* ⚡ Fast inference (~200ms)

* 🧪 Supports:

  * Single prediction
  * Batch analysis

---

## 🖥️ Demo

👉 Add your HuggingFace Space link here

---

## 🏗️ System Architecture

```
Input → Preprocessing → Tokenizer → Embedding → BiLSTM → Attention → Dense → Output
```

### 🔍 Why Attention?

* Focuses on emotionally relevant words
* Reduces noise impact
* Improves interpretability

---

## 📂 Project Structure

```
emotion-classification/
├── app/
├── src/
├── training/
├── models/
├── data/
├── research/
├── notebooks/
├── deployment/
├── README.md
```

---

## ⚙️ Training Pipeline

* Dataset: Social media text
* Labels: 7 emotions

### Preprocessing:

* URL removal
* @user normalization
* hashtag handling

### Model:

* Embedding
* BiLSTM
* Attention
* Dense + Softmax

---

## 📊 Sample Output

* **Emotion:** Joy 😊
* **Confidence:** ~81.7%
* **Inference Time:** ~227 ms

---

## 📈 Model Performance

| Metric      | Value  |
| ----------- | ------ |
| Accuracy    | 0.5926 |
| Precision   | 0.6261 |
| Weighted F1 | 0.6032 |

---

## 🛠️ Tech Stack

* Python 3.10
* TensorFlow / Keras
* Gradio
* NumPy / Pandas

---

## ⚡ Run Locally

```bash
pip install -r requirements.txt
python app/app.py
```

---

## 🚀 Deployment

* HuggingFace Spaces
* Render / Docker

---

## 💡 Future Improvements

* Transformer models (BERT / RoBERTa)
* Multilingual support
* Emoji-aware embeddings

---

## 👨‍💻 Author

**Suraj Kumar**
Full Stack + AI Developer

---

## 📬 Feedback

Feel free to open issues or contribute 🚀
