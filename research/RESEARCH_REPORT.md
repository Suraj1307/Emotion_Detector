# Emotion Classification (Research-Grade Report)

This document is designed to be populated with outputs from `research/research_suite.py`.

## 1. Problem Definition

### 1.1 Multi-class vs Multi-label
- Multi-class: each sample has exactly one label \(y \in \{1,\dots,C\}\).
- Multi-label: each sample can have multiple labels, represented by a binary vector \(\mathbf{y}\in\{0,1\}^C\).
- Current system: **multi-class** (single dominant emotion).
- Future extension: multi-label classification with sigmoid output + binary cross-entropy.

### 1.2 Task
Given text sequence \(x=(w_1,\dots,w_T)\), predict emotion class:
\[
\hat{y} = \arg\max_{c} P(y=c \mid x).
\]

## 2. Mathematical Formulation (Attention-BiLSTM-CNN)

1. Token embedding:
\[
\mathbf{e}_t = \text{Embedding}(w_t)\in\mathbb{R}^{d_e}
\]

2. BiLSTM contextual states:
\[
\mathbf{h}_t = [\overrightarrow{\mathbf{h}_t};\overleftarrow{\mathbf{h}_t]
\]

3. Additive attention:
\[
\mathbf{u}_t = \tanh(\mathbf{W}\mathbf{h}_t + \mathbf{b}),\quad
\alpha_t = \frac{\exp(\mathbf{u}_t^\top \mathbf{u})}{\sum_j \exp(\mathbf{u}_j^\top \mathbf{u})}
\]
\[
\mathbf{c} = \sum_t \alpha_t \mathbf{h}_t
\]

4. CNN features (local n-grams):
\[
\mathbf{f}^{(k)} = \text{MaxPool}(\text{Conv1D}_k(\mathbf{H}))
\]

5. Fusion + classifier:
\[
\mathbf{z} = [\mathbf{c};\mathbf{f}^{(3)};\mathbf{f}^{(5)}],\quad
\mathbf{p} = \text{softmax}(\mathbf{W}_o\mathbf{z}+\mathbf{b}_o)
\]

6. Objective:
\[
\mathcal{L} = -\sum_{i=1}^{N}\log p_{i,y_i}
\]

## 3. Dataset Statistics

Populate from `research/outputs/research_results.json`:
- Class distribution (train/test)
- Avg/median/p95 length
- Vocabulary size estimate
- Noise analysis: OOV ratio, symbol/noise indicators if raw text available

### 3.1 Class Distribution Table
| Label | Train Count | Train % | Test Count | Test % |
|---|---:|---:|---:|---:|
| ... | ... | ... | ... | ... |

### 3.2 Sequence Statistics
| Metric | Value |
|---|---:|
| Avg length | ... |
| Median length | ... |
| P95 length | ... |
| Vocab size | ... |
| OOV ratio | ... |

## 4. Emotion Co-occurrence Analysis

If raw multi-label data is available, compute top co-occurring emotion pairs:
| Pair | Count |
|---|---:|
| (joy, surprise) | ... |

If unavailable, report limitation explicitly and provide script for future runs.

## 5. Core Performance Metrics

Report:
- Accuracy
- Precision (macro)
- Recall (macro)
- F1 (macro)
- Per-class precision/recall/F1

### 5.1 Overall Metrics
| Metric | Value |
|---|---:|
| Accuracy | ... |
| Macro Precision | ... |
| Macro Recall | ... |
| Macro F1 | ... |

### 5.2 Per-class Metrics
| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| anger | ... | ... | ... |

## 6. Confusion Matrix Interpretation

Discuss dominant confusion channels:
- Which classes are most confused and why (semantic overlap, annotation ambiguity, sparse classes).
- Example: confusion between `fear` and `sadness` due to overlapping lexical signals.

## 7. Error Analysis (5 Misclassified Examples)

Provide 5 high-confidence errors from the generated list:
| Idx | True | Pred | Confidence | Text/Token Excerpt | Hypothesis |
|---:|---|---|---:|---|---|
| ... | ... | ... | ... | ... | ... |

## 8. Attention Weight Validation

Compare top-attended tokens with human emotional keywords:
- Keyword lexicon overlap score
- Qualitative examples where attention aligns/does not align

| Sample | Predicted Emotion | Top Attention Tokens | Human Keywords | Overlap |
|---|---|---|---|---:|
| ... | ... | ... | ... | ... |

## 9. Ablation Study

Ablations:
- Remove CNN
- Remove Attention
- Remove BiLSTM

| Variant | Accuracy | Macro-F1 | Delta vs Full |
|---|---:|---:|---:|
| Full model | ... | ... | 0 |
| No CNN | ... | ... | ... |
| No Attention | ... | ... | ... |
| No BiLSTM | ... | ... | ... |

## 10. Baseline Comparison

Baselines:
- Linear SVM
- LSTM
- BERT baseline (if available)

| Model | Accuracy | Macro-F1 | Inference ms/sample | Model Size (MB) |
|---|---:|---:|---:|---:|
| SVM | ... | ... | ... | ... |
| LSTM | ... | ... | ... | ... |
| Attention-BiLSTM-CNN | ... | ... | ... | ... |
| BERT (optional) | ... | ... | ... | ... |

## 11. Inference Time & Model Size

Report:
- Mean ms/sample
- Throughput samples/sec
- Serialized model size

## 12. Robustness Testing

Test categories:
- short text
- long text
- sarcasm
- mixed emotion

| Category | N | Accuracy | Macro-F1 | Notes |
|---|---:|---:|---:|---|
| Short | ... | ... | ... | ... |
| Long | ... | ... | ... | ... |
| Sarcasm | ... | ... | ... | ... |
| Mixed emotion | ... | ... | ... | ... |

## 13. Cross-domain Generalization

Train on source domain, test on different style/domain dataset.
If dataset unavailable, include protocol and expected setup.

| Source Train | Target Test | Accuracy | Macro-F1 |
|---|---|---:|---:|
| GoEmotions | Domain-X | ... | ... |

## 14. Hyperparameter Documentation

| Hyperparameter | Value |
|---|---|
| Embedding dim | 100 |
| BiLSTM hidden units | 64 |
| CNN filters | 64 |
| Kernel size | 3 |
| Dropout | 0.4 / 0.5 |
| Learning rate | 2e-4 |
| Batch size | 256 |
| Epochs | 8 |

## 15. Emotion Intensity (Optional Regression)

Add regression head:
\[
\hat{s}\in[0,1]
\]
with MSE/MAE evaluation for intensity labels if available.

## 16. Future Work

- Multi-label training with sigmoid outputs.
- Transformer integration (BERT/RoBERTa + attention fusion).
- Better calibration (temperature scaling).
- Domain adaptation and continual learning.
- Explainability validation with human annotation agreement.

## 17. Reproducibility

Run:
```bash
python research/research_suite.py
```

Artifacts generated:
- `research/outputs/research_results.json`
- `research/outputs/research_summary.md`
