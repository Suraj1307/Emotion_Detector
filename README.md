---
title: "Topic 7: Attention-Based BiLSTM Emotion Classification"
emoji: 😄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.34.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# Topic 7: Emotion Classification in Social Media Using Attention-Based BiLSTM

This project addresses short, noisy social media emotion classification where key
emotional clues appear in fragmented, informal, or mixed-emotion text.

Core objective:
- Design an attention-based BiLSTM classifier that identifies emotionally relevant words.
- Improve recognition performance for tweets/Reddit-style text.
- Provide interpretable outputs with research-style evaluation and visualizations.

Research-paper alignment:
- Attention-enhanced sequence modeling
- Social media text robustness focus
- Evaluation with confusion matrix, ablations, baselines, and error analysis

## Research-Grade Additions

The `research/` folder includes:

- `research/research_suite.py`
  - Dataset statistics
  - Class distribution
  - Precision/Recall/F1/Macro-F1
  - Confusion matrix
  - 5 misclassified examples
  - Baseline comparison (SVM, LSTM)
  - Ablation study (no CNN / no Attention / no BiLSTM)
  - Inference time + model size
  - Robustness by sequence length buckets
  - Co-occurrence analysis (if raw multi-label data exists)

- `research/RESEARCH_REPORT.md`
  - Full research report structure with:
    - Mathematical model formulation
    - Multi-class vs multi-label definitions
    - Error analysis
    - Attention validation protocol
    - Cross-domain protocol
    - Hyperparameter documentation
    - Future work

## Run Research Suite

```bash
python research/research_suite.py
```

Generated outputs:

- `research/outputs/research_results.json`
- `research/outputs/research_summary.md`

Use these files to populate `research/RESEARCH_REPORT.md` tables.
