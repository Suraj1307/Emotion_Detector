---
title: AffectLens AI: Emotion Intelligence Studio
emoji: 😄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.34.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# AffectLens AI: Emotion Intelligence Studio

This repository includes an attention-based BiLSTM-CNN emotion classifier and a
research-grade evaluation pipeline.

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
