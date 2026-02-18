# Research Evaluation Summary

## Core Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.1282 |
| Macro Precision | 0.1282 |
| Macro Recall | 0.1293 |
| Macro F1 | 0.0930 |
| Weighted F1 | 0.1464 |
| Inference ms/sample | 0.9729 |
| Model Size (MB) | 44.303 |

## Baseline Comparison

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| LinearSVM | 0.3903 | 0.1215 |
| LSTM | 0.3693 | 0.0771 |
| Attention-BiLSTM-CNN (Main) | 0.1282 | 0.0930 |

## Ablation Study

| Variant | Accuracy | Macro-F1 |
|---|---:|---:|
| no_cnn | 0.3693 | 0.0771 |
| no_attention | 0.4510 | 0.1460 |
| no_bilstm | 0.3693 | 0.0771 |

## Robustness by Length

| Bucket | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| short_0_10 | 3314 | 0.0845 | 0.0620 |
| medium_11_30 | 5363 | 0.1624 | 0.1129 |
| long_31_plus | 5 | 0.6000 | 0.1224 |

## Top 5 Misclassified Examples

| Idx | True | Pred | Confidence | Token Id Excerpt |
|---:|---|---|---:|---|
| 2728 | surprise | joy | 1.0000 | `[6, 46, 1989, 10, 1, 1529, 4, 1077, 45, 548, 18, 1, 1316, 0, 0, 0, 0, 0, 0, 0]` |
| 4286 | sadness | joy | 1.0000 | `[28, 128, 4, 893, 17, 92, 4, 391, 85, 6, 199, 2795, 39, 6, 46, 3222, 25, 74, 443, 0]` |
| 4434 | anger | joy | 1.0000 | `[6, 23, 8, 23, 8, 4, 832, 4, 677, 6, 99, 4, 832, 4, 33, 420, 25, 832, 666, 0]` |
| 5346 | anger | joy | 1.0000 | `[54, 4, 1526, 4, 33, 925, 4442, 1, 47, 6, 69, 16, 11, 5, 225, 131, 0, 0, 0, 0]` |
| 35 | surprise | joy | 0.9999 | `[4841, 5403, 11, 15, 15, 6108, 6109, 7, 49, 1199, 9, 87, 10, 66, 28, 4, 1150, 61, 574, 0]` |
