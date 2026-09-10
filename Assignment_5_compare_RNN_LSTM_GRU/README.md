# Assignment 5 — Comparative Study of RNN, LSTM, and GRU for Sequence Classification

> **BRACT's Vishwakarma Institute of Technology, Pune**
> Department of Computer Science & Engineering (Artificial Intelligence)
> **Practical No. 5 | Deep Learning (TY SEM-1, 2026-27)**
> **Student**: Pushkar Patankar | **PRN**: 12414995 | **Roll No.**: 70 | **Division E, Batch 3**

---

## Problem Statement
Implement and benchmark **SimpleRNN**, **LSTM**, and **GRU** architectures for **sequence classification** — binary sentiment analysis on the IMDB Movie Reviews dataset. Compare their performance using accuracy, precision, recall, F1-Score, ROC-AUC, confusion matrices, and computational profiling.

---

## Dataset
| Property | Details |
|:---|:---|
| **Dataset** | IMDB Movie Reviews (Keras built-in) |
| **Total Samples** | 50,000 labeled reviews |
| **Train / Validation / Test** | 20,000 / 5,000 / 25,000 |
| **Classes** | Binary: Positive (1) / Negative (0) |
| **Vocabulary** | Top 10,000 words |
| **Max Sequence Length** | 200 tokens (post-padded) |
| **Embedding Dimension** | 64 |

---

## Benchmark Results (25,000 Test Samples)

| Model | Test Loss | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC | Parameters |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SimpleRNN** | 0.6937 | 49.95% | 49.09% | 2.60% | 97.30% | 0.0494 | 0.5028 | 650,369 |
| **LSTM** | 0.5821 | 75.88% | 74.77% | 78.13% | 73.63% | 0.7641 | 0.7967 | 675,137 |
| **GRU** | **0.3758** | **82.66%** | **83.67%** | **81.15%** | **84.16%** | **0.8239** | **0.9135** | **667,073** |

---

## Key Findings
1. **SimpleRNN** collapses to ~50% accuracy on 200-token sequences due to vanishing gradients during BPTT.
2. **LSTM** recovers long-range context via Constant Error Carousel, achieving 75.88% accuracy and 0.7967 AUC.
3. **GRU outperforms LSTM** with 82.66% accuracy and 0.9135 AUC using fewer parameters and lower latency.

## Computational Profiling
| Model | Parameters | Training Time | Inference Latency / 1k |
|:---:|:---:|:---:|:---:|
| SimpleRNN | 650,369 | 17.99s | 81.9 ms |
| LSTM | 675,137 | 80.81s | 237.5 ms |
| GRU | 667,073 | 100.23s | 216.0 ms |

## Repository Contents
```
Assignment_5_compare_RNN_LSTM_GRU/
├── DL_Assignment_5_Pushkar_Patankar.ipynb  (20 cells, pre-executed)
├── models/
│   ├── simplernn_model.keras
│   ├── lstm_model.keras
│   └── gru_model.keras
└── README.md
```

## Training Configuration
| Hyperparameter | Value |
|:---|:---|
| Optimizer | Adam (lr=0.001) |
| Loss | Binary Cross-Entropy |
| Max Epochs | 6 |
| Batch Size | 128 |
| Early Stopping | patience=3 |
| Hidden Units | 64 |
| Seed | 42 |
