# Assignment 1 - Environment Configuration & MNIST Dataset Preprocessing

> **BRACT's Vishwakarma Institute of Technology, Pune**  
> Department of Computer Science & Engineering (Artificial Intelligence)  
> **Practical No. 1 | Deep Learning (TY SEM-1, 2026-27)**  
> **Student**: Pushkar Patankar | **PRN**: 12414995 | **Roll No.**: 70 | **Division E, Batch 3**  

---

## 1. Problem Statement & Objectives

Install and configure **TensorFlow** and **Keras** in the development environment (Google Colab / Local). Load the standard **MNIST Handwritten Digit Benchmark Dataset** and perform essential foundational data preprocessing, including:
1. Verifying environment library installations and runtime versions.
2. Partitioning and validating the predefined training and testing splits.
3. Inspecting spatial tensor dimensions, raw pixel formats, and intensity matrices.
4. Normalizing pixel intensity values to the continuous range $[0.0, 1.0]$.
5. Visualizing sample digits and analyzing class frequency distributions to verify dataset balance.

---

## 2. Theoretical Foundations: Why Data Preprocessing in Deep Learning?

### 1. Pixel Normalization (Min-Max Scaling)
Raw 8-bit grayscale images store pixel intensities as integer values:
$$X_{\text{raw}} \in [0, 255]$$

Passing large unnormalized inputs to neural networks causes steep activation gradients, leading to the **exploding gradient problem** or saturating non-linear activation functions (such as Sigmoid or Tanh). By scaling pixel intensities:
$$X_{\text{norm}} = \frac{X_{\text{raw}}}{255.0} \in [0.0, 1.0]$$
The input variance is standardized, leading to smoother loss surfaces, faster gradient descent convergence, and numerical stability.

### 2. Dataset Balance & Generalization
A uniform class distribution across training and testing partitions ensures that models trained on the dataset do not develop inductive bias toward majority classes, allowing fair evaluation across all 10 digit categories ($0$ through $9$).

---

## 3. Dataset Characteristics & Summary

| Parameter | Specification / Value |
|:---|:---|
| **Dataset Name** | Modified National Institute of Standards and Technology (MNIST) |
| **Data Modality** | Grayscale 2D Matrix Images ($28 \times 28$ pixels) |
| **Total Samples** | 70,000 images |
| **Training Partition** | 60,000 images (85.71%) |
| **Testing Partition** | 10,000 images (14.29%) |
| **Target Classes** | 10 Distinct Classes (Digits 0, 1, 2, 3, 4, 5, 6, 7, 8, 9) |
| **Raw Pixel Range** | $[0, 255]$ (8-bit unsigned integer) |
| **Normalized Range** | $[0.0, 1.0]$ (32-bit floating point) |

---

## 4. Preprocessing & Verification Pipeline

```
[keras.datasets.mnist]
          │
          ▼
1. Environment Verification       ──► TensorFlow: 2.20.0 | Keras: 3.13.2
          │
          ▼
2. Load Train & Test Splits       ──► Train: (60000, 28, 28) | Test: (10000, 28, 28)
          │
          ▼
3. Visual & Matrix Inspection     ──► Render 2x5 sample grid + 28x28 intensity heatmaps
          │
          ▼
4. Float32 Normalization          ──► X_norm = X_raw / 255.0 (Verify min: 0.0, max: 1.0)
          │
          ▼
5. Class Frequency Analysis       ──► Compute np.unique(y_train) & plot distribution bar chart
          │
          ▼
6. Pipeline Readiness             ──► Verified dataset ready for ANN/CNN deep learning models
```

---

## 5. Summary of Outputs & Results

```
============================================================
           MNIST DATASET PREPROCESSING SUMMARY
============================================================
TensorFlow Version   : 2.20.0
Keras Version        : 3.13.2
Training Data Shape  : (60000, 28, 28)
Testing Data Shape   : (10000, 28, 28)
Training Labels      : (60000,)
Testing Labels       : (10000,)
Raw Pixel Range      : 0 to 255
Normalized Range     : 0.0 to 1.0
Class Distribution   : Balanced across all 10 digits (~6,000 samples/class)
============================================================
```

---

## 6. Repository Structure

```
Assignment_1_MNIST_Preprocessing/
├── DL_Assignment_1_Pushkar_Patankar.ipynb  # Pre-executed Jupyter Notebook with environment checks, EDA & normalization
└── README.md                              # Comprehensive documentation and theoretical foundations
```
