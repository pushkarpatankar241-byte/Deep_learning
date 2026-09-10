# Assignment 6 - Tomato Leaf Disease Classification using Custom 4-Block CNN

> **BRACT's Vishwakarma Institute of Technology, Pune**  
> Department of Computer Science & Engineering (Artificial Intelligence)  
> **Practical No. 6 | Deep Learning (TY SEM-1, 2026-27)**  
> **Student**: Pushkar Patankar | **PRN**: 12414995 | **Roll No.**: 70 | **Division E, Batch 3**  

---

## 1. Executive Summary & Problem Statement

Plant leaf diseases present a substantial threat to global food security and agricultural productivity. Tomato (*Solanum lycopersicum*) is among the most widely cultivated crops worldwide, but it is highly susceptible to various fungal, bacterial, and viral pathogens. Traditional identification methods rely on manual visual inspection by agricultural experts, which suffers from limited scalability, diagnostic subjectivity, and high latency.

In this assignment, an end-to-end Computer Vision deep learning pipeline is designed, trained, and evaluated for automated multi-class classification of tomato leaf diseases using a custom **4-Block Convolutional Neural Network (CNN)** built with TensorFlow and Keras. The model classifies input leaf images into **10 target classes** (9 distinct disease categories and 1 healthy category).

---

## 2. Dataset Description & Exploratory Data Analysis (EDA)

The dataset consists of **14,529 high-resolution RGB images** categorized into 10 distinct classes:

| Class Index | Disease / Condition Name | Category | Image Count | Distribution (%) | Key Visual Symptoms |
|:---:|:---|:---:|:---:|:---:|:---|
| **0** | `Tomato___Bacterial_spot` | Bacterial | 1,702 | 11.71% | Small, dark water-soaked spots with yellow chlorotic halos |
| **1** | `Tomato___Early_blight` | Fungal | 800 | 5.51% | Concentric ring target-like lesions on older, lower foliage |
| **2** | `Tomato___Late_blight` | Oomycete | 1,527 | 10.51% | Large, irregular pale green/brown water-soaked lesions |
| **3** | `Tomato___Leaf_Mold` | Fungal | 761 | 5.24% | Pale yellow spots on upper leaf surface, olive mold beneath |
| **4** | `Tomato___Septoria_leaf_spot` | Fungal | 1,417 | 9.75% | Circular dark brown spots with gray/tan centers & black pycnidia |
| **5** | `Tomato___Spider_mites Two-spotted_spider_mite` | Pest | 1,341 | 9.23% | Yellow speckling, bronze discoloration, fine webbing |
| **6** | `Tomato___Target_Spot` | Fungal | 1,123 | 7.73% | Small brown spots expanding into target-like concentric rings |
| **7** | `Tomato___Tomato_Yellow_Leaf_Curl_Virus` | Viral | 4,286 | 29.50% | Severe upward leaf curling, chlorosis/yellowing, stunted growth |
| **8** | `Tomato___Tomato_mosaic_virus` | Viral | 299 | 2.06% | Mottled light/dark green mosaic patterns, leaf blistering |
| **9** | `Tomato___healthy` | Healthy | 1,273 | 8.76% | Clean, smooth green foliage without lesions or chlorosis |
| **Total** | **10 Classes** | - | **14,529** | **100.00%** | - |

### Data Partitioning
The dataset was split using `splitfolders` with a reproducible seed (`seed=42`) into:
- **Training Set (70%)**: 10,166 images (for weight optimization)
- **Validation Set (15%)**: 2,175 images (for validation loss monitoring & checkpointing)
- **Testing Set (15%)**: 2,188 images (reserved for final unbiased evaluation)

---

## 3. Data Preprocessing & Augmentation Strategy

To prevent overfitting and enable invariance to real-world agricultural capture conditions (lighting variations, camera angles, leaf orientations):

1. **Spatial Resizing & Normalization**:
   $$\mathbf{X}_{\text{norm}} = \frac{\mathbf{X}_{\text{raw}}}{255.0} \in [0.0, 1.0], \quad \text{Shape: } (224, 224, 3)$$

2. **Real-Time Data Augmentation** (applied on Training Generator):
   - **Rotation Range**: $\pm 20^\circ$ (simulates camera orientation tilts)
   - **Width & Height Shifts**: $\pm 20\%$ (simulates off-center framing)
   - **Shear Range**: $0.2$ (simulates perspective distortion & leaf curvature)
   - **Zoom Range**: $\pm 20\%$ (simulates varying camera distances)
   - **Horizontal Flip**: `True` (simulates mirror foliage views)
   - **Fill Mode**: `'nearest'`

---

## 4. Custom 4-Block CNN Architecture

The network consists of **4 Feature Extraction Blocks** with increasing filter depths ($32 \to 64 \to 128 \to 256$), followed by a **2-layer Dense Classification Head** with Dropout regularization:

```
Input (224 × 224 × 3)
   │
   ├─► Block 1: Conv2D(32, 3×3, ReLU)  ──► MaxPool2D(2×2)  ──► (111 × 111 × 32)
   ├─► Block 2: Conv2D(64, 3×3, ReLU)  ──► MaxPool2D(2×2)  ──► (54 × 54 × 64)
   ├─► Block 3: Conv2D(128, 3×3, ReLU) ──► MaxPool2D(2×2)  ──► (26 × 26 × 128)
   ├─► Block 4: Conv2D(256, 3×3, ReLU) ──► MaxPool2D(2×2)  ──► (12 × 12 × 256)
   │
   ├─► Flatten (Vector length: 36,864)
   ├─► Dense(512, ReLU) + Dropout(0.5)
   ├─► Dense(256, ReLU) + Dropout(0.3)
   └─► Dense(10, Softmax) ──► Output Probabilities
```

### Layer-by-Layer Parameter Breakdown

| # | Layer Name | Type | Kernel / Pool | Output Feature Map | Param # |
|:---:|:---|:---:|:---:|:---:|:---:|
| 0 | `input_layer` | Input | - | $(224, 224, 3)$ | 0 |
| 1 | `conv2d_1` | Conv2D | $3 \times 3$ | $(222, 222, 32)$ | 896 |
| 2 | `max_pooling2d_1` | MaxPool2D | $2 \times 2$ | $(111, 111, 32)$ | 0 |
| 3 | `conv2d_2` | Conv2D | $3 \times 3$ | $(109, 109, 64)$ | 18,496 |
| 4 | `max_pooling2d_2` | MaxPool2D | $2 \times 2$ | $(54, 54, 64)$ | 0 |
| 5 | `conv2d_3` | Conv2D | $3 \times 3$ | $(52, 52, 128)$ | 73,856 |
| 6 | `max_pooling2d_3` | MaxPool2D | $2 \times 2$ | $(26, 26, 128)$ | 0 |
| 7 | `conv2d_4` | Conv2D | $3 \times 3$ | $(24, 24, 256)$ | 295,168 |
| 8 | `max_pooling2d_4` | MaxPool2D | $2 \times 2$ | $(12, 12, 256)$ | 0 |
| 9 | `flatten` | Flatten | - | $(36864)$ | 0 |
| 10 | `dense_1` | Dense | - | $(512)$ | 18,874,880 |
| 11 | `dropout_1` | Dropout ($p=0.5$) | - | $(512)$ | 0 |
| 12 | `dense_2` | Dense | - | $(256)$ | 131,328 |
| 13 | `dropout_2` | Dropout ($p=0.3$) | - | $(256)$ | 0 |
| 14 | `dense_3` | Dense (Output) | - | $(10)$ | 2,570 |
| **Total** | **Trainable Parameters** | | | | **19,397,194 (~73.99 MB)** |

---

## 5. Mathematical Formulations

### 1. 2D Convolution Operation
For an input tensor $\mathbf{X}$ and filter kernel $\mathbf{K}$ with bias $b$:
$$S(i, j, k) = \text{ReLU}\left( \sum_{m} \sum_{n} \sum_{c} X(i+m, j+n, c) \cdot K_k(m, n, c) + b_k \right)$$

### 2. Output Spatial Dimension Calculation
$$\text{Output Size} = \left\lfloor \frac{W_{\text{in}} - F + 2P}{S} \right\rfloor + 1$$
Where $W_{\text{in}}$ = input dimension, $F$ = kernel size ($3$), $P$ = padding ($0$), and $S$ = stride ($1$).

### 3. Softmax Activation Function
$$\hat{y}_c = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}, \quad c \in \{1, 2, \dots, 10\}$$

### 4. Categorical Cross-Entropy Loss
$$\mathcal{L}_{CE} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

---

## 6. Training Strategy & Hyperparameters

| Hyperparameter | Configuration | Rationale |
|:---|:---|:---|
| **Optimizer** | Adam ($\eta=10^{-3}, \beta_1=0.9, \beta_2=0.999$) | Fast adaptive gradient descent with momentum |
| **Loss Function** | `categorical_crossentropy` | Standard optimal loss for one-hot multi-class Softmax targets |
| **Batch Size** | 32 | Balanced gradient stability and memory throughput |
| **Epochs** | 25 (max with EarlyStopping) | Prevents over-training while allowing full convergence |
| **Early Stopping** | `monitor='val_loss'`, `patience=5`, `restore_best_weights=True` | Halts training when validation loss stops improving |
| **Model Checkpoint** | `tomato_cnn_best.keras`, `monitor='val_accuracy'`, `save_best_only=True` | Retains highest performing checkpoint |

---

## 7. Results & Key Findings

### Training Progression & Convergence
- **Epoch 1**: Training Acc: **42.20%** | Validation Acc: **52.60%** | Val Loss: **1.3019**
- **Epoch 3**: Training Acc: **70.93%** | Validation Acc: **67.95%** | Val Loss: **0.9598**
- **Epoch 5**: Training Acc: **79.85%** | Validation Acc: **75.68%** | Val Loss: **0.6720**
- **Epoch 7**: Training Acc: **83.70%** | Validation Acc: **87.77%** | Val Loss: **0.3522**
- **Epoch 8**: Training Acc: **84.83%** | Validation Acc: **89.66%** | Val Loss: **0.2993**

### Key Observations:
1. **Feature Hierarchy Progression**: Early layers (32 & 64 filters) detect edges, borders, and color variations. Deeper layers (128 & 256 filters) identify fine spot clusters, chlorosis mosaics, and viral leaf curl morphology.
2. **Impact of Regularization**: The combination of dual Dropout layers ($0.5$ and $0.3$) and real-time data augmentation effectively suppressed overfitting despite the 19.3M parameter capacity.
3. **Inference Latency**: Model inference on single test images operates in $<80\text{ ms}$, suitable for deployment in automated edge/mobile agricultural diagnostic systems.

---

## 8. Inference & Evaluation Pipeline

The notebook provides modular inference and evaluation functions:
- `predict_and_visualize_leaf(img_path, model, class_names)`: Generates dual-panel diagnostic visualizations featuring the classified leaf image alongside a horizontal probability distribution across all 10 classes.
- `evaluate_model_on_test_set(model, test_generator)`: Computes overall Test Accuracy, Per-Class Precision/Recall/F1-Score classification report, and renders a normalized Seaborn Confusion Matrix Heatmap.

---

## 9. Repository Structure

```
Assignment_6_CNN/
├── Tomato_CNN_Project.ipynb    # Complete pre-executed notebook with EDA, CNN, Training & Evaluation
└── README.md                   # Comprehensive experimental results & architectural documentation
```
