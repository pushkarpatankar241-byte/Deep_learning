# Assignment 2 - Multilayer Perceptron (MLP) for Iris Flower Classification

> **BRACT's Vishwakarma Institute of Technology, Pune**  
> Department of Computer Science & Engineering (Artificial Intelligence)  
> **Practical No. 2 | Deep Learning (TY SEM-1, 2026-27)**  
> **Student**: Pushkar Patankar | **PRN**: 12414995 | **Roll No.**: 70 | **Division E, Batch 3**  

---

## 1. Problem Statement & Objectives

Design and implement a **Multilayer Perceptron (MLP)** feedforward neural network to classify flower samples from the benchmark **Iris Dataset** into one of three species: *Iris setosa*, *Iris versicolor*, or *Iris virginica*. Evaluate the model using classification accuracy, a detailed per-class classification report, and a normalized confusion matrix.

---

## 2. Theoretical Foundations: Multilayer Perceptrons (MLPs)

A Multilayer Perceptron (MLP) is a class of feedforward artificial neural network (ANN) consisting of an input layer, one or more hidden layers with non-linear activation functions, and an output layer.

### Mathematical Formulation
1. **Hidden Layer Activation**:
   $$\mathbf{h}^{(1)} = \text{ReLU}\left(\mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}\right)$$
   $$\mathbf{h}^{(2)} = \text{ReLU}\left(\mathbf{W}^{(2)} \mathbf{h}^{(1)} + \mathbf{b}^{(2)}\right)$$
2. **Output Layer & Softmax Normalization**:
   $$\mathbf{z} = \mathbf{W}^{(3)} \mathbf{h}^{(2)} + \mathbf{b}^{(3)}$$
   $$\hat{y}_c = \frac{e^{z_c}}{\sum_{j=1}^{C} e^{z_j}}, \quad c \in \{0, 1, 2\}$$

3. **Loss Function (Cross-Entropy Loss)**:
   $$\mathcal{L} = -\sum_{c=1}^{3} y_c \log(\hat{y}_c)$$

---

## 3. Dataset Characteristics

| Parameter | Specification |
|:---|:---|
| **Dataset** | Fisher's Iris Dataset (`sklearn.datasets.load_iris`) |
| **Total Samples** | 150 instances (50 per class) |
| **Input Features (4)** | Sepal Length (cm), Sepal Width (cm), Petal Length (cm), Petal Width (cm) |
| **Target Classes (3)** | 0: *Setosa*, 1: *Versicolor*, 2: *Virginica* |
| **Split Ratio** | 80% Training (120 samples) / 20% Testing (30 samples) with `random_state=42` |
| **Feature Scaling** | `StandardScaler` ($\mu=0, \sigma=1$) |

---

## 4. Model Architecture & Hyperparameters

- **Input Dimension**: $4$ features
- **Hidden Layer 1**: $10$ neurons (ReLU)
- **Hidden Layer 2**: $10$ neurons (ReLU)
- **Output Layer**: $3$ neurons (Softmax)
- **Optimizer**: Adam
- **Max Iterations**: 1000
- **Random State**: 42

---

## 5. Experimental Results & Evaluation

### 1. Classification Accuracy
$$\text{Test Accuracy} = \mathbf{96.67\%} \quad (29 / 30 \text{ correct predictions})$$

### 2. Confusion Matrix
```
               Predicted Setosa  Predicted Versicolor  Predicted Virginica
Actual Setosa         10                  0                    0
Actual Versicolor      0                  8                    1
Actual Virginica       0                  0                   11
```

### 3. Classification Report
| Species | Precision | Recall | F1-Score | Support |
|:---|:---:|:---:|:---:|:---:|
| **Setosa** | 1.00 | 1.00 | 1.00 | 10 |
| **Versicolor** | 1.00 | 0.89 | 0.94 | 9 |
| **Virginica** | 0.92 | 1.00 | 0.96 | 11 |
| **Accuracy** | | | **0.97** | 30 |
| **Macro Avg** | 0.97 | 0.96 | 0.97 | 30 |
| **Weighted Avg** | 0.97 | 0.97 | 0.97 | 30 |

---

## 6. Repository Structure

```
Assignment_2_Iris_MLP/
├── DL_Assignment_2_Pushkar_Patankar.ipynb  # Pre-executed Jupyter Notebook with MLP classifier, training & confusion matrix
└── README.md                              # Detailed experimental report & mathematical formulation
```
