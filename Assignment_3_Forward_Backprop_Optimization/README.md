# Assignment 3 - Forward & Backpropagation with Learning Rate and Epoch Optimization

> **BRACT's Vishwakarma Institute of Technology, Pune**  
> Department of Computer Science & Engineering (Artificial Intelligence)  
> **Practical No. 3 | Deep Learning (TY SEM-1, 2026-27)**  
> **Student**: Pushkar Patankar | **PRN**: 12414995 | **Roll No.**: 70 | **Division E, Batch 3**  

---

## 1. Problem Statement & Objectives

Implement **forward propagation** and **backpropagation** algorithms using TensorFlow/Keras on the MNIST handwritten digit classification dataset. Perform a rigorous empirical study to analyze the sensitivity of model convergence and performance with respect to:
1. **Learning Rate ($\eta$)**: Compare convergence behavior for $\eta \in \{0.1, 0.01, 0.001\}$.
2. **Number of Training Epochs**: Compare model generalization across $E \in \{5, 10, 20\}$.
3. **Loss & Accuracy Learning Curves**: Monitor training vs. validation loss/accuracy across epochs to identify overfitting regimes.

---

## 2. Theoretical Foundations: Forward & Backpropagation

### 1. Forward Propagation
Given an input vector $\mathbf{x} \in \mathbb{R}^{784}$:
$$\mathbf{z}^{[1]} = \mathbf{W}^{[1]} \mathbf{x} + \mathbf{b}^{[1]}, \quad \mathbf{a}^{[1]} = \text{ReLU}\left(\mathbf{z}^{[1]}\right)$$
$$\mathbf{z}^{[2]} = \mathbf{W}^{[2]} \mathbf{a}^{[1]} + \mathbf{b}^{[2]}, \quad \mathbf{a}^{[2]} = \text{ReLU}\left(\mathbf{z}^{[2]}\right)$$
$$\mathbf{z}^{[3]} = \mathbf{W}^{[3]} \mathbf{a}^{[2]} + \mathbf{b}^{[3]}, \quad \mathbf{\hat{y}} = \text{Softmax}\left(\mathbf{z}^{[3]}\right)$$

### 2. Loss Computation (Sparse Categorical Cross-Entropy)
$$\mathcal{L} = -\sum_{k=1}^{10} y_k \log(\hat{y}_k)$$

### 3. Backpropagation (Gradient Descent via Chain Rule)
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}} \cdot \left(\mathbf{a}^{[l-1]}\right)^T$$
$$\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \eta \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}}$$

---

## 3. Neural Network Architecture Specification

- **Input Dimension**: $784$ ($28 \times 28$ flattened grayscale pixels)
- **Hidden Layer 1**: $128$ units (ReLU)
- **Hidden Layer 2**: $64$ units (ReLU)
- **Output Layer**: $10$ units (Softmax)
- **Loss Function**: `sparse_categorical_crossentropy`
- **Optimizer**: Adam

---

## 4. Experimental Results & Analysis

### 1. Learning Rate Comparison (5 Epochs)
| Learning Rate ($\eta$) | Test Accuracy | Convergence Behavior |
|:---:|:---:|:---|
| **$0.001$ (Optimal)** | **$\sim 87.8\%$ - $97.5\%$** | Smooth, steady gradient updates with optimal convergence |
| **$0.01$ (Moderate)** | **$\sim 85.7\%$** | Fast initial progress but slight sub-optimal local oscillations |
| **$0.1$ (Excessive)** | **$\sim 50.1\%$** | Divergence / overshooting local minima; poor gradient stability |

### 2. Epochs Comparison ($\eta = 0.001$)
| Epochs ($E$) | Validation / Test Accuracy | Observation |
|:---:|:---:|:---|
| **5 Epochs** | $\sim 87.5\%$ | Fast convergence; slight under-fitting |
| **10 Epochs** | $\sim 87.2\%$ | Good plateau region |
| **20 Epochs** | **$97.4\%$ (Test)** | Peak test accuracy; slight validation divergence after epoch 12 |

---

## 5. Repository Structure

```
Assignment_3_Forward_Backprop_Optimization/
├── DL_Assignment_3_Pushkar_Patankar.ipynb  # Pre-executed Jupyter Notebook with LR/Epoch sweep & training curves
└── README.md                              # Detailed experimental report & mathematical formulation
```
