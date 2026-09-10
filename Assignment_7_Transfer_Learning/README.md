# Assignment 7 — Transfer Learning with Pre-trained CNNs

> **BRACT's Vishwakarma Institute of Technology, Pune**
> Department of Computer Science & Engineering (Artificial Intelligence)
> **Practical No. 7 | Deep Learning (TY SEM-1, 2026-27)**
> **Student**: Pushkar Patankar | **PRN**: 12414995 | **Roll No.**: 70 | **Division E, Batch 3**

---

## Problem Statement
Implement **Transfer Learning** using pre-trained CNN architectures — AlexNet, VGG-16, ResNet-50, and EfficientNet-B0 — on the MNIST dataset. Perform a rigorous comparative evaluation of fine-tuning strategies, convergence behavior, and classification performance.

---

## Dataset
| Property | Details |
|:---|:---|
| **Dataset** | MNIST Handwritten Digits |
| **Total Samples** | 70,000 images (60k train / 10k test) |
| **Classes** | 10 (digits 0–9) |
| **Input Shape** | 28×28 grayscale → resized to model input size |

---

## Models Compared
| Architecture | Pre-trained On | Parameters | Key Feature |
|:---:|:---:|:---:|:---|
| **AlexNet** | ImageNet | ~61M | Pioneer deep CNN, 5 conv layers |
| **VGG-16** | ImageNet | ~138M | Deep uniform 3×3 conv stacks |
| **ResNet-50** | ImageNet | ~25M | Residual skip connections |
| **EfficientNet-B0** | ImageNet | ~5.3M | Compound scaling, most efficient |

---

## Framework & Hardware
- **Language**: Python 3.13
- **Framework**: PyTorch with CUDA 12.4
- **GPU**: NVIDIA GeForce RTX 4060 Laptop (8GB GDDR6 VRAM)

---

## Repository Contents
```
Assignment_7_Transfer_Learning/
├── DL_Assignment_7_Transfer_Learning.ipynb  (pre-executed notebook)
└── README.md
```

> **Note**: MNIST binary data files (~45 MB) are not included in the repo.
> They are auto-downloaded by the notebook on first run via PyTorch's torchvision datasets.
