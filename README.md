# Deep Learning for Kidney Disease Detection

A Convolutional Neural Network (CNN) that classifies kidney CT scan images into four categories — **Cyst, Tumor, Stone, and Normal** — to support faster, automated screening of renal conditions.

Built as a 7th-semester B.E. project at KLE Technological University, Hubballi (2023–2024).

---

## Overview

Chronic kidney disease often goes undetected until an advanced stage, when treatment options are limited. This project uses a CNN to automatically classify CT scans, helping reduce diagnostic time in settings where nephrologists are scarce. The model reaches **~99% accuracy** on the validation set.

## Dataset

The model is trained on the **CT Kidney Dataset (Normal–Cyst–Tumor–Stone)** from Kaggle. Scans were originally sourced from PACS systems across hospitals in Dhaka, Bangladesh, where patients had already been diagnosed.

| Class  | Images |
|--------|--------|
| Normal | 5,077  |
| Cyst   | 3,709  |
| Tumor  | 2,283  |
| Stone  | 1,377  |
| **Total** | **12,446** |

> Update this section with the exact Kaggle dataset link before publishing.

## Approach

1. **Load & label** images from each class directory (Cyst=0, Normal=1, Stone=2, Tumor=3).
2. **Preprocess** — resize to 28×28, convert to arrays, normalize, and one-hot encode labels.
3. **Train/validation split** for the CNN.
4. **Train** the model and evaluate using accuracy, loss curves, F1 score, and a confusion matrix.
5. **Predict** the class of a single input scan.

## Model Architecture

A sequential CNN built with Keras/TensorFlow:

```
Conv2D(28, (3,3), relu, input_shape=(28,28,3))
MaxPooling2D((2,2))
Conv2D(64, (3,3), relu)
MaxPooling2D((2,2))
Flatten()
Dense(4, softmax)
```

## Results

- **Accuracy:** ~99% on the validation set
- Training loss decreases and converges across epochs
- Confusion matrix shows strong per-class performance, with minor confusion between Normal and Stone

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- OpenCV, scikit-image
- Matplotlib, Seaborn

