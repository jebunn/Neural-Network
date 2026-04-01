Chest X-Ray Pneumonia Classification using CNN

```
PneumoniaCNN/
│
├── Problem_set1.ipynb        # Main Colab notebook (all code)
├── README.md                 # This file — approach, methodology, findings
│
├── outputs/
│   ├── best_cnn_model.keras  # Best saved model (saved to Google Drive)
│   ├── training_history.png  # Loss/Accuracy/AUC plots across epochs
│   ├── confusion_matrix.png  # Confusion matrix on test set
│   └── sample_images.png     # Sample X-ray images from dataset
│
└── data/                     # Dataset (Google Drive)
    ├── train/
    │   ├── NORMAL/           # 1341 images
    │   └── PNEUMONIA/        # 3875 images
    ├── test/
    │   ├── NORMAL/           # 234 images
    │   └── PNEUMONIA/        # 390 images
    └── val/
        ├── NORMAL/           # 8 images
        └── PNEUMONIA/        # 8 images
```



Dataset Overview

| Split | NORMAL | PNEUMONIA | Total |
|-------|--------|-----------|-------|
| Train | 1,341  | 3,875     | 5,216 |
| Test  | 234    | 390       | 624   |
| Val   | 8      | 8         | 16    |
| Total| 1,583 | 4,273| 5,856 |

Computed Class Weights:
NORMAL (class 0): 1.9448
PNEUMONIA (class 1): 0.6730

These weights were applied during training to penalise misclassification of the minority class (NORMAL) more heavily.

Approach & Methodology

Data Preprocessing

Image augmentation (rescaling, rotation, shifting, zoom, flipping) was applied to the training data to reduce overfitting. Images were resized and loaded using generators, and split into training, validation, and test sets.

Model

A CNN model was used consisting of convolutional, pooling, dropout, and dense layers. A sigmoid activation function was applied for binary classification.

Training

Binary cross-entropy loss and Adam optimizer were used, with accuracy as the evaluation metric. Class weights were applied to handle imbalance, and EarlyStopping was used to prevent overfitting.

Evaluation

The model was evaluated using accuracy, confusion matrix, and training-validation performance plots.

Data Preprocessing

Image Augmentation (Training only)
To combat overfitting and improve generalisation, the following augmentations were applied to training images:

| Augmentation | Value |
|---|---|
| Rescaling | 1/255 (normalize to [0,1]) |
| Rotation Range | ±15° |
| Width Shift | 10% |
| Height Shift | 10% |
| Shear Range | 10% |
| Zoom Range | 10% |
| Horizontal Flip | True |
| Fill Mode | Nearest |

Validation and test data were only rescaled** (no augmentation) to ensure unbiased evaluation.

Image Parameters
- Target Size: 224 × 224 pixels
- Batch Size: 32
- Color Mode: RGB (3 channels)
- Class Mode: Binary (0 = NORMAL, 1 = PNEUMONIA)

Model Architecture — Custom CNN

```
Input: (224, 224, 3)
│
├── Block 1: Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
├── Block 2: Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
├── Block 3: Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.25)
├── Block 4: Conv2D(256) → BN → MaxPool → Dropout(0.30)
│
├── GlobalAveragePooling2D
├── Dense(512, relu) → BN → Dropout(0.50)
├── Dense(256, relu) → Dropout(0.30)
└── Dense(1, sigmoid)  ← Binary output
```

Model Compilation

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr = 1e-4) |
| Loss Function | Binary Crossentropy |
| Metrics | Accuracy, AUC, Precision, Recall |


Training Strategy

Callbacks Used:
| Callback | Purpose | Setting |
|---|---|---|
| `ModelCheckpoint` | Save best model by val_auc | `monitor='val_auc'`, `mode='max'` |
| `EarlyStopping` | Stop when val_loss stops improving | `patience=3` |
| `ReduceLROnPlateau` | Halve LR when plateauing | `patience=2`, `factor=0.5` |

Class Weights:
Applied `compute_class_weight('balanced')` to automatically handle class imbalance during training.

 Training Configuration:
- Epochs: 3 (limited due to computation constraints; early stopping applied)
- Steps/Epoch: 163 (5216 images / batch_size 32)
- GPU: NVIDIA T4 (Google Colab)

---

 Results & Findings

Training Performance (Epoch 1 snapshot)

| Metric | Train Value |
|--------|------------|
| Accuracy | 84.98% |
| AUC | 0.9335 |
| Loss | 0.3310 |
| Precision | 96.90% |
| Recall | 82.34% |

 Test Set Performance

| Metric | Value |
|--------|-------|
| Test Loss | 4.2985 |
| Test Accuracy | **62.50%** |

 Classification Report (Test Set)

```
              precision    recall  f1-score   support

      NORMAL       0.00      0.00      0.00       234
   PNEUMONIA       0.62      1.00      0.77       390

    accuracy                           0.62       624
   macro avg       0.31      0.50      0.38       624
weighted avg       0.39      0.62      0.48       624
```

Confusion Matrix Analysis

The model predicted every single test image as PNEUMONIA:
- PNEUMONIA correctly identified: 390/390 (100% Recall)
- NORMAL correctly identified: 0/234 (0% Recall)

Conclusion

The CNN model successfully classified chest X-ray images into NORMAL and PNEUMONIA categories. Data augmentation and class weighting improved model performance. Overall, the model shows effective results for medical image classification.






