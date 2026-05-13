# 🩸 Blood Cell Type Prediction

An image classification project that identifies the type of blood cell present in a microscopy image. Built using transfer learning with a pretrained MobileNetV2 CNN on TensorFlow/Keras.

---

## 📋 Problem Description

Given an image of a blood cell, the model predicts which of 4 cell types it belongs to:

- **EOSINOPHIL**
- **LYMPHOCYTE**
- **MONOCYTE**
- **NEUTROPHIL**

This is a multi-class image classification task evaluated on accuracy and per-class metrics.

---

## 📁 Project Structure

```
blood-cell-type-prediction.ipynb    # Main notebook
```

**Expected data (Kaggle):**

```
/kaggle/input/blood-cells/dataset2-master/dataset2-master/images/
    ├── TRAIN/
    │   ├── EOSINOPHIL/
    │   ├── LYMPHOCYTE/
    │   ├── MONOCYTE/
    │   └── NEUTROPHIL/
    └── TEST/
        ├── EOSINOPHIL/
        ├── LYMPHOCYTE/
        ├── MONOCYTE/
        └── NEUTROPHIL/
```

---

## ⚙️ ML Pipeline

### 1. Data Loading
- Images resized to **224×224** (MobileNetV2 input size)
- MobileNetV2 preprocessing applied via `ImageDataGenerator`
- Training split: **80% train / 20% validation** (stratified by folder)
- Batch size: 32

### 2. Model Architecture — Transfer Learning

| Layer | Details |
|---|---|
| **Base** | MobileNetV2 pretrained on ImageNet (`include_top=False`, `pooling='avg'`) |
| **Frozen** | All MobileNetV2 weights are frozen |
| **Dense** | 128 units, ReLU activation |
| **Output** | 4 units, Softmax activation |

The model is compiled with:
- Optimizer: `Adam`
- Loss: `categorical_crossentropy`
- Metric: `accuracy`

### 3. Training
- Up to **100 epochs** with **EarlyStopping** (patience=3, monitors `val_loss`)
- Best weights are automatically restored

### 4. Evaluation
- **Test set** accuracy + confusion matrix + classification report
- **Validation set** accuracy + confusion matrix + classification report
- Training/validation loss curve plotted with Plotly

---

## 🔧 Dependencies

```bash
pip install numpy matplotlib seaborn plotly tensorflow scikit-learn
```

| Library | Usage |
|---|---|
| `tensorflow` | Model building, training, image generators |
| `numpy` | Array operations |
| `matplotlib` / `seaborn` | Confusion matrix visualization |
| `plotly` | Training loss curves |
| `scikit-learn` | Accuracy score, classification report |

---

## 🚀 Reproducing Results

1. Load the notebook on Kaggle with the [`blood-cells`](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) dataset
2. Run all cells in order
3. EarlyStopping typically halts training well before 100 epochs

---

## 📊 Target Metric

**Accuracy** on the held-out test set, complemented by per-class precision, recall, and F1-score from the classification report.
