# 🧬 MVFAN-Kcr: A Multi-View Feature Fusion and Attention-Based Network for Lysine Crotonylation Site Identification

MVFAN-Kcr is a deep learning-based framework that integrates **multi-view feature fusion** (ESM2 embeddings, One-Hot encoding, and ANOVA feature selection) and **attention-based neural networks** for accurate identification of **Lysine Crotonylation (Kcr) sites**.

> 🔬 **Key Models**:  
> - DNN-Attention (Deep Neural Network with Attention)  
> - CNN-Attention (Convolutional Neural Network with Attention)  
> - MLP-Attention (Multilayer Perceptron with Attention)  
>  
> 🎯 **Ensemble Voting Strategy**: Final predictions are obtained by **majority voting** across the three models.

---

## 📦 Requirements

Please install the following dependencies before running:

```bash
pip install torch pandas numpy scikit-learn matplotlib tkinter imbalanced-learn biopython
```

## 🏋️ Model Training (5-Fold Cross Validation)

### 🧪 Input Feature Files

- `onehot+anova+S.csv`: Positive training samples
- `onehot+anova-S.csv`: Negative training samples

### 📌 Training Commands

```bash

python train/DNN_ATT.py
python train/CNN_ATT.py
python train/MLP_ATT.py
```

## 🔬 Model Testing

### 🧪 Input Feature Files

- `test-onehot+anova+.csv`: Positive test samples
- `test-onehot+anova-.csv`: Negative test samples

### 📌 Run Testing Script

```bash
python Test.py
```

## 🖥️ GUI-Based Online Predictor

An integrated GUI allows **non-technical users** to run predictions directly using kcr_cv+.txt and kcr_cv-.txt files.

### ✅ Features

- Automatic extraction of:
  - **ESM2 protein embeddings**
  - **One-hot encoding**
- **ANOVA**-based feature selection (top informative features)
- Runs **DNN, CNN, and MLP Attention-based classifiers**
- Displays:
  - Per-model predictions
  - Final **majority voting ensemble**
- Exports prediction results as `.csv`

### 📌 Launch GUI

```bash
python predicter.py
```

> 🪟 This opens an interactive interface where you can select input files, run the predictor, and view results graphically. Final prediction results can be saved as a CSV file.


## 📬 Contact

For questions, feel free to contact [zuoyun@jiangnan.edu.cn] or open an issue.
