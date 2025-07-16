#  MVFAN-Kcr: A Multi-View Feature Fusion and Attention-Based Network for Lysine Crotonylation Site Identification

**MVFAN-Kcr** is a deep learning-based framework designed for accurate identification of **Lysine Crotonylation (Kcr) sites**. It integrates **multi-view feature fusion** (ESM2 embeddings, One-Hot encoding, ANOVA feature selection) with **attention-based neural networks** to enhance prediction performance.



##  Requirements

Install dependencies using:

```bash
pip install torch pandas numpy scikit-learn matplotlib tkinter imbalanced-learn biopython
```


##  Model Training 



###  Run Training

```bash
python train/DNN_ATT.py
python train/CNN_ATT.py
python train/MLP_ATT.py
```


##  Model Testing

###  Input Files

- `test-onehot+anova+.csv`: Positive test samples
- `test-onehot+anova-.csv`: Negative test samples

### Run Testing

```bash
python Test.py
```



## GUI-Based Online Predictor

An interactive GUI is available for users to make predictions without coding.

###  Launch GUI

```bash
python predicter.py
```

> A user-friendly interface opens to select input files, run prediction, and view results.



## Contact

If you have questions, feel free to contact: **zuoyun@jiangnan.edu.cn** or open an issue.
