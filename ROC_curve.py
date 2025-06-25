import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Load model prediction results
data = pd.read_csv('final_predictions.csv')

y_true = data['Actual']

# Predicted probabilities from individual models
dnn_attn_probs = data['DNN_Predicted_Probability']
cnn_attn_probs = data['CNN_Predicted_Probability']
mlp_attn_probs = data['MLP_Predicted_Probability']

# Ensemble prediction using soft voting (average probabilities)
ensemble_probs = (dnn_attn_probs + cnn_attn_probs + mlp_attn_probs) / 3

# Create ROC plot
plt.figure(figsize=(10, 8))

# === ROC Curve: DNN-Attention ===
fpr_dnn, tpr_dnn, _ = roc_curve(y_true, dnn_attn_probs)
roc_auc_dnn = auc(fpr_dnn, tpr_dnn)
plt.plot(fpr_dnn, tpr_dnn, color='blue', lw=2,
         label=f'DNN-Attention (AUC = {roc_auc_dnn:.3f})')

# === ROC Curve: CNN-Attention ===
fpr_cnn, tpr_cnn, _ = roc_curve(y_true, cnn_attn_probs)
roc_auc_cnn = auc(fpr_cnn, tpr_cnn)
plt.plot(fpr_cnn, tpr_cnn, color='green', lw=2,
         label=f'CNN-Attention (AUC = {roc_auc_cnn:.3f})')

# === ROC Curve: MLP-Attention ===
fpr_mlp, tpr_mlp, _ = roc_curve(y_true, mlp_attn_probs)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)
plt.plot(fpr_mlp, tpr_mlp, color='red', lw=2,
         label=f'MLP-Attention (AUC = {roc_auc_mlp:.3f})')

# === ROC Curve: Ensemble (Soft Voting) ===
fpr_ensemble, tpr_ensemble, _ = roc_curve(y_true, ensemble_probs)
roc_auc_ensemble = auc(fpr_ensemble, tpr_ensemble)
plt.plot(fpr_ensemble, tpr_ensemble, color='darkorange', lw=3,
         label=f'Ensemble-Attention (AUC = {roc_auc_ensemble:.3f})')

# === Plot settings ===
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curves for Crotonylation Site Prediction\n(With Attention Mechanisms)')
plt.legend(loc="lower right")

# Save plot
plt.savefig('ROC_Curves_Attention_Models.png', dpi=300, bbox_inches='tight')
plt.show()

# Print AUC results
print("=== Model Performance Comparison ===")
print(f"DNN-Attention AUC: {roc_auc_dnn:.4f}")
print(f"CNN-Attention AUC: {roc_auc_cnn:.4f}")
print(f"MLP-Attention AUC: {roc_auc_mlp:.4f}")
print(f"Ensemble-Attention AUC: {roc_auc_ensemble:.4f}")
