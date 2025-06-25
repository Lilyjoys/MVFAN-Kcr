import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from DNN_Attention import DNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = DNN().to(device)
model.load_state_dict(torch.load('DNN_ATT-esm+anova1.pth', map_location=device))
model.eval()


positive_data_path = '../onehot+esm2/onehot+anova/test-onehot+anova+.csv'
negative_data_path = '../onehot+esm2/onehot+anova/test-onehot+anova-.csv'

# Read positive and negative datasets
positive_data = pd.read_csv(positive_data_path)
negative_data = pd.read_csv(negative_data_path)

X_positive = torch.tensor(positive_data.values, dtype=torch.float32)
X_negative = torch.tensor(negative_data.values, dtype=torch.float32)

y_positive = torch.ones(X_positive.size(0), dtype=torch.float32)
y_negative = torch.zeros(X_negative.size(0), dtype=torch.float32)

X_test = torch.cat((X_positive, X_negative), dim=0)
y_test = torch.cat((y_positive, y_negative), dim=0)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

y_true = []
y_pred = []
y_pred_proba = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        outputs = model(X_batch).squeeze()
        preds = torch.round(outputs)  # Predicted labels
        y_true.extend(y_batch.tolist())
        y_pred.extend(preds.tolist())
        y_pred_proba.extend(outputs.tolist())

result_df = pd.DataFrame({
    'Actual': y_true,
    'Predicted': y_pred,
    'Predicted_Probability': y_pred_proba
})
result_df.to_csv('CSV/DNN-ATT-esm+anova.csv', index=False)

# Calculate performance metrics
from sklearn.metrics import confusion_matrix, roc_auc_score
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
tn_t, fp_t, fn_t, tp_t = torch.tensor(tn), torch.tensor(fp), torch.tensor(fn), torch.tensor(tp)
sn = tp / (tp + fn)
sp = tn / (tn + fp)
acc = (tp + tn) / (tp + tn + fp + fn)
mcc = ((tp_t * tn_t) - (fp_t * fn_t)) / torch.sqrt((tp_t + fp_t) * (tp_t + fn_t) * (tn_t + fp_t) * (tn_t + fn_t))
mcc = mcc.item() if mcc.numel() == 1 else mcc
auc = roc_auc_score(y_true, y_pred_proba)

metrics_dict = {
    'Sensitivity (Sn)': [sn],
    'Specificity (Sp)': [sp],
    'Accuracy (Acc)': [acc],
    'Matthews Correlation Coefficient (MCC)': [mcc],
    'Area Under the ROC Curve (AUC)': [auc]
}

metrics_df = pd.DataFrame(metrics_dict)

metrics_df.to_csv('EST-DNN-ATT.csv', index=False)

print(f' {sn:.4f}, {sp:.4f},  {acc:.4f},  {mcc:.4f},  {auc:.4f}')