import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import numpy as np

# Attention module
class Attention(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(in_features, hidden_dim)
        self.key = nn.Linear(in_features, hidden_dim)
        self.value = nn.Linear(in_features, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(query.shape[-1]).float())
        attention_weights = self.softmax(attention_scores)
        attention_output = torch.matmul(attention_weights, value)
        return attention_output


class CNNAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNNAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.fc_input_size = self._get_conv_output_size(input_size)
        self.fc1 = nn.Linear(self.fc_input_size, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.attention = Attention(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def _get_conv_output_size(self, input_size):
        x = torch.randn(1, 1, input_size)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = self.attention(x)
        x = torch.sigmoid(self.fc2(x))
        return x


positive_data = pd.read_csv('onehot+anova+S.csv')
negative_data = pd.read_csv('onehot+anova-S.csv')
positive_data['label'] = 1
negative_data['label'] = 0
data = pd.concat([positive_data, negative_data], ignore_index=True)

X = data.drop('label', axis=1).values
y = data['label'].values
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
sn_list, sp_list, acc_list, mcc_list, auc_list, results = [], [], [], [], [], []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)

    input_size = X_train.shape[1]
    model = CNNAttention(input_size, hidden_size=128, output_size=1)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Fold {fold}, Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}')

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    predicted = (all_preds > 0.5).astype(int)

    tp = np.sum((predicted == 1) & (all_labels == 1))
    tn = np.sum((predicted == 0) & (all_labels == 0))
    fp = np.sum((predicted == 1) & (all_labels == 0))
    fn = np.sum((predicted == 0) & (all_labels == 1))

    sn = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn)
    mcc = matthews_corrcoef(all_labels, predicted)
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    auc_score = auc(fpr, tpr)

    sn_list.append(sn)
    sp_list.append(sp)
    acc_list.append(acc)
    mcc_list.append(mcc)
    auc_list.append(auc_score)

    results.append({
        'Fold': fold,
        'Sn': sn,
        'Sp': sp,
        'Acc': acc,
        'Mcc': mcc,
        'AUC': auc_score
    })

    print(f"Fold {fold}:")
    print(f"  Sn: {sn:.4f}")
    print(f"  Sp: {sp:.4f}")
    print(f"  Acc: {acc:.4f}")
    print(f"  Mcc: {mcc:.4f}")
    print(f"  AUC: {auc_score:.4f}\n")
    fold += 1


results.append({
    'Fold': 'Average',
    'Sn': np.mean(sn_list),
    'Sp': np.mean(sp_list),
    'Acc': np.mean(acc_list),
    'Mcc': np.mean(mcc_list),
    'AUC': np.mean(auc_list)
})

results_df = pd.DataFrame(results)
results_df.to_csv('cnn_attention_5_fold.csv', index=False)
print("Results saved to cnn_attention_5_fold.csv")
