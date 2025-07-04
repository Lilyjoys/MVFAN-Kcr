import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import numpy as np


class Attention(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(in_features, hidden_dim)
        self.key = nn.Linear(in_features, hidden_dim)
        self.value = nn.Linear(in_features, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [batch_size, feature_dim]
        x = x.unsqueeze(1)  # -> [batch, 1, dim]
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k).float())
        weights = self.softmax(scores)
        out = torch.matmul(weights, value)
        return out.squeeze(1)  # back to [batch, dim]


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(880, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.output = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.46)
        self.attention = Attention(64, 64)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.attention(x)
        x = torch.sigmoid(self.output(x))
        return x


# 读取数据
positive_data_path = 'onehot+anova+S.csv'
negative_data_path = 'onehot+anova-S.csv'

positive_data = pd.read_csv(positive_data_path)
negative_data = pd.read_csv(negative_data_path)

positive_data['label'] = 1
negative_data['label'] = 0

data = pd.concat([positive_data, negative_data], ignore_index=True)

X = data.drop('label', axis=1).values
y = data['label'].values

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
sn_list = []
sp_list = []
acc_list = []
mcc_list = []
auc_list = []
results = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    model = MLP()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Fold {fold}, Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')


    model.eval()
    all_preds = []
    all_labels = []
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

    sn = tp / (tp + fn) if (tp + fn) != 0 else 0
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0
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


    fold += 1


avg_sn = np.mean(sn_list)
avg_sp = np.mean(sp_list)
avg_acc = np.mean(acc_list)
avg_mcc = np.mean(mcc_list)
avg_auc = np.mean(auc_list)


results.append({
        'Fold': fold,
        'Sn': sn,
        'Sp': sp,
        'Acc': acc,
        'Mcc': mcc,
        'AUC': auc_score
    })


results_df = pd.DataFrame(results)
results_df.to_csv('FIVE_mlp_old_attention.csv', index=False)
