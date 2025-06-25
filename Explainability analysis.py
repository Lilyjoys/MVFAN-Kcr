import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import shap
import matplotlib.pyplot as plt
import numpy as np


# Attention mechanism module
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


# MLP model with attention
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


# Load the pre-trained model
model = MLP()
model.load_state_dict(torch.load('DNN_ATT-esm+anova1.pth'))
model.eval()


# SHAP wrapper to use with Explainer
class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            tensor_X = torch.tensor(X, dtype=torch.float32)
            outputs = self.model(tensor_X)
            return outputs.detach().numpy()


# Load positive and negative test feature datasets
positive_data = pd.read_csv('../onehot+esm2/onehot+anova/test-onehot+anova+.csv')
negative_data = pd.read_csv('../onehot+esm2/onehot+anova/test-onehot+anova-.csv')
all_data = pd.concat([positive_data, negative_data])
X_all = all_data.values

# Create background set for SHAP (random subset)
background = X_all[np.random.choice(X_all.shape[0], 10, replace=False)]

# Instantiate SHAP explainer
explainer = shap.Explainer(ModelWrapper(model).predict, background)

# Compute SHAP values on a limited sample
sample_size = min(1000, len(X_all))
shap_values_all = explainer.shap_values(X_all[:sample_size])

feature_names = [f'feat_{i}' for i in range(880)]
plt.figure(figsize=(15, 15))
shap.summary_plot(shap_values_all, X_all[:sample_size], feature_names=feature_names, show=False)
plt.title("All Samples")
plt.tight_layout()
plt.show()

# # Optional: dependence plots for specific features
# for i in range(5):
#     shap.dependence_plot(f'feat_{i}', shap_values_all, X_all[:sample_size],
#                          feature_names=feature_names,
#                          title=f'Feature {i} Dependence')
