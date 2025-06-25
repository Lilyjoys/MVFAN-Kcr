import torch
import torch.nn as nn
import torch.nn.functional as F


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
            torch.tensor(query.shape[-1]).float()
        )
        attention_weights = self.softmax(attention_scores)
        attention_output = torch.matmul(attention_weights, value)
        return attention_output


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Input shape: (batch_size, 300), treated as 1D sequence
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.pool3 = nn.MaxPool1d(2)

        # After 3 poolings (factor 2 each), input length reduces: 300 -> 150 -> 75 -> 37 (approx)
        self.fc1 = nn.Linear(512 * 37, 256)
        self.bn4 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 64)
        self.bn6 = nn.BatchNorm1d(64)

        self.output = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.46)

        self.attention = Attention(64, 64)

    def forward(self, x):
        # Reshape input for 1D convolution: (batch_size, 1, 300)
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout(x)

        x = F.relu(self.bn6(self.fc3(x)))
        x = self.dropout(x)

        # Apply attention mechanism
        x = self.attention(x)

        # Output layer with sigmoid activation for binary classification
        x = torch.sigmoid(self.output(x))
        return x
