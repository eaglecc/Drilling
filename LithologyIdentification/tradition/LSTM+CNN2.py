import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd


class LSTM_CNN_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, kernel_size):
        super(LSTM_CNN_Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # CNN layers
        self.conv1d = nn.Conv1d(hidden_size, hidden_size, kernel_size)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Reshape for CNN
        out = out.permute(0, 2, 1)  # Reshaping for CNN: (batch_size, hidden_size, seq_length)

        # CNN layer
        out = self.conv1d(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # Flatten
        out = out.reshape(out.size(0), -1)

        # Fully connected layer
        out = self.fc(out)
        return out


# Hyperparameters
input_size = 5
hidden_size = 64
num_layers = 2
num_classes = 9
kernel_size = 3
learning_rate = 0.001
num_epochs = 10

# Initialize model
model = LSTM_CNN_Model(input_size, hidden_size, num_layers, num_classes, kernel_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load CSV data
data = pd.read_csv('../data/train_data.csv')
data = data.fillna(0)  # 将数据中的所有缺失值替换为0

X = data.iloc[:, 4:9].values  # 特征列
y = data.iloc[:, 0].values  # 标签列

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
