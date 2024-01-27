import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from math import sqrt
import numpy as np
from sklearn.model_selection import KFold


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, dropout_rate=0.5):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=128),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=128, out_features=64),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=64, out_features=32),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=32, out_features=32),
            nn.ELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=32, out_features=1)
        )

    def forward(self, x):
        return self.layers(x)
    
class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_dropout_rate=0.5, fc_dropout_rate=0.5):
        super(RecurrentNeuralNetwork, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear_layers = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ELU(),
            nn.Dropout(p=fc_dropout_rate),
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Dropout(p=fc_dropout_rate),
            nn.Linear(32, 1)
        )
        self.rnn_dropout = nn.Dropout(p=rnn_dropout_rate)

    def forward(self, x):
        output, _ = self.rnn(x)
        output = output[:, -1, :] if len(output.shape) == 3 else output

        # Apply dropout to the recurrent layer
        output = self.rnn_dropout(output)
        
        # Feed the output through the linear layers
        output = self.linear_layers(output)

        return output



def k_fold_training(X, y, train_index, val_index, model, optimizer, criterion,epoch_rmse):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

    # Create Dataset & DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Training loop
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Evaluation loop
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        val_rmse = sqrt(val_loss.item())
        epoch_rmse += val_rmse

    return model, epoch_rmse

def train_neural_network(X, y, input_size, num_epochs=20, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    epoch_rmse_list = []

    # Instantiate the neural network
    model = NeuralNetwork(input_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(num_epochs):
        epoch_rmse = 0

        for train_index, val_index in kf.split(X):
            model, epoch_rmse = k_fold_training(X, y, train_index, val_index, model, optimizer, criterion, epoch_rmse)

        # Calculate and store the average RMSE for the epoch
        epoch_rmse /= n_splits
        epoch_rmse_list.append(epoch_rmse)

        print(f"Epoch {epoch + 1}/{num_epochs}, Average Validation RMSE: {epoch_rmse}")

    return model

def predict_neural_network(model, X_test):
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
        y_pred = model(X_test_tensor)
        y_pred_np = y_pred.numpy().flatten()

    return y_pred_np

