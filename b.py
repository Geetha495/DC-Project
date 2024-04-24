import zmq
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Define the number of peers
NUM_PEERS = 3

# Define model parameters
learning_rate = 0.01
n_epochs = 10

# Define a simple neural network model
class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)

def train_local_model(X_local, y_local, model, optimizer, criterion):
    """Train a local model."""
    for epoch in range(n_epochs):
        inputs = torch.tensor(X_local, dtype=torch.float32)
        labels = torch.tensor(y_local, dtype=torch.float32)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

def send_model_parameters(socket, model):
    """Send model parameters to a peer."""
    params = model.state_dict()
    print('send')
    socket.send_pyobj(params)

def receive_model_parameters(socket, model):
    """Receive model parameters from a peer."""
    params = socket.recv_pyobj()
    model.load_state_dict(params)

def aggregate_models(models):
    """Aggregate models using Federated Averaging."""
    aggregated_model = {}
    for key in models[0].state_dict():
        aggregated_model[key] = torch.mean(torch.stack([model.state_dict()[key] for model in models]), dim=0)
    return aggregated_model

def main():
    input_size = X.shape[1]
    # Initialize ZeroMQ context
    context = zmq.Context()

    # Create a socket for each peer
    sockets = [context.socket(zmq.REP) for _ in range(NUM_PEERS)]
    for i, socket in enumerate(sockets):
        port = 5555 + i
        socket.bind(f"tcp://127.0.0.1:{port}")

    # Initialize models and optimizer
    models = [Model(input_size) for _ in range(NUM_PEERS)]
    optimizers = [optim.SGD(model.parameters(), lr=learning_rate) for model in models]
    criterion = nn.BCEWithLogitsLoss()

    # Train local models
    for i in range(NUM_PEERS):
        train_local_model(X, y, models[i], optimizers[i], criterion)

    # Synchronize models
    for i, socket in enumerate(sockets):
        for j, other_socket in enumerate(sockets):
            if i != j:
                send_model_parameters(socket, models[j])
                receive_model_parameters(other_socket, models[i])

    # Aggregate models using Federated Averaging
    aggregated_model_params = aggregate_models(models)

    # Update models with aggregated parameters
    for model in models:
        model.load_state_dict(aggregated_model_params)

    # Output aggregated model parameters
    print("Aggregated Model Parameters:")
    print(aggregated_model_params)

if __name__ == "__main__":
    main()
