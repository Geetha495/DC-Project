from model import *
import torch
import torch.nn as nn
from config import *
import matplotlib.pyplot as plt
from data import test_loader

def main():
    model = AlexNetMNIST()

    # load model from models/
    model.load_state_dict(torch.load(f"models/{model_file_p2p}"))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Overall Test Accuracy: {100 * correct / total} %')


if __name__ == "__main__":
    main()