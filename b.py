# load models from file and run test on several testing x
from model import Model
import torch
import torch.nn as nn
from config import *
import matplotlib.pyplot as plt

def main():
    x_test = 2 * torch.rand(10, 1)
    # for i in range(num_procs):
    i=0
    model = Model(1)
    model.load_state_dict(torch.load(f"models/model_{i}.pt"))
    y_test = model(x_test)
    loss = nn.MSELoss()(y_test, f(x_test))
    log_file = open(f"log.txt", "a")
    log_file.write(f"Loss: {loss}\n")
    plt.scatter(x_test, y_test.detach().numpy(), label=f"Model {i}", color="blue")
    plt.plot(x_test, f(x_test).detach().numpy(), label="True", color="red")


    model.load_state_dict(torch.load(f"models/model_server.pt"))
    y_test = model(x_test)
    loss = nn.MSELoss()(y_test, f(x_test))
    log_file = open(f"log.txt", "a")
    log_file.write(f"Loss server: {loss}\n")
    plt.scatter(x_test, y_test.detach().numpy(), label=f"Model server", color="green")
    plt.legend()
    plt.savefig(f"plots/models.png")

if __name__ == "__main__":
    main()