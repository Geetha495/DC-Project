# load models from file and run test on several testing x
from model import Model
import torch
import torch.nn as nn
from config import *
import matplotlib.pyplot as plt

def main():
    # for i in range(num_procs):
    i=0
    model = Model(1)
    model.load_state_dict(torch.load(f"models/{model_file_p2p}"))
    with torch.no_grad():
        y_test = model(x_test)
        loss = nn.MSELoss()(y_test, f(x_test))
    log_file = open(f"log.txt", "a")
    log_file.write(f"Testing Loss serverless: {loss}\n")
   # plt.scatter(x_test, y_test.detach().numpy(), label=f"Model Peer-to-Peer", color="blue")
   # plt.plot(x_test, f(x_test).detach().numpy(), label="True", color="red")


    model.load_state_dict(torch.load(f"models/{model_file_server}"))
    with torch.no_grad():
        y_test = model(x_test)
        loss = nn.MSELoss()(y_test, f(x_test))
    log_file = open(log_filename, "a")
    log_file.write(f"Testing Loss server: {loss}\n")
    #plt.scatter(x_test, y_test.detach().numpy(), label=f"Model server", color="green")
    #plt.legend()
    #plt.savefig(f"plots/models.png")

if __name__ == "__main__":
    main()