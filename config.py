import torch

num_epochs = 10
local_update_steps = 10
num_procs = 4
num_selects = num_procs

def f(x):
    return 4 + 3*x


data = []
    
x = 2 * torch.rand(100, 1)  # Random feature values between 0 and 2
y = f(x) + torch.randn(100, 1)  # Linear relationship with some random noise
total = len(x)
for i in range(num_procs):
    start = int((i)/num_procs*total)
    end = int((i+1)/num_procs*total)
    data.append([x[start:end], y[start:end]])
    