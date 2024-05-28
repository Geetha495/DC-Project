import torch
import argparse
from sklearn.model_selection import train_test_split

# Write a argumentParser for num_procs
parser = argparse.ArgumentParser()
parser.add_argument("--num_procs", "-n", type=int, default=4)
parser.add_argument("--num_epochs", "-e", type=int, default=10)
parser.add_argument("--local_update_steps", "-lu", type=int, default=10)
parser.add_argument("--log-file", "-l", type=str, default="log.txt")
parser.add_argument("--sparse", "-s", type=float, default=0)
parser.add_argument("--model-file-server", "-ms", type=str, default="model_server.pt")
parser.add_argument("--model-file-p2p", "-mp", type=str, default="model_p2p.pt")
args = parser.parse_args()

num_epochs = args.num_epochs
local_update_steps = args.local_update_steps
num_procs = args.num_procs
log_filename = args.log_file
sparsity_index = args.sparse
model_file_server = args.model_file_server
model_file_p2p = args.model_file_p2p
len_data = 1000

# def f(x):
#     return 4 + 3*x

# data = []


# x = 10 * torch.rand(len_data, 1)  # Random feature values between 0 and 10
# y = f(x) + torch.randn(len_data, 1)  # Linear relationship with some random noise

# x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, random_state=42)
# x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)


# total = len(x_train)
# for i in range(num_procs):
#     start = int((i)/num_procs*total)
#     end = int((i+1)/num_procs*total)
#     data.append([x_train[start:end], y_train[start:end]])

 