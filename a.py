import zmq
import threading
import time
import torch
import torch.nn as nn
import torch.optim as optim

sockets = []
context = zmq.Context()
num_epochs = 50

class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.layers(x)

# Model train thread and send grads and Updates using all grads
# Continuosly receives msgs 

def trainer(id, adj, data, in_nodes, num_grads, recved_grads):
    '''
    model init
    for epochs
        train model for 1 epoch till grad comp
        for all procs p
            send grad
        
        while not recv grad from neighbors
            pass
        
        update model 
    '''
    model = Model(1)
    opt = optim.Adam(model.parameters(), lr=1e-2)
    num_procs = len(adj)
    for e in range(num_epochs):
        # zero grad
        with torch.no_grad():
            for param in model.parameters():
                param.grad = torch.zeros_like(param)
        
        y_pred = model(data[0])
        # print(data[0],data[1])
        loss = nn.MSELoss()(y_pred, data[1])
        loss.backward()


        for n in range(num_procs):
            neighbor_id = n
            grads = [param.grad for param in model.parameters()]
            sockets[id][neighbor_id].send_pyobj({'type':'param_msg', 'grads':grads})
             
        while num_grads[0] < in_nodes:
            pass
        
        with torch.no_grad():
            for param in model.parameters():
                param.grad = torch.zeros_like(param.grad)
            for recvgrad in recved_grads:
                for param, grad in zip(model.parameters(), recvgrad):
                    param.grad += grad

            for param in model.parameters():
                param.grad /= in_nodes
            lr = 0.01

            for param in model.parameters():
                param -= lr*param.grad
        
        
        recved_grads.clear()
        num_grads[0] = 0

    x_test = torch.tensor([1.0])
    print(model(x_test))
    

def receiver(ctx, id, in_nodes, num_grads, recved_grads):
    '''
    for epochs
        for all proc p
            recv grad

    '''
    # Create a ZMQ socket instance - consumer will use PULL socket
    consumer_socket = ctx.socket(zmq.PULL)
    consumer_socket.bind(f"inproc://#{id}")
    
    # Receive messages
    for _ in range(num_epochs*in_nodes):
        msg = consumer_socket.recv_pyobj()
        if msg['type'] == 'param_msg':
            recved_grads.append(msg['grads'])
            num_grads[0] += 1
        
        while num_grads[0] == in_nodes:
            pass

        
    consumer_socket.close()


def init_sockets(adj_list):
    for i in range(len(adj_list)):
        slist = []
        for j in adj_list[i]:
            skt = context.socket(zmq.PUSH)
            skt.connect(f"inproc://#{j}")
            slist.append(skt)
        sockets.append(slist)


def main():
    # Create a ZeroMQ context

    num_procs = 5
    adjlist = []
    recved_grads = [[] for _ in range(num_procs)]
    in_nodes = [0 for _ in range(num_procs)]
    num_grads = [[0] for _ in range(num_procs)]
    for i in range(num_procs):
        adj = []
        for j in range(num_procs):
            adj.append(j)
            in_nodes[j] += 1
        adjlist.append(adj)

    data = []
    
        
    x = 2 * torch.rand(100, 1)  # Random feature values between 0 and 2
    y = 4 + 3 * x + torch.randn(100, 1)  # Linear relationship with some random noise
    total = len(x)
    for i in range(num_procs):
        start = int((i)/num_procs*total)
        end = int((i+1)/num_procs*total)
        data.append([x[start:end], y[start:end]])
    
    init_sockets(adjlist)
    threads = []
    for i in range(num_procs):
        t = threading.Thread(target=trainer, args=(i, adjlist[i], data[i], in_nodes[i], num_grads[i], recved_grads[i]))
        threads.append(t)
        t.start()
        t2 = threading.Thread(target=receiver, args=(context, i, in_nodes[i], num_grads[i], recved_grads[i]))
        threads.append(t2)
        t2.start()


    for t in threads:
        t.join()
            

if __name__ == "__main__":
    main()