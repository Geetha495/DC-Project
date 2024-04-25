import zmq
import threading
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
from model import Model
from config import *
sockets = []
comm_costs = [0]*num_procs
context = zmq.Context()

# Model train thread and send grads and Updates using all grads
# Continuosly receives msgs 

def trainer(id, adj, data, in_nodes, num_grads, recved_models):
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
        for r in range(local_update_steps): 
            y_pred = model(data[0])
            # print(data[0],data[1])
            loss = nn.MSELoss()(y_pred, data[1])
            loss.backward()
            opt.step()
            opt.zero_grad()

        # get num_selects distinct random numbers from 0 to num_proc - 1 except id
        selected = random.sample([i for i in range(num_procs) if i != id], num_selects - 1)
        selected.append(id)
        not_selected = [i for i in range(num_procs) if i not in selected]
        for n in selected:
            neighbor_id = n
            sockets[id][neighbor_id].send_pyobj({'type':'param_msg', 'params':model.state_dict()})
        
        for n in not_selected:
            neighbor_id = n
            sockets[id][neighbor_id].send_pyobj({'type':'null_msg'})

        while num_grads[0] < in_nodes:
            pass
        
        with torch.no_grad():
            new_state_dict = {}
            for recved_model in recved_models:
                for param, value in recved_model.items():
                    new_state_dict[param] = value + new_state_dict.get(param,0.)
            
            for param in new_state_dict.keys():
                new_state_dict[param] /= num_procs
            
            model.load_state_dict(new_state_dict)                    
        
        recved_models.clear()
        num_grads[0] = 0
    torch.save(model.state_dict(), f"models/model_{id}.pt")
    # x_test = torch.tensor([1.0])
    # print(model(x_test))
    

def receiver(ctx, id, in_nodes, num_grads, recved_models):
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
            recved_models.append(msg['params'])
            comm_costs[id] += 1

        else:
            pass
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

    adjlist = []
    recved_models = [[] for _ in range(num_procs)]
    in_nodes = [0 for _ in range(num_procs)]
    num_grads = [[0] for _ in range(num_procs)]
    for i in range(num_procs):
        adj = []
        for j in range(num_procs):
            adj.append(j)
            in_nodes[j] += 1
        adjlist.append(adj)

    
    init_sockets(adjlist)
    threads = []
    for i in range(num_procs):
        t = threading.Thread(target=trainer, args=(i, adjlist[i], data[i], in_nodes[i], num_grads[i], recved_models[i]))
        threads.append(t)
        t2 = threading.Thread(target=receiver, args=(context, i, in_nodes[i], num_grads[i], recved_models[i]))
        threads.append(t2)
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    end = time.time()

    log_file = open("log.txt", "a")
    log_file.write("Peer-to-Peer:\n")
    log_file.write(f"num_procs: {num_procs}\n")
    log_file.write(f"Total time taken: {end-start}\n")
    for i in range(num_procs):
        log_file.write(f"Process {i} communication cost: {comm_costs[i]}\n")
    log_file.write("Total communication cost: {}\n".format(sum(comm_costs)))   

if __name__ == "__main__":
    main()