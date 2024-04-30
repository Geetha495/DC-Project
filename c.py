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
timeout = 2000
# Model train thread and send grads and Updates using all grads
# Continuosly receives msgs 

'''
-> Each epoch is considered as tick
-> For every round send model params from all processes
-> After receiving all models, update model
-> After receiving all safes, move to next epoch
'''

def trainer(id, adj, data, num_model_received, num_safe_received, num_ack_received,recved_models):
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
        selects = random.sample([i for i in range(num_procs) if i != id], num_selects)
        selects.append(id)
        for pid in selects:
                sockets[id][pid].send_pyobj({'sender_id': id, 'tick': e,'type':'model_msg', 'params':model.state_dict()})
        for pid in range(num_procs):
            if pid not in selects:
                sockets[id][pid].send_pyobj({'sender_id': id, 'tick': e,'type':'model_msg', 'params':None})

        while num_model_received[0] < num_procs:
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
        num_model_received[0] = 0
        
        while num_ack_received[0] < num_procs:
            pass

        num_ack_received[0] = 0

        for pid in range(num_procs):
            sockets[id][pid].send_pyobj({'type': 'safe_msg', 'sender_id': id})

        while num_safe_received[0] < num_procs:
            pass

        num_safe_received[0] = 0
        
    torch.save(model.state_dict(), f"models/model_{id}.pt")
    
    for pid in range(num_procs):
        sockets[id][pid].send_pyobj({'type': 'marker_msg', 'sender_id': id})
    

def receiver(ctx, id, num_model_received, num_ack_received, num_safe_received, recved_models):
    '''
        -> Each epoch is considered as tick
        -> For every round, receive model params from all processes and update shared variable num_model_received and send acks instantly
        -> receive acks 
        -> after receiving in_nodes number of acks, send safe
        -> after receiving safe from all, update shared var num_safe
    '''
    # Create a ZMQ socket instance - consumer will use PULL socket
    consumer_socket = ctx.socket(zmq.PULL)
    consumer_socket.bind(f"inproc://#{id}")
    # consumer_socket.setsockopt(zmq.RCVTIMEO, timeout)

    # Receive messages
    num_markers_received = 0
    while num_markers_received < num_procs:
        msg = consumer_socket.recv_pyobj()
        if msg['type'] == 'model_msg':
            if msg['params'] is not None:
                recved_models.append(msg['params'])
                comm_costs[id] += 1
            num_model_received[0] += 1
            sockets[id][msg['sender_id']].send_pyobj({'type': 'ack_msg', 'sender_id': id})
                        
        elif msg['type'] == 'ack_msg':
            num_ack_received[0] += 1
            
        elif msg['type'] == 'safe_msg':
            num_safe_received[0] += 1
        
        elif msg['type'] == 'marker_msg':
            num_markers_received += 1

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
    num_ack_received = [[0] for _ in range(num_procs)]
    num_safe_received = [[0] for _ in range(num_procs)]
    num_model_received = [[0] for _ in range(num_procs)]
    for i in range(num_procs):
        adj = []
        for j in range(num_procs):
            adj.append(j)
            in_nodes[j] += 1
        adjlist.append(adj)

    
    init_sockets(adjlist)
    threads = []
    for i in range(num_procs):
        threads.append(threading.Thread(target=trainer, args=(i, adjlist[i], data[i], num_model_received[i], num_safe_received[i], num_ack_received[i],recved_models[i])))
        threads.append(threading.Thread(target=receiver, args=(context, i, num_model_received[i], num_ack_received[i], num_safe_received[i], recved_models[i])))
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