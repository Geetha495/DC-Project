import zmq
import threading
import time
import torch
import torch.nn as nn
import torch.optim as optim
import random
from model import Model
from config import *
sockets = [{} for _ in range(num_procs)]
comm_costs = [0]*num_procs
context = zmq.Context()

adj = [[] for _ in range(num_procs)]

# Model train thread and send grads and Updates using all grads
# Continuosly receives msgs 

'''
-> Each epoch is considered as tick
-> For every round send model params from all processes
-> After receiving all models, update model
-> After receiving all safes, move to next epoch
'''


def generate_connected_graph_adjacency_list(num_nodes):
    # Initialize an empty adjacency list
    #adj = {i: [] for i in range(num_nodes)}

    # Generate a random spanning tree
    tree_edges = [(i, random.randint(0, i-1)) for i in range(1, num_nodes)]
    
    # Add the tree edges to the adjacency list
    for edge in tree_edges:
        adj[edge[0]].append(edge[1])
        adj[edge[1]].append(edge[0])

    # Calculate the maximum number of additional edges
    max_additional_edges = int((num_nodes - 1) * (num_nodes - 2) / 2)
    num_additional_edges = int(sparsity_index * max_additional_edges)

    # Add additional random edges to make the graph less sparse
    edges_added = 0
    while edges_added < num_additional_edges:
        node1 = random.randint(0, num_nodes - 1)
        node2 = random.randint(0, num_nodes - 1)
        if node2 not in adj[node1] and node1 != node2:
            adj[node1].append(node2)
            adj[node2].append(node1)
            edges_added += 1

    #return adj


def trainer(id, data, num_model_received, num_safe_received, num_ack_received,recved_models):
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
    best_val_loss = torch.inf

    model = Model(1)
    opt = optim.Adam(model.parameters(), lr=1e-2)
    num_procs = len(adj)
    for e in range(num_epochs):
        # zero grad
        if id == 0:
            print(f"Epoch {e}")
        for r in range(local_update_steps + (id + 1)*2): 
            y_pred = model(data[0])
            # print(data[0],data[1])
            loss = nn.MSELoss()(y_pred, data[1])
            loss.backward()
            opt.step()
            opt.zero_grad()

        # get num_selects distinct random numbers from 0 to num_proc - 1 except id
        for pid in adj[id]:
            sockets[id][pid].send_pyobj({'sender_id': id, 'tick': e,'type':'model_msg', 'params':model.state_dict()})

        while num_model_received[0] < len(adj[id]):
            pass
        
        with torch.no_grad():
            new_state_dict = model.state_dict()
            for recved_model in recved_models:
                for param, value in recved_model.items():
                    new_state_dict[param] = value + new_state_dict.get(param,0.)
            
            for param in new_state_dict.keys():
                new_state_dict[param] /= (1+len(adj[id]))
            
            model.load_state_dict(new_state_dict)                    
        
        recved_models.clear()
        num_model_received[0] = 0
        
        while num_ack_received[0] < len(adj[id]):
            pass

        num_ack_received[0] = 0

        if id == 0:
            with torch.no_grad():
                y_pred = model(x_val)
                val_loss = nn.MSELoss()(y_val, y_pred)

            if val_loss < best_val_loss:
                best_val_loss = val_loss            
                torch.save(model.state_dict(), f"models/{model_file_p2p}")


        for pid in adj[id]:
            sockets[id][pid].send_pyobj({'type': 'safe_msg', 'sender_id': id})

        while num_safe_received[0] < len(adj[id]):
            pass

        num_safe_received[0] = 0
    
    for pid in adj[id]:
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
    while num_markers_received < len(adj[id]):
        msg = consumer_socket.recv_pyobj()
        if msg['type'] == 'model_msg':
            recved_models.append(msg['params'])
            num_model_received[0] += 1
            sockets[id][msg['sender_id']].send_pyobj({'type': 'ack_msg', 'sender_id': id})
            comm_costs[id] += 1
                        
        elif msg['type'] == 'ack_msg':
            num_ack_received[0] += 1
            
        elif msg['type'] == 'safe_msg':
            num_safe_received[0] += 1
        
        elif msg['type'] == 'marker_msg':
            num_markers_received += 1

    consumer_socket.close()
                                                                        

def init_sockets(adj_list):
    for i in range(len(adj_list)):
        for j in adj_list[i]:
            skt = context.socket(zmq.PUSH)
            skt.connect(f"inproc://#{j}")
            sockets[i][j] = skt

def print_adj_list(adj_list):
    for i in range(len(adj_list)):
        print(f"Process {i} has neighbors: {adj_list[i]}")

def main():
    # Create a ZeroMQ context

    recved_models = [[] for _ in range(num_procs)]
    in_nodes = [0 for _ in range(num_procs)]
    num_ack_received = [[0] for _ in range(num_procs)]
    num_safe_received = [[0] for _ in range(num_procs)]
    num_model_received = [[0] for _ in range(num_procs)]
    
    generate_connected_graph_adjacency_list(num_procs)
    print_adj_list(adj)
    init_sockets(adj)
    threads = []
    for i in range(num_procs):
        threads.append(threading.Thread(target=trainer, args=(i, data[i], num_model_received[i], num_safe_received[i], num_ack_received[i],recved_models[i])))
        threads.append(threading.Thread(target=receiver, args=(context, i, num_model_received[i], num_ack_received[i], num_safe_received[i], recved_models[i])))
    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    end = time.time()

    log_file = open(log_filename, "a")
    log_file.write("Peer-to-Peer:\n")
    log_file.write(f"num_procs: {num_procs}\n")
    log_file.write(f"Time taken: {end-start}\n")
    #for i in range(num_procs):
    #    log_file.write(f"Process {i} communication cost: {comm_costs[i]}\n")
    log_file.write("Total communication cost: {}\n".format(sum(comm_costs)))   

if __name__ == "__main__":
    main()