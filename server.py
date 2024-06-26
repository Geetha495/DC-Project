# do same as a.py but with a server
import time
import socket
import threading
import torch
import torch.nn as nn
from model import Model
from config import *
import random
import zmq
sockets_to_server = []
sockets_to_clients = []
comm_costs = [0]*(num_procs+1)
context = zmq.Context()


def server_trainer(id, in_nodes, num_grads, recved_models):
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
    for e in range(num_epochs):
        # zero grad

        for i in range(1,in_nodes+1):
            sockets_to_clients[i].send_pyobj({'type':'param_msg', 'params':model.state_dict()})
        
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
            num_grads[0] = 0
            recved_models.clear()

            with torch.no_grad():
                y_pred = model(x_val)
                val_loss = nn.MSELoss()(y_val, y_pred)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f"models/{model_file_server}")    


def server_consumer(ctx, id, in_nodes, num_grads, recved_models):
    '''
    Continuosly receives msgs 
    '''
    consumer_socket = ctx.socket(zmq.PULL)
    consumer_socket.bind(f"inproc://#server")
    
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


def client_trainer(ctx, id,  data):
    '''
    for epochs
        for all proc p
            send grad
    '''
    consumer_socket = ctx.socket(zmq.PULL)
    consumer_socket.bind(f"inproc://#{id}")
    
    model = Model(1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for e in range(num_epochs):
        msg = consumer_socket.recv_pyobj()
        if msg['type'] == 'param_msg':
            model.load_state_dict(msg['params'])
            for r in range(local_update_steps): 
                y_pred = model(data[0])
                loss = nn.MSELoss()(y_pred, data[1])
                loss.backward()
                opt.step()
                opt.zero_grad()


            comm_costs[id] += 1

            sockets_to_server[id].send_pyobj({'type':'param_msg', 'params':model.state_dict()})
        
def main():
    # Initialize sockets
    for i in range(num_procs+1):
        sockets_to_server.append(context.socket(zmq.PUSH))
        sockets_to_server[i].connect(f"inproc://#server")

        sockets_to_clients.append(context.socket(zmq.PUSH))
        sockets_to_clients[i].connect(f"inproc://#{i}")
    
    # Create threads
    threads = []
    num_grads = [[0]]
    recved_models = []
    for i in range(num_procs+1):
        if i == 0:
            threads.append(threading.Thread(target=server_trainer, args=(i, num_procs, num_grads[0], recved_models)))
            threads.append(threading.Thread(target=server_consumer, args=(context, i, num_procs, num_grads[0], recved_models)))

        else:
            threads.append(threading.Thread(target=client_trainer, args=(context, i, data[i-1])))
    
    start = time.time()
    # Start threads
    for thread in threads:
        thread.start()

    # Join threads
    for thread in threads:
        thread.join()
    end = time.time()

    log_file = open(log_filename, "a")
    log_file.write("Server:\n")
    log_file.write(f"num_procs: {num_procs}\n")
    log_file.write(f"Time taken: {end-start}\n")
    #log_file.write(f"Server communication cost: {comm_costs[0]}\n")
    #for i in range(1,num_procs+1):
    #    log_file.write(f"Process {i} communication cost: {comm_costs[i]}\n")  
    log_file.write("Total communication cost: {}\n".format(sum(comm_costs)))
    log_file.close()        


if __name__ == "__main__":
    main()