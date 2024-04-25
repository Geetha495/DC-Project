'''
    log.txt has the following content:
    Peer-to-Peer:
    num_procs: 
    Total time taken:
    garbage for multiple lines
    Total communication cost:
    Server:
    num_procs:
    Time taken:
    garbage for multiple lines
    Total communication cost:
    The above is repeated for different values of num_procs.

    I want to parse the log.txt file and extract the following information:
    1. num_procs from Peer-to-Peer, Total time taken from Peer-to-Peer
    2. num_procs from Server, Time taken from Server
    3. num_procs from Peer-to-Peer, Total Communication cost from Peer-to-Peer
    4. num_procs from Server, Total Communication cost from Server

    Plot two curves with x-axis as num_procs and y-axis as Total time taken and Time taken respectively.
    Plot two curves with x-axis as num_procs and y-axis as Total Communication cost and Communication cost respectively.
'''

import re
import matplotlib.pyplot as plt

def parse_logs(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract num_procs and Total time taken from Peer-to-Peer
    peer_to_peer = re.findall(r"Peer-to-Peer:\nnum_procs: (\d+)\nTotal time taken: (\d+\.\d+)", content)
    peer_to_peer = [(int(x), float(y)) for x, y in peer_to_peer]
    
    # Extract num_procs and Time taken from Server
    server = re.findall(r"Server:\nnum_procs: (\d+)\nTime taken: (\d+\.\d+)", content)
    server = [(int(x), float(y)) for x, y in server]
    
    return peer_to_peer, server

def plot_graph(peer_to_peer, server):
    x1, y1 = zip(*peer_to_peer)
    x2, y2 = zip(*server)
    
    plt.plot(x1, y1, label='Peer-to-Peer')
    plt.plot(x2, y2, label='Server')
    plt.xlabel('num_procs')
    plt.ylabel('Time taken')
    plt.legend()
    plt.savefig('time_vs_num_procs.png')


if __name__ == '__main__':
    peer_to_peer, server = parse_logs('log.txt')
    plot_graph(peer_to_peer, server)