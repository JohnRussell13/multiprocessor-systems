import math
import copy
import numpy as np
import tensorflow.keras as keras
(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()
import cv2
from mpi4py import MPI

##
##	Pass input trough the network
##

def test(a, weights, b, dim, perc):
    #INIT
    output_temp = []
    for i in dim:
        output_temp.append(np.zeros(i))
    len_dim = len(dim)-1
    per = 0

    #MPI INIT
    per_layer = int(size/len_dim) #how many processors per layer
    if rank != size:
        layer = int(rank/per_layer) #layer map
        part = rank - layer*per_layer #which part of layer
        layer_part_size = int(dim[layer+1]/per_layer)

    per = int(len(a)*perc/100)

    input = a[:per]
    output = b[:per]

    t = 0
    p = 0

    for ind in range(per):
        if rank != size:
            #INIT
            if layer == 0:
                net_in = input[ind]
            
            else:
                net_in = []
                for i in range(per_layer):
                    net_in = np.append(net_in, comm.recv(source = (layer-1)*per_layer + i))

            
            net_out = np.zeros(int(dim[layer+1]/per_layer))
            for i in range(int(dim[layer+1]/per_layer)):
                net_out[i] = weights[i][-1] #bias
                for j in range(dim[layer]):
                    net_out[i] += weights[i][j]*net_in[j] #weights * previous_layer_output

            if layer < len_dim-1:
                net_out[net_out < 0] = 0 #relu
            else:
                net_out = np.exp(net_out)/np.sum(np.exp(net_out)) #softmax

            if layer < len_dim-1:
                for i in range(per_layer):
                    comm.send(net_out, dest = (layer+1)*per_layer + i)
            
            else:
                comm.send(net_out, dest = size)

        if rank == size:
            res = []
            for i in range(per_layer):
                res = np.append(res, comm.recv(source = (len_dim-1)*per_layer + i))

        if rank == size:
            if ind % (per/10) == 0:
                print(f"Testing {p*10}%")
                p += 1
            if np.argmax(res) == output[ind][0]:
                t += 1
    if rank == size:
        print(f"Accuracy: {t}/{per}")

comm = MPI.COMM_WORLD
size = comm.Get_size() - 1
rank = comm.Get_rank()
#rank == size - final part of the network

input = x_test
k = 14/28
input = [cv2.resize(i, (0, 0), fx = k, fy = k) for i in input]
input = [i.flatten()/256 - 1/2 for i in input]

output = y_test
output = [i.flatten() for i in output]

dim = [14*14,10,10,10]
len_dim = len(dim)-1

#MPI INIT
per_layer = int(size/len_dim) #how many processors per layer
if rank != size:
    layer = int(rank/per_layer) #layer map
    part = rank - layer*per_layer #which part of layer
    layer_part_size = int(dim[layer+1]/per_layer)

if rank != size:
    weights = np.random.normal(0, 0.1, (int(dim[layer+1]/per_layer), (dim[layer]+1)))

if rank == size:
    weights = 0

test(input, weights, output, dim, 10)