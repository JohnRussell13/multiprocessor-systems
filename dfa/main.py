import math
import copy
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
import cv2
import time
from mpi4py import MPI

##
##	Train the network
##

def train(input, output,  blind_in, blind_out, epochs, dim, p = 0.2, learn_rate = 0.01):
##
##	Init values
##
    # Calc once, and use it
    input = np.array(input)
    output = np.array(output)
    len_dim = len(dim)-1
    outs_temp = np.eye(dim[-1])

    #MPI INIT
    per_layer = int(size/len_dim) #how many processors per layer
    if rank != size:
        layer = int(rank/per_layer) #layer map
        part = rank - layer*per_layer #which part of layer
        layer_part_size = int(dim[layer+1]/per_layer)

    if rank != size:
        B = np.random.normal(0, 0.1, (layer_part_size * 2, dim[-1]))

    # Init weights
        weights = np.random.normal(0, 0.1, (layer_part_size, (dim[layer]+1)))
##
##	Start the training
##
    # Iterate epochs
    for e in range(epochs):
##
##	Split the data
##
        rand = np.random.rand(len(input))
        input_train = input[rand >= p]
        input_test = input[rand < p]
        output_train = output[rand >= p]
        output_test = output[rand < p]

        for (x,y) in zip(input_train, output_train):
##
##	Find the output error
##
            if rank != size:
                #INIT
                if layer == 0:
                    net_in = x
                
                else:
                    net_in = []
                    for i in range(per_layer):
                        net_in = np.append(net_in, comm.recv(source = (layer-1)*per_layer + i))

                
                net_out = np.zeros(layer_part_size)
                for i in range(layer_part_size):
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

            delta = 0
            if rank == size:
                res = []
                for i in range(per_layer):
                    res = np.append(res, comm.recv(source = (len_dim-1)*per_layer + i))
                delta = res - outs_temp[y][0]

            delta = comm.bcast(delta, root = size)

##
##	Find the error of each layer
##
            if rank != size:
                if layer < len_dim-1:
                    da = np.dot(B, delta)
                else:
                    delta_slice = delta[part*layer_part_size:(part+1)*layer_part_size]
                    da = np.append(delta_slice, delta_slice)
                dw = da[:layer_part_size]
                db = da[layer_part_size:]

##
##	Update the weights
##
                weights[:,:-1] += -learn_rate*np.tensordot(dw, net_in, axes=0)
                weights[:,-1] += -learn_rate*db
##
##	Test and print the results
##

        tot = 0
        siz = len(input_test)
        ce = 0
        for (in_t,out_t) in zip(input_test,output_test):
            if rank != size:
                #INIT
                if layer == 0:
                    net_in = in_t
                
                else:
                    net_in = []
                    for i in range(per_layer):
                        net_in = np.append(net_in, comm.recv(source = (layer-1)*per_layer + i))

                
                net_out = np.zeros(layer_part_size)
                for i in range(layer_part_size):
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

                    ce += -np.log(res[out_t[0]])
                    if np.argmax(res) == out_t:
                        tot += 1

        b_tot = 0
        b_siz = len(blind_in)
        b_ce = 0
        for (i,j) in zip(blind_in,blind_out):
            if rank != size:
                #INIT
                if layer == 0:
                    net_in = in_t
                
                else:
                    net_in = []
                    for i in range(per_layer):
                        net_in = np.append(net_in, comm.recv(source = (layer-1)*per_layer + i))

                
                net_out = np.zeros(layer_part_size)
                for i in range(layer_part_size):
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
                    
                b_ce += -np.log(res[out_t[0]])
                if np.argmax(res) == out_t:
                    b_tot += 1

        if rank == size:
            print(f'Epoch {e+1}/{epochs}: Seen - Err: {ce/siz:.5f}; Acc: {tot/siz:.5f}. Unseen - Err: {b_ce/b_siz:.5f}; Acc: {b_tot/b_siz:.5f}.')
    return weights

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
    weights = np.random.normal(0, 0.1, (layer_part_size, (dim[layer]+1)))

if rank == size:
    weights = 0

k = 14/28
epochs = 10

#
#   Preprocessing
#
size1 = 10
size2 = 10
a = x_train[:size1]
a = [cv2.resize(i, (0, 0), fx = k, fy = k) for i in a]
a = [i.flatten()/256 - 1/2 for i in a]

b = y_train[:size1]
b = [i.flatten() for i in b]

c = x_test[:size2]
c = [cv2.resize(i, (0, 0), fx = k, fy = k) for i in c]
c = [i.flatten()/256 - 1/2 for i in c]

d = y_test[:size2]
d = [i.flatten() for i in d]

#
#   Start training
#

if rank == size:
    start = time.perf_counter()

weights = train(a, b, c, d, epochs, dim)

if rank == size:
    end = time.perf_counter()
    print(f"Training finished in {end - start:.2f}s.")