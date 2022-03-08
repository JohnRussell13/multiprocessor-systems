import math
import copy
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()

import warnings
warnings.filterwarnings("error")

##
##	Pass input trough the network
##

def net(input, weights, output_temp, len_dim, dim):
    output = copy.deepcopy(output_temp)
    output[0] = input
    for i in range(len_dim):
        for j in range(dim[i+1]):
            output[i+1][j] += weights[i][j][-1]
            for k in range(dim[i]):
                output[i+1][j] += weights[i][j][k]*output[i][k]
        if (i+1) < len_dim:
            output[i+1][output[i+1] < 0] = 0 #relu
        else:
            # for q in range(10):
            #     try:
            #         np.exp(output[-1][q])
            #     except:
            #         print(output)
            #         print("------------------------------------------------------")
            #         print(weights)
            output[i+1] = np.exp(output[-1])/np.sum(np.exp(output[-1])) #softmax
    return output

##
##	Train the network
##

def train(input, output,  blind_in, blind_out, epochs, dim, p = 0.2, learn_rate = 0.05):

##
##	Init values
##

    # Calc once, and use it
    input = np.array(input)
    output = np.array(output)
    output_temp = []
    for i in dim:
        output_temp.append(np.zeros(i))
    len_dim = len(dim)-1
    r = []
    for i in range(len_dim):
        r.append([dim[i], dim[i+1]])
    nVar = 0
    dimVar = []
    for i in range(len_dim):
        nVar += (dim[i]+1) * dim[i+1]
        dimVar.append((dim[i+1], (dim[i]+1)))
    VarMin = -1
    VarMax = 1
    outs_temp = np.eye(dim[-1])
    B = []
    for i in range(len_dim-1):
        B.append(np.random.normal(0, 0.1, [dim[i+1] * 2, dim[-1]])) #bias
    deltas = copy.deepcopy(output_temp)
    epoch_w = []

    # Init weights
    #weights = np.random.uniform(VarMin, VarMax, nVar)
    weights = []
    for i in range(len_dim):
        weights.append(np.random.normal(0, 0.1, dimVar[i]))

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

            out = net(x, weights, output_temp, len_dim, dim)
            delta = out[-1] - outs_temp[y][0]

##
##	Find the error of each layer
##
            
            da = []
            for i in range(len_dim-1):
                da.append(np.dot(B[i], delta))
            da.append(np.append(delta, delta))

            #not needed, just for ilustration
            #use da[i][:dim[i+1]] instead of dw
            #use da[i][dim[i+1]:] instead of db
            dw = []
            db = []
            for i in range(len_dim):
                dw.append(da[i][:dim[i+1]])
                db.append(da[i][dim[i+1]:])

##
##	Update the weights
##

            for i in range(len_dim):
                weights[i][:,:-1] += -learn_rate*np.tensordot(dw[i], out[i], axes=0) #- lmd*weights[i][:,:-1]
                weights[i][:,-1] += -learn_rate*db[i] #- lmd*weights[i][:,-1]
            
            #weights[weights < -1] = -1
            #weights[weights > 1] = 1

##
##	Test and print the results
##

        tot = 0
        siz = len(input_test)
        ce = 0
        for (i,j) in zip(input_test,output_test):
            temp = net(i, weights, output_temp, len_dim, dim)[-1]
            ce += -np.log(temp[j[0]])
            if np.argmax(temp) == j:
                tot += 1
        b_tot = 0
        b_siz = len(blind_in)
        b_ce = 0
        for (i,j) in zip(blind_in,blind_out):
            temp = net(i, weights, output_temp, len_dim, dim)[-1]
            b_ce += -np.log(temp[j[0]])
            if np.argmax(temp) == j:
                b_tot += 1
        print(f'Epoch {e+1}/{epochs}: Seen - Err: {ce/siz:.5f}; Acc: {tot/siz:.5f}. Unseen - Err: {b_ce/b_siz:.5f}; Acc: {b_tot/b_siz:.5f}.')
        epoch_w.append(weights)
    return epoch_w

    
import cv2
import time

size = 100
size2 = 100

k = 14/28

a = x_train[:size]
a = [cv2.resize(i, (0, 0), fx = k, fy = k) for i in a]
a = [i.flatten()/256 - 1/2 for i in a]

b = y_train[:size]
b = [i.flatten() for i in b]

c = x_test[:size2]
c = [cv2.resize(i, (0, 0), fx = k, fy = k) for i in c]
c = [i.flatten()/256 - 1/2 for i in c]

d = y_test[:size2]
d = [i.flatten() for i in d]

epochs = 30
dim = [14*14,10,10,10]
start = time.perf_counter()
epoch_w = train(a, b, c, d, epochs, dim)
end = time.perf_counter()
print(f"Training finished in {end - start:.2f}s.")
if epochs > 1:
    weights = epoch_w[-1]
else:
    weights = epoch_w