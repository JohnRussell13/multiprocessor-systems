import math
import copy
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.mnist.load_data()
import cv2

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
          output[i+1] = np.exp(output[-1])/np.sum(np.exp(output[-1])) #softmax
    return output

def test(a, weights, b, dim, perc):
    output_temp = []
    for i in dim:
        output_temp.append(np.zeros(i))
    len_dim = len(dim)-1

    per = int(len(a)*perc/100)
    input = a[:per]
    output = b[:per]

    t = 0
    p = 0

    for i in range(per):
        if i % (per/10) == 0:
            print(f"Testing {p*10}%")
            p += 1
        if np.argmax(net(input[i], weights, output_temp, len_dim, dim)[-1]) == output[i][0]:
            t += 1
    print(f"Accuracy: {t}/{per}")


input = x_test
k = 14/28
input = [cv2.resize(i, (0, 0), fx = k, fy = k) for i in input]
input = [i.flatten()/256 - 1/2 for i in input]

output = y_test
output = [i.flatten() for i in output]

dim = [14*14,10,10,10]


len_dim = len(dim)-1
dimVar = []
for i in range(len_dim):
    dimVar.append((dim[i+1], (dim[i]+1)))
weights = []
for i in range(len_dim):
    weights.append(np.random.normal(0, 0.1, dimVar[i]))

test(input, weights, output, dim, 10)