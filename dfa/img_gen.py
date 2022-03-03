import math
import copy
import numpy as np
import tensorflow.keras as keras
import cv2
(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()

size = 10000
size2 = 10000

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

f = open("img/x_train.txt", "w")
for x in a:
    for i in range(14*14):
        f.write(str(x[i]) + " ")
f.close()

f = open("img/y_train.txt", "w")
for x in b:
    f.write(str(x[0]) + " ")
f.close()

f = open("img/x_test.txt", "w")
for x in c:
    for i in range(14*14):
        f.write(str(x[i]) + " ")
f.close()

f = open("img/y_test.txt", "w")
for x in d:
    f.write(str(x[0]) + " ")
f.close()
