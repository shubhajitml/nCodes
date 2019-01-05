# -*- coding: utf-8 -*-

# A fully-connected ReLU network with a single hidden layer,which
# will be trained with gradient descent to fit random data by
# minimizing the Euclidean distance between the network output 
# and the true output.
import numpy as np
import time

# N : batch size, D_in : input dimension,
# H : hidden dimension, D_out : output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Random input data (x) and output data (y)
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize the weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

# for future experiment
w1_old = w1.copy()
w2_old = w2.copy()

learning_rate = 1e-6
start_time = time.time()
for t in range(500):
    # Forward pass : compute y_pred
    h = x.dot(w1)
    relu_h = np.maximum(h, 0)
    y_pred = relu_h.dot(w2)

    # Compute loss and print 
    loss = np.square(y_pred - y).sum()
    print(f'step :',t,f'loss = ',loss)

    # Backprop to compute gradients of w1 and w2 WRT. loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = relu_h.T.dot(grad_y_pred)
    grad_relu_h = grad_y_pred.dot(w2.T)    
    grad_h = grad_relu_h.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

finish_time = time.time()

# # experimenting weight_matrices values
# print(f'w1 before running sgd: ', w1_old)
# print(f'w1 after running sgd: ', w1)
# print(f'w2 before running sgd: ', w2_old)
# print(f'w2 after running sgd: ', w2)

print(f'time of execution: ' , finish_time - start_time) # in my first run 1.160381555557251 s