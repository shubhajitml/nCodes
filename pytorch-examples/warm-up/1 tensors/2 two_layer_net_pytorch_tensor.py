# Tensor is an n-dimensional array
# Pytorch Tensor is conceptually identical to numpy array 
import torch
import time

dtype = torch.float
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# N : batch size, D_in : input dimension,
# H : hidden dimension, D_out : output dimension
N, D_in , H , D_out = 64, 1000, 100, 10

# create random input and output data
x = torch.randn(N, D_in, device = device, dtype = dtype)
y = torch.randn(N, D_out, device = device, dtype = dtype)

# randomly initialize the weights
w1 = torch.randn(D_in, H, device = device, dtype = dtype)
w2 = torch.randn(H, D_out, device = device, dtype = dtype)

learning_rate = 1e-6
start_time = time.time()
for t in range(500):
    # forward pass: compute predicted y
    h = x.mm(w1)
    relu_h = h.clamp(min = 0)
    y_pred = relu_h.mm(w2)

    # compute loss and print
    loss = (y_pred - y).pow(2).sum().item() # Now loss is a Tensor of shape (1,) and loss.item() gets the a scalar value held in the loss
    print(t, loss)

    # backprop to compute gradients of w1 and w2 WRT. loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = relu_h.t().mm(grad_y_pred)
    grad_relu_h = grad_y_pred.mm(w2.t())
    grad_h = grad_relu_h.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

finish_time = time.time()
print(f'time of execution: ' ,(finish_time - start_time)) # in my first run # in my first run 1.1758019924163818 ms