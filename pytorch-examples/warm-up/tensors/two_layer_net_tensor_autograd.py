# When using autograd (automatic differentiation) of pytorch,
# The forward pass of your network will define a computational graph;
# nodes in the graph will be Tensors, and edges will be functions 
# that produce output Tensors from input Tensors.
# Backpropagating through this graph then allows you to easily compute gradients.

# coding: utf-8
import torch
import time

dtype = torch.float
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# create random tensors to hold inputs and outputs
# setting requires_grad=False indicates that we don't ned to compute gradients
# WRT. these Tensors during the backward pass 
x = torch.randn(N, D_in, device = device, dtype = dtype)
y = torch.randn(N, D_out, device = device, dtype = dtype)

# create random Tensors for weights.
# requires_grad = True  indicates that we want to compute gradients 
# WRT. these Tensors during backward pass
w1 = torch.randn(D_in, H, device = device, dtype = dtype, requires_grad = True)
w2 = torch.randn(H, D_out, device = device, dtype = dtype, requires_grad = True)

learning_rate = 1e-6
start_time = time.time()
for t in range(500):
    # forward pass : compute predicted y
    # since we are not implementing backpropagation by hand,
    # we don't need to keep references to intermediate values
    y_pred = x.mm(w1).clamp(min = 0).mm(w2)

    # Now loss is a Tensor of shape (1,)
    # loss.item() gets the a scalar value held in the loss.
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    # An alternative way is to operate on weight.data and weight.grad.data.
    # Recall that tensor.data gives a tensor that shares the storage with
    # tensor, but doesn't track history.
    # You can also use torch.optim.SGD to achieve this.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()

finish_time = time.time()
print(f'time of execution: ', finish_time - start_time)  # in my first run 1.188096284866333 ms