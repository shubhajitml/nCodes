# Code referenced from https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/01-basics/pytorch_basics/main.py
# (not the copy & paste version!)

# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #

# 1. Basic autograd example 1               (Line 25 to 39)
# 2. Basic autograd example 2               (Line 46 to 83)
# 3. Loading data from numpy                (Line 90 to 97)
# 4. Input pipline                          (Line 104 to 129)
# 5. Input pipline for custom dataset       (Line 136 to 156)
# 6. Pretrained model                       (Line 163 to 176)
# 7. Save and load model                    (Line 183 to 189) 

import torch
import torch.nn as nn
import torch.optim as optim

# ================================================================== #
#                     1. Basic autograd example 1                    #
# ================================================================== #

# Create tensors
x = torch.tensor(5., requires_grad=True)
w = torch.tensor(3., requires_grad=True)
b = torch.tensor(2., requires_grad=True)

# Build a computational graph
y = w * x + b  #(y = 3*x + b)

# Compute gradients (derivatives)
y.backward()

# Print out the gradients
print(x.grad) # x.grad = 3 (x.grad=> dy/dx = w = 3 )
print(w.grad) # w.grad = 5 (w.grad=> dy/dw = x = 5 )
print(b.grad) # b.grad = 3 (b.grad=> dy/db = 1 )

# ================================================================== #
#           2. Basic autograd example 2 (in training process)        #
# ================================================================== #

# Data Loading
# Create tensors of shape (4,3) and (4,2)
x = torch.randn(4, 3)
y = torch.randn(4, 2)

# Create the model architecture
# Build a fully connected neural network
linear_model = nn.Linear(3, 2)
print('w:', linear_model.weight)
print('b:', linear_model.bias)

# Build loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)

# Training
#1. Forward pass
y_out = linear_model(x)

#2. Compute loss
loss = criterion(y_out, y)
print('loss: ', loss.item())

# 3. Backward pass
loss.backward()

# Print out the gradients
print('dL/dw:', linear_model.weight.grad)
print('dL/db:', linear_model.bias.grad)

# 4. Optimizer step
# (here, 1-epoch) 1-step Gradient Descent 
optimizer.step()

# You can also perform gradient descent at the low level.
linear_model.weight.data.sub_(1e-2 * linear_model.weight.grad.data)
linear_model.bias.data.sub_(1e-2 * linear_model.bias.grad.data)

# linear.weight.data.sub_(0.01 * linear.weight.grad.data)
# linear.bias.data.sub_(0.01 * linear.bias.grad.data)

# Print out the loss after 1-step gradient descent.
y_out = linear_model(x)
loss = criterion(y_out, y)
print('loss after 1 step optimization: ', loss.item())



# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #




# ================================================================== #
#                         4. Input pipline                           #
# ================================================================== #



# ================================================================== #
#                5. Input pipline for custom dataset                 #
# ================================================================== #


# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #



# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #

