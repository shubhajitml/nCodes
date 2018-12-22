# When building neural networks we frequently think of arranging the computation into layers, some of which have learnable parameters which will be optimized during learning.
# In pytorch, The nn package defines a set of Modules, which are roughly equivalent to neural network layers.
# A Module receives input Tensors and computes output Tensors, but may also hold internal state such as Tensors containing learnable parameters. 
# The nn package also defines a set of useful loss functions that are commonly used when training neural networks.

# Here, we use the nn package to implement our two-layer network
# coding : utf-8
import time
import torch

# N : batch size, D_in : input dimension,
# H : hidden dimension, D_out : output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Use the nn package to define our model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each Linear Module computes output from input using a
# linear function, and holds internal Tensors for its weight and bias.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
start_time = time.time()
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

finish_time = time.time()
print(f'time of execution: ', finish_time - start_time)  # in my first run 1.188096284866333 s
