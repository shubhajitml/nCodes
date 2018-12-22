# A fully-connected ReLU network that on each forward pass chooses a random 
# number between 0 and 4 and uses that many hidden layers, reusing the same 
# weights multiple times to compute the innermost hidden layers
# Here we can implement weight sharing among the innermost layers by simply 
# reusing the same Module multiple times when defining the forward pass

# coding : utf-8
import time
import random
import torch

class DynamicNet(torch.nn.Module):
    def __init__(self,D_in, H, D_out):
        """
        In the constructor we construct three nn.Linear instances that we will use
        in the forward pass.
        """
        super().__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.hidden_linear = torch.nn.Linear(H,H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
        and reuse the hidden_linear Module that many times to compute hidden layer
        representations.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same Module many
        times when defining a computational graph. This is a big improvement from 
        Lua Torch, where each Module could be used only once.
        """
        relu_h = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0,3)):
            relu_h = self.hidden_linear(relu_h).clamp(min=0)
        y_pred = self.output_linear(relu_h)
        return y_pred

# N : batch size, D_in : input dimension,
# H : hidden dimension, D_out : output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct our model by instantiating the class defined above
model = DynamicNet(D_in, H, D_out)

# Construct our loss function and an Optimizer. Training this strange model with
# vanilla stochastic gradient descent is tough, so we use momentum
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

start_time = time.time()
for t in range(500):
    # Forward pass : compute predicted y by passing x to the model
    y_pred = model(x)

    # compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

finish_time = time.time()
print(f'time of execution: ', finish_time - start_time)  # in my first run 2.1568965911865234 s