# Whenever we want to specify models that are more complex than a simple 
# sequence of existing Modules;we define your own Modules by 
# subclassing nn.Module and defining a forward which receives input Tensors 
# and produces output Tensors using other modules or other autograd operations
# on Tensors

# A fully-connected ReLU network with one hidden layer, trained to predict y from x by minimizing squared Euclidean distance

# coding : utf-8
import time
import torch

class TwoLayerNet(torch.nn.Module):
    '''
    A fully-connected ReLU network with one hidden layer, trained to predict y from x
    by minimizing squared Euclidean distance
    '''
    def __init__(self, D_in, H, D_out):
        """
        In this constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        relu_h = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(relu_h)
        return y_pred

# N : batch size, D_in : input dimension,
# H : hidden dimension, D_out : output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# Construct the model by instantiating the class defined above
model = TwoLayerNet(D_in, H, D_out)

# Construct out loss function and an Optimizer. The call to
# model.parameters() in the SGD constructor will contain the learnable parameters
# the two nn.Linear modules which are members of the model
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

start_time = time.time()
for t in range(500):
    # Forward Pass : compute predicted y by passing x to the model 
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    print(t, loss.item())

    # Zero gradients, perform a backward pass, and update weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


finish_time = time.time()
print(f'time of execution: ', finish_time - start_time)  # in my first run 1.7642054557800293 s
