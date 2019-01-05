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
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np

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

# Create a numpy array.
x = np.array([[1,2],[3,4]])

# Convert the numpy array to a torch tensor.
y = torch.from_numpy(x)

# Convert the torch tensor to a numpy array.
z = y.numpy()

# ================================================================== #
#                         4. Input pipline                           #
# ================================================================== #

# Download and construct CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',train=True, transform=transforms.ToTensor(), download=True)

# Fetch one data pair (read from disk)
image, label = train_dataset[0]
print(image.size)
print(label)

# Data loader (this provides queues and threads in a very simple way).
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# When iteration starts, queue and thread start to load data from files.
data_iter = iter(train_loader)

# Mini-batch images and labels.
images, labels = data_iter.next()

# Actual usage of the data loader is as below.
for images, labels in train_loader:
    # Training code should be written here.
    pass

# ================================================================== #
#                5. Input pipline for custom dataset                 #
# ================================================================== #
# We should build your custom dataset as below.
class CustomDataSet(torch.utils.data.Dataset):
    def __init__(self):
        # TODO
        # 1. Initialize file paths or a list of file names. 
        pass

    def __get_item__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
        
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0 

# We can then use the prebuilt data loader. 
custom_dataset = CustomDataSet()
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #

# Download and load the pretrained ResNet-34
resnet = torchvision.models.resnet34(pretrained=True)

# If you want to finetune only the top layer of the model, set as below.
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 10) # 10 is no of desired output classes

# Training
# Forward pass 
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print(outputs.size()) # (64,10)
# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #

# Save and load the entire model.
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended).
torch.save(resnet.state_dict(), 'model-params.ckpt')
resnet.load_state_dict(torch.load('model-params.ckpt'))