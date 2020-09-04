import os
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as f
from torch import nn as nn
from torch import optim as optim
from torch import utils as utils
from collections import OrderedDict
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    """
    Neural network class

    This net consists of 4 layers. Two are convolutional and
    the other two are dropout.
    """

    def __init__(self):
        super(Net, self).__init__()
        # define layers of network
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Feed forward.

        Takes data x and outputs predicted label.
        :type x: tensor
        :param x: input data
        :return: probability
        """

        x = self.conv1(x)  # self.conv1.weight.data[0][0][0][0].item() -- first weight of convolutional layer no. 1
        x = f.relu(x)  # ReLU activation to avoid VANISHING GRADIENT
        x = self.conv2(x)
        x = f.relu(x)
        x = f.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = f.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = f.log_softmax(x, dim=1)
        return output



def train(model, train_loader, optimizer, device):
    model.train()
    iterations = []
    train_loss = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model.forward(data)

        loss = f.nll_loss(output, target)  # compute loss
        loss.backward()
        optimizer.step()

def test(model, test_loader, device):
    model.eval()
    # correct = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            test_loss += f.nll_loss(output, target, reduction="sum").item()
            # pred = output.argmax(dim=1, keepdim=True)
            # correct += pred.eq(target.view_as(pred).sum().item())

    test_loss /= len(test_loader.dataset)
    print(test_loss)
    return test_loss


device = torch.device("cpu")

model = Net().to(device)

if not os.path.isfile("init_state.pt"):
    torch.save(model.state_dict(), "init_state.pt")
    print("New init state saved.")

"""DATA PREPROCESSING
    Prepare data transformations

    transform.ToTensor() converts images into tensors
    transform.Normalize(mean, std) normalizes tensor norm = (img_pix - mean) / std
"""

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.1307, 0.3081)])

"""LOAD DATA
    Load data from torchvision and apply transform
"""

train_set = datasets.MNIST("../data", train=True, download=True,
                           transform=transform)
test_set = datasets.MNIST("../data", train=False, download=True,
                          transform=transform)

train_loader = utils.data.DataLoader(train_set, 64, shuffle=True)
test_loader = utils.data.DataLoader(test_set, 64, shuffle=False)


model.load_state_dict(torch.load("init_state.pt"))
theta_i = copy.deepcopy(model.state_dict())
theta_0 = copy.deepcopy(model.state_dict())
test(model, test_loader, device)  # ok

optimizer = optim.Adadelta(model.parameters(), lr=1.0)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

if not os.path.isfile("final_state.pt"):
    for epoch in range(1, 2):
        train(model, train_loader, optimizer, device)
        scheduler.step()
        print("Finished epoch no. ", epoch)

    torch.save(model.state_dict(), "final_state.pt")

model.load_state_dict(torch.load("final_state.pt"))
theta_f = copy.deepcopy(model.state_dict())
theta_1 = copy.deepcopy(model.state_dict())
test(model, test_loader, device)  # ok
"""INTERPOLATION"""

alpha = np.linspace(-0.25, 1.5, 13)
loss_list = []
theta = OrderedDict()

for alpha_act in alpha:
    print(alpha_act)
    for param_name0, param_name1 in zip(theta_i, theta_f):
        theta_0[param_name0] = torch.mul(theta_i[param_name0],
                                           (1.0 - alpha_act))
        theta_1[param_name1] = torch.mul(theta_f[param_name1],
                                           alpha_act)
        theta[param_name0] = torch.add(theta_0[param_name0],
                                         theta_1[param_name1])

    loss = 0.
    #print(type(theta_0))
    #print(type(theta))
    #theta = copy.deepcopy(theta_0)
    if not model.load_state_dict(theta):
        print("FALXKLDKJFLKSJK") # load parameters in model
    loss = test(model, test_loader, device)
    loss_list.append(loss)

print(loss_list)

fig, axe = plt.subplots()
axe.plot(alpha, loss_list, "x-",color='b')
axe.spines['right'].set_visible(False)
axe.spines['top'].set_visible(False)
plt.xlabel("alpha")
plt.ylabel("loss")
plt.show()
