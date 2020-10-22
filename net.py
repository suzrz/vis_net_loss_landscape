import os
import copy
import torch
import argparse
import data_load
import torch.nn.functional as f
from torch import optim as optim
from torch import nn as nn
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

        Parameters
        ----------
        :param x: pyTorch tensor
            Input data
        :return: float
            Output data. Probability of data belonging to one of classes
        """

        x = self.conv1(x)
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


def train(model, train_loader, optimizer, device, epoch):
    """ Trains the network.

    :param model : Neural network model to be trained
    :param train_loader : Data loader
    :param optimizer : Optimizer
    :param device : Device on which will be the net trained
    :param epoch : Number of actual epoch
    """
    model.train()  # put net into train mode
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # load data
        optimizer.zero_grad()  # zero all gradients
        output = model.forward(data)  # feed data through net

        loss = f.nll_loss(output, target)  # compute train loss
        train_loss += f.nll_loss(output, target, reduction="sum").item()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print("Train epoch: {} [{}/{} ({:.0f} %)]\tLoss: {:.6f}".format(
                epoch, batch_idx*len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader.dataset)
    return train_loss


def test(model, test_loader, device):
    """ Validates the neural network.

    :param model : Neural network model to be validated
    :param test_loader : Data loader
    :device : Device on which will be validation performed
    """
    model.eval()  # put model in evaluation mode
    test_loss = 0  # validation loss
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            test_loss += f.nll_loss(output, target, reduction="sum").item()

    test_loss /= len(test_loader.dataset)  # compute validation loss of neural network
    return test_loss


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = Net().to(device)  # create instance of neural network class
#model.share_memory()

# If initialized state does not exist, create new one. Else skip.
if not os.path.isfile("init_state.pt"):
    torch.save(model.state_dict(), "init_state.pt")
    print("New init state saved.")

model.load_state_dict(torch.load("init_state.pt"))  # load initialized state from file
theta_i = copy.deepcopy(model.state_dict())  # save model parameters into dict which should remain unchanged
theta_0 = copy.deepcopy(model.state_dict())  # save model parameters into dict which will be changed

torch.manual_seed(1)  # set seed



optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # set optimizer
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)  # set scheduler

# If model was not trained yet, it will train. Else skip.
if not os.path.isfile("final_state.pt"):
    print("Final state not found - beginning training")
    for epoch in range(1, 10):  # here can be set number of epochs
        train(model, data_load.train_loader, optimizer, device, epoch)
        scheduler.step()
        print("Finished epoch no. ", epoch)

    torch.save(model.state_dict(), "final_state.pt")  # save final parameters of model

theta_f = copy.deepcopy(torch.load("final_state.pt"))  # save final parameters into thetas (same as theta_i, theta_0)
theta_1 = copy.deepcopy(torch.load("final_state.pt"))
