"""
TODO
    - val loss < train loss ?
    - two params
    - jeden skalar z tensoru parametru

"""
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
        #if batch_idx % 10 == 0:
        #    print("Train epoch: {} [{}/{} ({:.0f} %)]\tLoss: {.6f}".format(
        #        epoch, batch_idx*len(data), len(train_loader.dataset),
        #        100. * batch_idx / len(train_loader), loss.item()))

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


def process_data():
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
    test_loader = utils.data.DataLoader(test_set, 1000, shuffle=False)

    return train_set, test_set, train_loader, test_loader

def main():
    device = torch.device("cpu")  # set device which will script work on
                                  # TODO implement GPU

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

    train_set, test_set, train_loader, test_loader = process_data()  # load train and test data

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # set optimizer
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)  # set scheduler

    # If model was not trained yet, it will train. Else skip.
    if not os.path.isfile("final_state.pt"):
        for epoch in range(1, 14):  # here can be set number of epochs
            train(model, train_loader, optimizer, device, epoch)
            scheduler.step()
            print("Finished epoch no. ", epoch)

        torch.save(model.state_dict(), "final_state.pt")  # save final parameters of model

    theta_f = copy.deepcopy(torch.load("final_state.pt"))  # save final parameters into thetas (same as theta_i, theta_0)
    theta_1 = copy.deepcopy(torch.load("final_state.pt"))

    """INTERPOLATION"""
    alpha = np.linspace(-0.25, 1.5, 13)  # set interpolation coefficient
    train_loss_list = []  # prepare clean list for train losses
    val_loss_list = []  # prepare clean list for validation losses
    theta = OrderedDict()  # prepare clean parameter dict

    for alpha_act in alpha:  # interpolate
        for param_name0, param_name1 in zip(theta_i, theta_f):
            theta_0[param_name0] = torch.mul(theta_i[param_name0],
                                               (1.0 - alpha_act))
            theta_1[param_name1] = torch.mul(theta_f[param_name1],
                                               alpha_act)
            theta[param_name0] = torch.add(theta_0[param_name0],
                                             theta_1[param_name1])

        if not model.load_state_dict(theta):
            print("Something went wrong.")  # loading parameters in model failed
        train_loss = train(model, train_loader, optimizer, device, 0)
        train_loss_list.append(train_loss)
        val_loss = test(model, test_loader, device)  # get loss with new parameters
        val_loss_list.append(val_loss)  # save obtained loss into list

    # plot
    fig, axe = plt.subplots()
    axe.plot(alpha, val_loss_list, "x-", label="validation loss")
    axe.plot(alpha, train_loss_list, "o-", color="orange", label="train loss")  # not normalized! should be lower than validation loss but because it is measured on more samples, it looks worse
    axe.spines['right'].set_visible(False)
    axe.spines['top'].set_visible(False)
    plt.legend()
    plt.xlabel("alpha")
    plt.ylabel("loss")
    plt.show()

if __name__ == "__main__":
    main()  # run
