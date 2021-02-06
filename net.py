import copy
import torch
import data_load
import numpy as np
from paths import *
from torch import optim
from torch import nn as nn
import torch.nn.functional as f
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
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

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
        x = f.relu(x)
        x = f.max_pool2d(x, 2)
        x = self.conv2(x)
        x = f.relu(x)
        x = f.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = f.relu(x)
        x = self.fc2(x)
        x = f.relu(x)
        x = self.fc3(x)

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
            #print("Train epoch: {} [{}/{} ({:.0f} %)]\tLoss: {:.6f}".format(
            #   epoch, batch_idx*len(data), len(train_loader.dataset),
            #    100. * batch_idx / len(train_loader), loss.item()))

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
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            test_loss += f.nll_loss(output, target, reduction="sum").item()

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)  # compute validation loss of neural network
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy


def pre_train_subset(model, device, subset_list, epochs, test_loader):
    """
    Function to examine impact of different sizes of training subset.
    """
    if train_subs_loss.exists() and train_subs_acc.exists():
        return

    loss_list = []
    acc_list = []
    theta_i = copy.deepcopy(torch.load(init_state))
    theta_f = copy.deepcopy(torch.load(final_state))

    for n_samples in subset_list:
        model.load_state_dict(theta_i)

        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # set optimizer
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)  # set scheduler

        for epoch in range(1, epochs):
            train_loader, test_loader = data_load.data_load(train_samples=n_samples)

            train(model, train_loader, optimizer, device, epoch)
            test(model, test_loader, device)

            scheduler.step()
            print("[pre_train_subset] : Finished epoch {} for training set of size: {}".format(epoch,
                                                                                                           n_samples))

        loss, acc = test(model, test_loader, device)

        loss_list.append(loss)
        acc_list.append(acc)

    np.savetxt(train_subs_loss, loss_list)
    np.savetxt(train_subs_acc, acc_list)

    model.load_state_dict(theta_f)


def pre_test_subset(model, device, subset_list):
    if test_subs_loss.exists() and test_subs_acc.exists():
        return

    subset_losses = []
    subset_accs = []
    theta_f = copy.deepcopy(torch.load(final_state))

    model.load_state_dict(theta_f)

    for n_samples in subset_list:
        losses = []
        accs = []
        for x in range(100):  # 10x pruchod experimentem TODO
            print("[pre_test_subset] : Measuring performance for {} samples, {} [{} %]".format(n_samples, x, x/100*100))
            _, test_loader = data_load.data_load(test_samples=n_samples)  # to choose random data each time
            loss, acc = test(model, test_loader, device)
            losses.append(loss)
            accs.append(acc)

        subset_losses.append(losses)
        subset_accs.append(accs)

    np.savetxt(test_subs_loss, subset_losses)
    np.savetxt(test_subs_acc, subset_accs)


def pre_epochs(model, device, epochs_list):
    if epochs_loss.exists() and epochs_acc.exists():
        return

    loss_list = []
    acc_list = []

    theta_i = copy.deepcopy(torch.load(init_state))

    model.load_state_dict(theta_i)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # set optimizer
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)  # set scheduler
    train_loader, test_loader = data_load.data_load()

    for epoch in range(max(epochs_list)):
        train(model, train_loader, optimizer, device, epoch)
        test(model, test_loader, device)

        scheduler.step()

        print("[pre_epochs] : Finished epoch {}".format(epoch))
        if epoch in epochs_list:
            loss, acc = test(model, test_loader, device)

            loss_list.append(loss)
            acc_list.append(acc)

    np.savetxt(epochs_loss, loss_list)
    np.savetxt(epochs_acc, loss_list)