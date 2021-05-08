import copy
import torch
import pickle
from lib import data_load
import numpy as np
from lib.paths import *
from torch import optim
from torch import nn as nn
import torch.nn.functional as f
from torch.optim.lr_scheduler import StepLR


logger = logging.getLogger("vis_net")


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
        logger.info(f"Network was initialized.")
        logger.debug(f"Network architecture:\n{self}")

    def forward(self, x):
        """
        Forward pass data

        :param x: Input data
        :return: Output data. Probability of a data sample belonging to one of the classes
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

    def get_flat_params(self, device):
        params = {}
        for name, param in self.named_parameters():
            params[name] = param.data

        flat_params = torch.Tensor().to(device)

        for _, param in params.items():
            flat_params = torch.cat((flat_params, torch.flatten(param)))

        return flat_params

    def load_from_flat_params(self, f_params):
        shapes = []
        for name, param in self.named_parameters():
            shapes.append((name, param.shape, param.numel()))

        state = {}
        c = 0
        for shape in shapes:
            name, tsize, tnum = shape
            param = f_params[c: c + tnum].reshape(tsize)
            state[name] = torch.nn.Parameter(param)
            c += tnum

        self.load_state_dict(state, strict=True)


def train(model, train_loader, optimizer, device, epoch, checkpoint_file=True):
    """ Trains the network.

    :param model : Neural network model to be trained
    :param train_loader : Data loader
    :param optimizer : Optimizer
    :param device : Device on which will be the net trained
    :param epoch : Number of actual epoch
    :param checkpoint_file: creates checkpoint file
    :return: training loss for according epoch
    """
    model.train()  # put net into train mode
    train_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optim_path = {"flat_w": [], "loss": []}
        data, target = data.to(device), target.to(device)  # load data
        optimizer.zero_grad()  # zero all gradients
        output = model.forward(data)  # feed data through net

        loss = f.nll_loss(output, target)  # compute train loss
        train_loss += f.nll_loss(output, target, reduction="sum").item()
        loss.backward()
        optimizer.step()

        if checkpoint_file:
            filename = Path(os.path.join(checkpoints), f"checkpoint_epoch_{epoch}_step_{batch_idx}.pkl")

            logger.debug(f"Creating checkpoint file {filename}")

            optim_path["flat_w"].append(model.get_flat_params(device))
            optim_path["loss"].append(loss)

            with open(filename, "wb") as fd:
                pickle.dump(optim_path, fd)

    train_loss /= len(train_loader.dataset)
    logger.info(f"Training in epoch {epoch} has finished (loss = {train_loss})")
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
    logger.debug(f"Validation has finished:"
                 f"\n      Validation loss: {test_loss}"
                 f"\n      Accuracy: {accuracy} %")
    return test_loss, accuracy


def pre_train_subset(model, device, subset_list, epochs):
    """
    Function to examine impact of different sizes of training subset.

    :param model: NN model
    :param device: device to be used
    :param subset_list: list of subsets sizes to be examinated
    :param epochs: number of training epoch
    """
    logger.info("Running impact of size of training subset preliminary experiment")
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

        train_loader, test_loader = data_load.data_load(train_samples=n_samples)

        logger.info(f"Training subset size {n_samples}")

        for epoch in range(1, epochs):
            train(model, train_loader, optimizer, device, epoch)

            scheduler.step()

        loss, acc = test(model, test_loader, device)

        loss_list.append(loss)
        acc_list.append(acc)

    np.savetxt(train_subs_loss, loss_list)
    np.savetxt(train_subs_acc, acc_list)

    model.load_state_dict(theta_f)


def pre_test_subset(model, device, subset_list):
    """
    Function examines impact of test dataset size on stability of measurements

    :param model: NN model
    :param device: device to be used
    :param subset_list: list of subset sizes to be examined
    """
    logger.info("Running impact of size of test subset preliminary experiment")
    if test_subs_loss.exists() and test_subs_acc.exists():
        return

    subset_losses = []
    subset_accs = []
    theta_f = copy.deepcopy(torch.load(final_state))

    model.load_state_dict(theta_f)

    for n_samples in subset_list:
        losses = []
        accs = []

        _, test_loader = data_load.data_load(test_samples=n_samples)  # to choose random data each time

        logger.info(f"Test subset size {n_samples}")

        for x in range(10):
            loss, acc = test(model, test_loader, device)

            losses.append(loss)
            accs.append(acc)

            logger.debug(f"Validation loss: {loss}"
                         f"Accuracy: {acc}")

        subset_losses.append(losses)
        subset_accs.append(accs)

    np.savetxt(test_subs_loss, subset_losses)
    np.savetxt(test_subs_acc, subset_accs)


def pre_epochs(model, device, epochs_list):
    """
    Function examines performance of the model after certain number of epochs

    :param model: NN model
    :param device: device to be used
    :param epochs_list: list of epochs numbers after which will be the model evaluated
    """
    logger.info("Epochs performance experiment started.")
    if epochs_loss.exists() and epochs_acc.exists():
        return

    loss_list = []
    acc_list = []

    theta_i = copy.deepcopy(torch.load(init_state))

    model.load_state_dict(theta_i)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # set optimizer
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)  # set scheduler
    train_loader, test_loader = data_load.data_load()

    for epoch in range(max(epochs_list) + 1):
        train(model, train_loader, optimizer, device, epoch)
        test(model, test_loader, device)

        scheduler.step()

        logger.debug(f"Finished epoch {epoch}")
        if epoch in epochs_list:
            loss, acc = test(model, test_loader, device)

            loss_list.append(loss)
            acc_list.append(acc)
            logger.info(f"Performance of the model for epoch {epoch}"
                        f"Validation loss: {loss}"
                        f"Accuracy: {acc}")

    np.savetxt(epochs_loss, loss_list)
    np.savetxt(epochs_acc, loss_list)
