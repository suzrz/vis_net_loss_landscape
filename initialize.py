import os
import torch
import torch.nn.functional as f
from torch import nn as nn


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

if not os.path.isfile("init_state.pt"):
    model = Net().to(torch.device("cpu"))
    init_state = model.state_dict()
    torch.save(model.state_dict(), "init_state.pt")
    print("New init state created.")
else:
    print("Init state already exists.")
