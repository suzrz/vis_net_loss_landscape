import os
import torch
import initialize as ini
import torch.nn.functional as f
from torch import optim as optim
from torch import utils as utils
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


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

        #iterations.append(batch_idx * len(data))
        #train_loss.append(loss)
    #return train_loss


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

"""INITIALIZE NETWORK
    Create instance of Net() and load parameters set once in initialize.py
"""
device = torch.device("cpu")

model = ini.Net().to(device)
model.load_state_dict(torch.load("init_state.pt"))
#theta_i = model.state_dict()

#optimizer = optim.SGD(model.parameters(), lr=1.0)
optimizer = optim.Adadelta(model.parameters(), lr=1.0)
scheduler = StepLR(optimizer, step_size=1, gamma=0.7)


if not os.path.isfile("final_state.pt"):
    for epoch in range(1, 2):
        train(model, train_loader, optimizer, device)
        scheduler.step()
        print("Finished epoch no. ", epoch)


    torch.save(model.state_dict(), "final_state.pt")
# test(model, test_loader, device)
#theta_f = model.state_dict()






