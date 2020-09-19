import torch
from torch import utils as utils
from torchvision import datasets, transforms



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

tr = list(range(0, len(train_set), 6))
te = list(range(0, len(test_set), 10))

train_set = torch.utils.data.Subset(train_set, tr)
test_set = torch.utils.data.Subset(test_set, te)

train_loader = utils.data.DataLoader(train_set, 64, shuffle=True)
test_loader = utils.data.DataLoader(test_set, 1000, shuffle=False)

#print(len(train_loader.dataset))
#print(len(test_loader.dataset))


