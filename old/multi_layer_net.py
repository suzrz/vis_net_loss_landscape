import initialize
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch import optim as optim
from torch import utils as utils
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()  # sets module in training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # set all gradients to zero
        output = model.forward(data)
        loss = F.nll_loss(output, target)  # compute loss

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print("Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

            iteration_list.append(batch_idx * len(data))
            train_loss_list.append(loss)
            if args.dry_run:
                break
    final_state = model.state_dict()
    return final_state  # thetaf (final weights)

def test(args, model, device, test_loader):
    # begin TESTING
    model.eval()
    test_loss = 0
    correct = 0
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction="sum").item()

            # get index of max log probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            count += 1

            iteration_list_v.append(count * len(data))
            accuracy_list.append(100. * correct / len(test_loader.dataset))
            test_loss_list.append(F.nll_loss(output, target))  # validation loss

    test_loss /= len(test_loader.dataset)  # validation loss
    return test_loss

# for printing results
train_loss_list = []
test_loss_list = []
iteration_list = []
accuracy_list = []
iteration_list_v = []
parameter_val = []


def main():
    """Parse arguments"""
    parser = argparse.ArgumentParser(description="MNIST PyTorch")
    parser.add_argument("--batch-size", type=int, default=64, metavar='N',
                        help="Input batch size for training (default: 64)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar='N',
                        help="Input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=14, metavar='N',
                        help="Number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR",
                        help="Learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar='M',
                        help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="quickly check a single pass")
    parser.add_argument("--seed", type=int, default=1, metavar='S',
                        help="Random seed (default: 1)")
    parser.add_argument("--log-interval", type=int, default=10, metavar='N',
                        help="How many batches to wait before logging training status")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="Save the current model")

    args = parser.parse_args()
    """END of arguments parsing"""

    torch.manual_seed(args.seed)  # set seed
    device = torch.device("cpu")  # TODO maybe implement support for CUDA

    """ data preprocessing
    Prepare transformations for data (MNIST dataset)
    
    transforms.ToTensor() converts images into numbers (tensors) (3 channels, brightness levels)
    transforms.Normalize(mean, std) normalizes tensor: normalized = (image_pixel - mean) / std
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    #  load data and apply transform
    train_set = datasets.MNIST("../data", train=True, download=True,
                               transform=transform)
    test_set = datasets.MNIST("../data", train=False, download=True,
                              transform=transform)

    train_loader = utils.data.DataLoader(train_set, args.batch_size,
                                         shuffle=True)
    test_loader = utils.data.DataLoader(test_set, args.test_batch_size,
                                        shuffle=False)

    # print(train_set.data.shape)  # torch.Size([60000, 28, 28]) - 60000 images of size 28x28px
    # print(test_set.data.shape)  # torch.Size([10000, 28, 28]) - 10000 images of size 28x28px

    #model = initialize.Net().to(device)  # initialize neural network on specified device
    #init_state = model.state_dict()  # save initialized state (theta0)


    model = initialize.Net().to(device)
    model.load_state_dict(torch.load("init_state.pt"))  # load initialized state
    theta_i = model.state_dict()

    optimizer = optim.Adadelta(initialize.model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    final_state = train(args, initialize.model, device, train_loader, optimizer, args.epochs)  # save final weights
    torch.save(final_state, "final_state.pt")
    theta_f = model.state_dict()
    test(args, initialize.model, device, test_loader)

    # scheduler.step()

    if args.save_model:  # save final state if desired
        torch.save(initialize.model.state_dict(), "mnist_cnn.pt")


    """INTERPOLATION"""
    alpha = np.linspace(0, 1, 10)
    theta_0 = theta_i
    loss_list = []
    theta = dict()

    for alpha_act in alpha:
        print(alpha_act)
        for param_tensor0, param_tensor1 in zip(theta_0, theta_f):
            # print(theta_0[param_tensor0]) #<class 'str'>
            theta_0[param_tensor0] = torch.mul(theta_0[param_tensor0], (1 - alpha_act))
            theta[str(param_tensor0)] = torch.add(theta_0[param_tensor0], theta_f[param_tensor1], alpha=alpha_act)

        model.load_state_dict(theta)  # load parameters in model
        theta_0 = model.state_dict()
        loss = test(args, model, device, test_loader)
        loss_list.append(loss)

    print(loss_list)

    plt.plot(alpha, loss_list, "x-")
    plt.xlabel("alpha")
    plt.ylabel("loss")
    plt.show()

    #visualization of parameter vs time
    #plt.plot(iteration_list, parameter_val, label="parameter value")
    #plt.plot(iteration_list, loss_list, label="loss")
    #plt.xlabel("Number of iterations")
    #plt.xscale("log")
    #plt.title("CNN: Parameter value vs Loss")
    #plt.show()
    # visualization of test accuracy
    plt.plot(iteration_list, train_loss_list)
    plt.show()

    plt.plot(iteration_list_v, accuracy_list, color="red")
    plt.xlabel("Number of iteration")
    plt.xscale("log")
    plt.ylabel("Accuracy")
    plt.title("CNN: Accuracy vs Number of iteration")
    plt.show()


# run
if __name__ == "__main__":
    main()
