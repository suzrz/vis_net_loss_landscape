import os
import net
import plot
import torch
import argparse
import calc_loss
import data_load
import directions
from torch import optim as optim
from torch.optim.lr_scheduler import StepLR


def main():
    """PARSE ARGUMENTS"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    """CREATE INSTANCE OF NEURAL NETWORK"""
    model = net.Net().to(device)
    # model.share_memory()

    # If initialized state does not exist, create new one. Else skip.
    if not os.path.isfile("init_state.pt"):
        torch.save(model.state_dict(), "init_state.pt")
        print("New init state saved.")

    model.load_state_dict(torch.load("init_state.pt"))  # load initialized state from file

    torch.manual_seed(1)  # set seed

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # set optimizer
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)  # set scheduler

    # check if exists trained model params
    if not os.path.isfile("final_state.pt"):
        print("Final state not found - beginning training")
        for epoch in range(1, 2):  # here can be set number of epochs
            net.train(model, data_load.train_loader, optimizer, device, epoch)
            net.test(model, data_load.test_loader, device)
            scheduler.step()
            print("Finished epoch no. ", epoch)

        torch.save(model.state_dict(), "final_state.pt")  # save final parameters of model

    model.load_state_dict(torch.load("init_state.pt"))
    dirs = directions.random_directions(model)  # get random directions
    calc_loss.calculate_loss(model, dirs, device)  # calculate val loss and save it to surface file

    plot.vis()


if __name__ == "__main__":
    main()  # run
