"""https://github.com/okn-yu/Visualizing-the-Loss-Landscape-of-Neural-Nets"""
import os
import net
import plot
import torch
import argparse
import data_load
import directions
import calculate_loss
from torch import optim as optim
from torch.optim.lr_scheduler import StepLR


def main():
    """PARSE ARGUMENTS"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--interpolation-samples", type=int, default=13, help="Set number of interpolation samples (default = 13)")
    parser.add_argument("--hist", type=bool, default=False)
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

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

    if args.hist:
        print("hist")
        calculate_loss.diff_len_data(model, optimizer, scheduler, device)
        #plot.plot_subset_hist()


    train_loader, test_loader = data_load.data_load()
    # check if exists trained model params
    if not os.path.isfile("final_state.pt"):
        print("Final state not found - beginning training")
        for epoch in range(1, 14):  # here can be set number of epochs
            net.train(model, train_loader, optimizer, device, epoch)
            net.test(model, test_loader, device)
            scheduler.step()
            print("Finished epoch no. ", epoch)

        torch.save(model.state_dict(), "final_state.pt")  # save final parameters of model

    model.load_state_dict(torch.load("init_state.pt"))

    # prepare files for 2D plot
    calculate_loss.single(model, train_loader, test_loader, device, args.interpolation_samples, optimizer)
    # plot
    #plot.line2D_single_parameter()
    plot.plot_accuracy(args.interpolation_samples)
    plot.plot_2D_loss(args.interpolation_samples)

    # prepare files for 3D plot
    dirs = directions.random_directions(model)  # get random directions
    calculate_loss.double(model, test_loader, dirs, device)  # calculate val loss and save it to surface file
    # plot
    plot.surface3D_rand_dirs()


if __name__ == "__main__":
    main()  # run
