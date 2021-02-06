import net
import plot
import torch
import argparse
import data_load
import numpy as np
from paths import *
from interpolate import Interpolator
from torch import optim as optim
from torch.optim.lr_scheduler import StepLR


def parse_arguments():
    """PARSE ARGUMENTS"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument("--alpha-start", type=float, default=-1., help="Set starting point of interpolation (float, default = -1.0)")
    parser.add_argument("--alpha-end", type=float, default=1., help="Set ending point of interpolation (float, default = 1.0)")
    parser.add_argument("--alpha-steps", type=int, default=20, help="Set number of interpolation steps (int, default = 20)")
    parser.add_argument("--single-param-only", action="store_true",
                        help="Only one single parameter will be interpolated")
    parser.add_argument("--two-params-only", action="store_true",
                        help="Only two parameters will be interpolated")
    parser.add_argument("--epochs", type=int, default=14, help="Set number of training epochs (default = 14)")
    parser.add_argument("--preliminary", action="store_true", help="Preliminary experiments will be executed.")

    args = parser.parse_args()

    return args


def get_net(device, train_loader, test_loader, epochs):
    # Create instance of neural network
    model = net.Net().to(device)

    # Save initial state of network if not saved yet
    if not init_state.exists():
        torch.save(model.state_dict(), init_state)
        print("New initial state saved.")

    # Get optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # set optimizer
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)  # set scheduler

    # Initialize neural network model
    model.load_state_dict(torch.load(init_state))  # Loading initial state to be sure that net is in initial state
    torch.manual_seed(1)  # set seed

    # Train model if not trained yet
    if not final_state.exists():
        print("Final state not found - beginning training")
        for epoch in range(1, epochs):
            net.train(model, train_loader, optimizer, device, epoch)
            net.test(model, test_loader, device)
            scheduler.step()
            print("Finished epoch no. ", epoch)

        torch.save(model.state_dict(), final_state)  # save final parameters of model

    model.load_state_dict(torch.load(final_state))

    return model  # return neural network in final (trained) state


def main():
    args = parse_arguments()

    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Check if directory for results exists
    if not os.path.isdir("results"):
        os.makedirs("results")

    # Prepare dataset
    train_loader, test_loader = data_load.data_load()  # Get test and train data loader

    # Get neural network model
    model = get_net(device, train_loader, test_loader, args.epochs)

    alpha = np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps)  # Prepare interpolation coefficient
    interpolate = Interpolator(model, device, alpha, final_state, init_state)  # Create interpolator

    interpolate.get_final_loss_acc(test_loader)  # get final loss and accuracy
    interpolate.single_acc_vloss(test_loader, "conv2", [4, 0, 0, 0])  # examine parameter
    interpolate.vec_acc_vlos(test_loader, "conv2")

    #Preliminary experiments
    if args.preliminary:
        subs_train = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 30000, 40000, 50000, 60000]
        subs_test = [1000, 1500, 2000, 3000, 4000, 5000, 7000, 8000, 9000, 10000]
        epochs = [2, 5, 10, 15, 17, 20, 22, 25, 27, 30]

        net.pre_train_subset(model, device, subs_train, args.epochs, test_loader)
        net.pre_test_subset(model, device, subs_test)
        net.pre_epochs(model, device, epochs)

        plot.plot_impact(subs_train, np.loadtxt(train_subs_loss), np.loadtxt(train_subs_acc), xlabel="Size of training data set")
        plot.plot_impact(epochs, np.loadtxt(epochs_loss), np.loadtxt(epochs_acc), annotate=False, xlabel="Number of epochs")
        plot.plot_box(subs_test, show=True, xlabel="Size of test subset")

    """
    if not args.single_param_only:
        # prepare files for 3D plot
        dirs = directions.random_directions(model, device)  # get random directions
        calculate_loss.double(model, test_loader, dirs, device, args.results_dir)  # calculate validation loss
        # plot
        plot.surface3d_rand_dirs(args.results_dir)
    """


if __name__ == "__main__":
    main()  # run
