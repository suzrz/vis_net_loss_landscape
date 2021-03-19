import sys
import net
import plot
import torch
import logging
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
                        help="Disables CUDA training.")
    parser.add_argument("--alpha-start", type=float, default=-1.,
                        help="Set starting point of interpolation (float, default = -1.0).")
    parser.add_argument("--alpha-end", type=float, default=1.,
                        help="Set ending point of interpolation (float, default = 1.0).")
    parser.add_argument("--alpha-steps", type=int, default=20,
                        help="Set number of interpolation steps (int, default = 20).")
    parser.add_argument("--single-param-only", action="store_true",
                        help="Only one single parameter experiment will be executed.")
    parser.add_argument("--two-params-only", action="store_true",
                        help="Only two parameters will be interpolated.")
    parser.add_argument("--epochs", type=int, default=14,
                        help="Set number of training epochs (default = 14).")
    parser.add_argument("--idxs", nargs='+', default=[0, 0, 0, 0],
                        help="Set index of examined parameter (default = [0, 0, 0, 0]). Recommended to set.")
    parser.add_argument("--layer", default="conv1",
                        help="Set layer of examined parameter (default = conv1). Recommended to set.")
    parser.add_argument("--trained", action="store_true",
                        help="Plot difference between interpolated and actual trained results.")
    parser.add_argument("--preliminary", action="store_true",
                        help="Preliminary experiments will be executed.")
    parser.add_argument("--debug", action="store_true", help="Enables debug logging.")

    args = parser.parse_args()

    return args


def get_net(device, train_loader, test_loader, epochs):
    # Create instance of neural network
    logging.debug("[main]: Getting NN model")
    model = net.Net().to(device)
    logging.debug("[main]: Model:"
                  "{}".format(model))
    loss_list = []
    acc_list = []

    # Save initial state of network if not saved yet
    if not init_state.exists():
        logging.info("[main]: No initial state of the model found. Initializing ...")
        torch.save(model.state_dict(), init_state)
        logging.debug("[main]: Initial state saved into {}".format(init_state))

    # Get optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # set optimizer
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)  # set scheduler

    # Initialize neural network model
    model.load_state_dict(torch.load(init_state))  # Loading initial state to be sure that net is in initial state
    torch.manual_seed(1)  # set seed

    # Train model if not trained yet
    if not final_state.exists():
        logging.info("[main]: No final state of the model found. Training ...")
        for epoch in range(1, epochs):
            net.train(model, train_loader, optimizer, device, epoch)
            loss, acc = net.test(model, test_loader, device)
            loss_list.append(loss)
            acc_list.append(acc)
            scheduler.step()
            logging.debug("[main]: Finished training epoch {}".format(epoch))
            torch.save(model.state_dict(), os.path.join(results, "state_{}".format(epoch)))

        torch.save(model.state_dict(), final_state)  # save final parameters of model

        np.savetxt(os.path.join(results, "actual_loss"), loss_list)
        np.savetxt(os.path.join(results, "actual_acc"), acc_list)

    model.load_state_dict(torch.load(final_state))
    logging.debug("[main get_net]: Loaded final parameters in the model.")

    return model  # return neural network in final (trained) state


def main():
    args = parse_arguments()

    alpha = np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps)  # Prepare interpolation coefficient

    if args.debug:
        lvl = logging.DEBUG
    else:
        lvl = logging.INFO

    logging.basicConfig(level=lvl, format="%(asctime)s - %(levelname)s - %(message)s",
                        filename="main.log", filemode='w')

    logging.debug("[main]: Command line arguments: {}".format(args))

    # Set device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        logging.info("[main]: CUDA is enabled.")
    else:
        logging.info("[main]: CUDA is disabled.")

    init_dirs()

    # Prepare dataset
    train_loader, test_loader = data_load.data_load()  # Get test and train data loader

    # Get neural network model
    model = get_net(device, train_loader, test_loader, args.epochs)

    #Preliminary experiments
    if args.preliminary:
        logging.info("[main]: Preliminary experiments enabled. Executing...")
        subs_train = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 20000, 30000, 40000, 50000, 60000]
        subs_test = [1000, 1500, 2000, 3000, 4000, 5000, 7000, 8000, 9000, 10000]
        epochs = [2, 5, 10, 15, 17, 20, 22, 25, 27, 30]

        net.pre_train_subset(model, device, subs_train, args.epochs, test_loader)
        net.pre_test_subset(model, device, subs_test)
        net.pre_epochs(model, device, epochs)

        plot.plot_impact(subs_train, np.loadtxt(train_subs_loss), np.loadtxt(train_subs_acc), xlabel="Size of training data set")
        plot.plot_impact(epochs, np.loadtxt(epochs_loss), np.loadtxt(epochs_acc), annotate=False, xlabel="Number of epochs")
        plot.plot_box(subs_test, show=True, xlabel="Size of test subset")
        sys.exit(0)

    logging.info("[main]: Executing interpolation experiments...")
    interpolate = Interpolator(model, device, alpha, final_state, init_state)  # Create interpolator

    interpolate.single_acc_vloss(test_loader, args.layer, list(map(int, args.idxs)))  # examine parameter
    interpolate.vec_acc_vlos(test_loader, args.layer, trained=args.trained)
    #interpolate.rand_dirs(test_loader)
    #plot.surface3d_rand_dirs()


    #plot.plot_single(alpha, "conv1", True)
    #plot.plot_single(alpha, "conv2", True)
    #plot.plot_single(alpha, "fc1", True)
    #plot.plot_single(alpha, "fc2", True)
    #plot.plot_single(alpha, "fc3", True)
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
