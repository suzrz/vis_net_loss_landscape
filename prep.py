import net
import torch
import argparse
import numpy as np
from paths import *
from torch import optim as optim
from torch.optim.lr_scheduler import StepLR


mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

logger = logging.getLogger("vis_net")


def parse_arguments():
    """
    Function parses arguments from command line

    :return: program arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true',
                        help="Disables CUDA training.")
    parser.add_argument("--alpha-start", type=float, action="store", default=-1., nargs='?',
                        help="Set starting point of interpolation (float, default = -1.0).")
    parser.add_argument("--alpha-end", type=float, action="store", default=1., nargs='?',
                        help="Set ending point of interpolation (float, default = 1.0).")
    parser.add_argument("--alpha-steps", type=int, action="store", default=20, nargs='?',
                        help="Set number of interpolation steps (int, default = 20).")
    parser.add_argument("--epochs", type=int, action="store", default=14, nargs='?',
                        help="Set number of training epochs (default = 14).")
    parser.add_argument("--idxs", nargs='+', default=(0, 0, 0, 0),
                        help="Set index of examined parameter (default = [0, 0, 0, 0]).")
    parser.add_argument("--layer", default="conv1",
                        help="Set layer of examined parameter (default = conv1).")
    parser.add_argument("--trained", action="store_true",
                        help="Plot difference between interpolated and actual trained results.")
    parser.add_argument("--preliminary", action="store_true",
                        help="Preliminary experiments will be executed.")
    parser.add_argument("--single", action="store_true",
                        help="Individual parameter interpolation.")
    parser.add_argument("--layers", action="store_true",
                        help="Interpolation of parameters of layer")
    parser.add_argument("--quadratic", action="store_true",
                        help="Quadratic interpolation of individual parameter")
    parser.add_argument("--surface", action="store_true",
                        help="Loss function surface visualization in random directions")
    parser.add_argument("--auto", type=int, action="store", default=10, nargs='?',
                        help="Runs the single parameters and layers experiments automatically (default=10).")
    parser.add_argument("--debug", action="store_true", help="Enables debug logging.")

    args = parser.parse_args()

    return args


def get_net(device, train_loader, test_loader, epochs):
    """
    Function prepares a neural network model for experiments

    :param device: device to use
    :param train_loader: training dataset loader
    :param test_loader: test dataset loader
    :param epochs: number of training epochs
    :return: Net object (NN model)
    """
    # Create instance of neural network
    logger.debug("[main]: Getting NN model")
    model = net.Net().to(device)
    loss_list = []
    acc_list = []

    # Save initial state of network if not saved yet
    if not init_state.exists():
        logger.info("[main]: No initial state of the model found. Initializing ...")
        torch.save(model.state_dict(), init_state)
        logger.debug("[main]: Initial state saved into {}".format(init_state))

    # Get optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # set optimizer
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)  # set scheduler

    # Initialize neural network model
    model.load_state_dict(torch.load(init_state))  # Loading initial state to be sure that net is in initial state
    torch.manual_seed(1)  # set seed

    # Train model if not trained yet
    if not final_state.exists():
        logger.info("[main]: No final state of the model found. Training ...")

        loss, acc = net.test(model, test_loader, device)
        loss_list.append(loss)
        acc_list.append(acc)

        for epoch in range(1, epochs):
            logger.debug("[main]: Epoch {}".format(epoch))
            logger.debug("[main]: Loss {}".format(loss))
            logger.debug("[main]: Acc {}".format(acc))
            net.train(model, train_loader, optimizer, device, epoch)
            loss, acc = net.test(model, test_loader, device)
            loss_list.append(loss)
            acc_list.append(acc)
            scheduler.step()
            logger.debug("[main]: Finished training epoch {}".format(epoch))
            torch.save(model.state_dict(), os.path.join(results, "state_{}".format(epoch)))

        torch.save(model.state_dict(), final_state)  # save final parameters of model

        np.savetxt(os.path.join(results, "actual_loss"), loss_list)
        np.savetxt(os.path.join(results, "actual_acc"), acc_list)

    model.load_state_dict(torch.load(final_state))
    logger.debug("[main get_net]: Loaded final parameters in the model.")

    return model  # return neural network in final (trained) state
