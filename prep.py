import sys
import net
import torch
import random
import itertools
import argparse
import numpy as np
import individual_param
import quadr_interpolation
import layer_params
import q_interpolation_layers
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
    parser.add_argument("--auto", action="store_true",
                        help="Runs the single parameters and layers experiments automatically.")
    parser.add_argument("--auto-n", type=int, action="store", default=10, nargs='?',
                        help="Sets number of examined parameters (default = 10).")
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
            torch.save(model.state_dict(), os.path.join(checkpoints, "checkpoint_{}".format(epoch)))

        torch.save(model.state_dict(), final_state)  # save final parameters of model

        np.savetxt(os.path.join(results, "actual_loss"), loss_list)
        np.savetxt(os.path.join(results, "actual_acc"), acc_list)

    model.load_state_dict(torch.load(final_state))
    logger.debug("[main get_net]: Loaded final parameters in the model.")

    return model  # return neural network in final (trained) state


def sample(indexes, n_samples=30):
    samples = []

    if (n_samples > len(indexes)):
        logger.warning(f"Number of samples {n_samples} is bigger than len of sampled list {len(indexes)}. Using full set...")
        return indexes

    interval = len(indexes) // n_samples
    logger.debug(f"Samples interval: {interval}")
    count = 0
    for i in range(len(indexes) - 1):
        if i % interval == 0 and count < n_samples:
            samples.append(indexes[i])
            count += 1

    return samples


def _run_interpolation(idxs, args):
    if args.single:
        layer_params.run_layers(args)
    #if args.quadratic:
    #    q_interpolation_layers.run_quadr_interpol_layers(args)

    for i in idxs:
        args.idxs = i
        logger.debug(f"layer: {args.layer}, idxs: {idxs}")
        if args.single:
            individual_param.run_single(args)
        if args.quadratic:
            quadr_interpolation.run_quadr_interpolation(args)


def run_all(args):
    """
        Runs linear and quadratic interpolation automatically over all layers
        and chosen number of parameters.

        Warning: Works only for model architecture specified in net.py

        :param args: experiment parameters
    """
    """-------------- CONV1 --------------"""
    aux = [list(np.arange(0, 6)), [0], list(np.arange(0, 3)), list(np.arange(0, 3))]
    conv1_idxs = list(itertools.product(*aux))
    conv1_idxs = random.sample(conv1_idxs, args.auto_n)
    logger.debug(f"Number of parameters to be examined in layer conv1: {len(conv1_idxs)}")

    args.layer = "conv1"

    _run_interpolation(conv1_idxs, args)

    """-------------- CONV2 --------------"""
    aux = [list(np.arange(0, 6)), list(np.arange(0, 6)), list(np.arange(0, 3)), list(np.arange(0, 3))]
    conv2_idxs = list(itertools.product(*aux))
    conv2_idxs = random.sample(conv2_idxs, args.auto_n)
    logger.debug(f"Number of parameters to be examined in conv2 layer: {len(conv2_idxs)}")

    args.layer = "conv2"

    _run_interpolation(conv2_idxs, args)

    """-------------- FC1 --------------"""
    aux = [list(np.arange(0, 120)), list(np.arange(0, 576))]
    fc1_idxs = list(itertools.product(*aux))
    fc1_idxs = random.sample(fc1_idxs, args.auto_n)
    logger.debug(f"Number of parameters to be examined in fc1 layer: {len(fc1_idxs)}")

    args.layer = "fc1"

    _run_interpolation(fc1_idxs, args)

    """-------------- FC2 --------------"""
    aux = [list(np.arange(0, 84)), list(np.arange(0, 120))]
    fc2_idxs = list(itertools.product(*aux))
    fc2_idxs = random.sample(fc2_idxs, args.auto_n)
    logger.debug(f"Number of parameters to be examined in fc2 layer: {len(fc2_idxs)}")

    args.layer = "fc2"

    _run_interpolation(fc2_idxs, args)

    """-------------- FC3 --------------"""
    aux = [list(np.arange(0, 10)), list(np.arange(0, 84))]
    fc3_idxs = list(itertools.product(*aux))
    fc3_idxs = random.sample(fc3_idxs, args.auto_n)
    logger.debug(f"Number of parameters to be examined in fc3 layer: {len(fc3_idxs)}")

    args.layer = "fc3"

    _run_interpolation(fc3_idxs, args)

    sys.exit(0)

