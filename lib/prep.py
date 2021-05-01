import re
import sys
from lib import net, plot
import torch
import random
import linear
import itertools
import argparse
import numpy as np
import quadratic
from lib.paths import *
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
    parser.add_argument("--alpha-start", type=float, action="store", default=-0.5, nargs='?',
                        help="Set starting point of interpolation (float, default = -0.5).")
    parser.add_argument("--alpha-end", type=float, action="store", default=1.5, nargs='?',
                        help="Set ending point of interpolation (float, default = 1.5).")
    parser.add_argument("--alpha-steps", type=int, action="store", default=40, nargs='?',
                        help="Set number of interpolation steps (int, default = 40).")
    parser.add_argument("--epochs", type=int, action="store", default=14, nargs='?',
                        help="Set number of training epochs (default = 14).")
    parser.add_argument("--idxs", nargs='+', type=int, default=(0, 0, 0, 0),
                        help="Set index of examined parameter (default = (0, 0, 0, 0)).")
    parser.add_argument("--layer", default="conv1",
                        help="Set layer of examined parameter (default = conv1).")
    parser.add_argument("--trained", action="store_true",
                        help="Plot difference between interpolated and actual trained results. "
                             "Available only for layer level experiments.")
    parser.add_argument("--preliminary", action="store_true",
                        help="Preliminary experiments execution.")
    parser.add_argument("--linear-i", action="store_true",
                        help="Individual parameter linear path examination.")
    parser.add_argument("--linear-l", action="store_true",
                        help="Layer linear path examination.")
    parser.add_argument("--quadratic-i", action="store_true",
                        help="Individual parameter quadratic path examination.")
    parser.add_argument("--quadratic-l", action="store_true",
                        help="Layer quadratic path examination.")
    parser.add_argument("--surface", action="store_true",
                        help="Loss function surface visualization.")
    parser.add_argument("--path", action="store_true",
                        help="Optimizer path visualization")
    parser.add_argument("--res", type=int, action="store", default=3, nargs='?',
                        help="Sets the resolution the path visualization (default = 3).")
    parser.add_argument("--auto", action="store_true",
                        help="Runs the 1D experiments automatically.")
    parser.add_argument("--auto-n", type=int, action="store", default=1, nargs='?',
                        help="Sets number of examined parameters for "
                             "auto execution of the 1D experiments (default = 1).")
    parser.add_argument("--plot", action="store_true",
                        help="Plot available data.")
    parser.add_argument("--debug", action="store_true", help="Enables debug logging.")

    args = parser.parse_args()
    args.idxs = tuple(args.idxs)

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
    logger.info("Creating instance of a NN model.")

    model = net.Net().to(device)

    loss_list = []
    acc_list = []

    # Save initial state of network if not saved yet
    if not init_state.exists():
        logger.debug("No initial state of the model found. Initializing ...")
        torch.save(model.state_dict(), init_state)
        logger.debug(f"Initial state saved in {init_state}")

    # Get optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # set optimizer
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)  # set scheduler

    # Load initial state to the model
    model.load_state_dict(torch.load(init_state))
    torch.manual_seed(1)  # set seed

    # Train model if not trained yet
    if not final_state.exists():
        logger.debug("No final state of the model found. Training ...")

        loss, acc = net.test(model, test_loader, device)
        loss_list.append(loss)
        acc_list.append(acc)

        logger.debug(f"Training the NN model.")
        for epoch in range(1, epochs):
            net.train(model, train_loader, optimizer, device, epoch)

            loss, acc = net.test(model, test_loader, device)
            logger.debug(f"Epoch {epoch}\n"
                         f"Loss {loss}\n"
                         f"Accuracy {acc}")

            loss_list.append(loss)
            acc_list.append(acc)

            scheduler.step()

            torch.save(model.state_dict(), Path(checkpoints, f"checkpoint_{epoch}"))

        torch.save(model.state_dict(), final_state)  # save final parameters of model

        np.savetxt(os.path.join(results, "actual_loss"), loss_list)
        np.savetxt(os.path.join(results, "actual_acc"), acc_list)

    model.load_state_dict(torch.load(final_state))

    return model  # return neural network in final (trained) state


def sample(indexes, n_samples=30):
    samples = []

    if n_samples > len(indexes):
        logger.warning(f"Number of samples {n_samples} is bigger than "
                       f"len of sampled list {len(indexes)}. Using full set...")
        return indexes

    interval = len(indexes) // n_samples
    logger.debug(f"Samples interval: {interval}")
    count = 0
    for i in range(len(indexes) - 1):
        if i % interval == 0 and count < n_samples:
            samples.append(indexes[i])
            count += 1

    return samples


def _run_interpolation(idxs, args, device):
    """
    Runs the interpolation on both levels.

    :param idxs: list of positions of the examined parameters
    :param args: experiment configuration
    :param device: device to be used
    """
    linear.run_layer(args, device)
    quadratic.run_layers(args, device)

    for i in idxs:
        args.idxs = i
        logger.debug(f"layer: {args.layer}, idxs: {idxs}")
        linear.run_individual(args, device)
        quadratic.run_individual(args, device)


def run_all(args, device):
    """
        Runs linear and quadratic interpolation automatically over all layers
        and chosen number of parameters.

        Warning: Works only for the provided model architecture specified in net.py

        :param args: experiment parameters
        :param device: device to be used
    """
    linear.run_complete(args, device)
    quadratic.run_complete(args, device)

    """-------------- CONV1 --------------"""
    aux = [list(np.arange(0, 6)), [0], list(np.arange(0, 3)), list(np.arange(0, 3))]
    conv1_idxs = list(itertools.product(*aux))
    conv1_idxs = random.sample(conv1_idxs, args.auto_n)
    logger.debug(f"Number of parameters to be examined in layer conv1: {len(conv1_idxs)}")

    args.layer = "conv1"

    _run_interpolation(conv1_idxs, args, device)

    """-------------- CONV2 --------------"""
    aux = [list(np.arange(0, 6)), list(np.arange(0, 6)), list(np.arange(0, 3)), list(np.arange(0, 3))]
    conv2_idxs = list(itertools.product(*aux))
    conv2_idxs = random.sample(conv2_idxs, args.auto_n)
    logger.debug(f"Number of parameters to be examined in conv2 layer: {len(conv2_idxs)}")

    args.layer = "conv2"

    _run_interpolation(conv2_idxs, args, device)

    """-------------- FC1 --------------"""
    aux = [list(np.arange(0, 120)), list(np.arange(0, 576))]
    fc1_idxs = list(itertools.product(*aux))
    fc1_idxs = random.sample(fc1_idxs, args.auto_n)
    logger.debug(f"Number of parameters to be examined in fc1 layer: {len(fc1_idxs)}")

    args.layer = "fc1"

    _run_interpolation(fc1_idxs, args, device)

    """-------------- FC2 --------------"""
    aux = [list(np.arange(0, 84)), list(np.arange(0, 120))]
    fc2_idxs = list(itertools.product(*aux))
    fc2_idxs = random.sample(fc2_idxs, args.auto_n)
    logger.debug(f"Number of parameters to be examined in fc2 layer: {len(fc2_idxs)}")

    args.layer = "fc2"

    _run_interpolation(fc2_idxs, args, device)

    """-------------- FC3 --------------"""
    aux = [list(np.arange(0, 10)), list(np.arange(0, 84))]
    fc3_idxs = list(itertools.product(*aux))
    fc3_idxs = random.sample(fc3_idxs, args.auto_n)
    logger.debug(f"Number of parameters to be examined in fc3 layer: {len(fc3_idxs)}")

    args.layer = "fc3"

    _run_interpolation(fc3_idxs, args, device)

    # prepare x-axis and opacity dictionary for plotting all parameters of a layer
    x = np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps)
    d = plot.map_distance(individual)

    # plot parameters of each layer in one plot
    plot.plot_params_by_layer(x, "conv1", d)
    plot.plot_params_by_layer(x, "conv2", d)
    plot.plot_params_by_layer(x, "fc1", d)
    plot.plot_params_by_layer(x, "fc2", d)
    plot.plot_params_by_layer(x, "fc3", d)

    # plot all layers in one
    alpha = np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps)

    plot.plot_vec_all_la(alpha)
    plot.plot_lin_quad_real(alpha)
    plot.plot_individual_lin_quad(alpha)

    sys.exit(0)


def plot_available(args):
    individual_files = os.listdir(individual)
    layer_files = os.listdir(layers)

    alpha = np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps)

    for fil in individual_files:
        if not re.search("distance", fil):
            if re.search("loss", fil):
                plot.plot_metric(alpha, np.loadtxt(Path(individual, fil)), Path(individual_img, fil), "loss")
            if re.search("acc", fil):
                plot.plot_metric(alpha, np.loadtxt(Path(individual, fil)), Path(individual_img, fil), "acc")

    for fil in layer_files:
        if not re.search("distance", fil):
            if re.search("loss", fil):
                plot.plot_metric(alpha, np.loadtxt(Path(layers, fil)), Path(layers_img, fil), "loss")
            if re.search("acc", fil):
                plot.plot_metric(alpha, np.loadtxt(Path(layers, fil)), Path(layers_img, fil), "acc")

    d = plot.map_distance(individual)

    # plot parameters of each layer in one plot
    plot.plot_params_by_layer(alpha, "conv1", d)
    plot.plot_params_by_layer(alpha, "conv2", d)
    plot.plot_params_by_layer(alpha, "fc1", d)
    plot.plot_params_by_layer(alpha, "fc2", d)
    plot.plot_params_by_layer(alpha, "fc3", d)

    plot.plot_vec_all_la(alpha)

    plot.plot_lin_quad_real(alpha)
    plot.plot_individual_lin_quad(np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps))
