import prep
import nnvis
import numpy as np


def run_complete(args, device):
    """
    Function runs a linear path experiment over all parameters of the model

    :param args: CLI arguments with experiment configuration
    :param device: device to be used
    """
    logger.info("Running linear path examination on the level of model.")
    logger.debug(f"Interpolation coefficient alpha\n"
                 f"start: {args.alpha_start}\n"
                 f"end: {args.alpha_end}\n"
                 f"steps: {args.alpha_steps}")

    alpha = np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps)

    train_loader, test_loader = nnvis.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)

    interpolate = nnvis.Linear(model, device, alpha, nnvis.final_state, nnvis.init_state)

    interpolate.interpolate_all_linear(test_loader)


def run_layer(args, device):
    """
    Function setups and executes experiment of interpolation of parameters

    :param args: experiment configuration
    :param device: device to be used
    """
    logger.info(f"Running linear path examination on the level of layer {args.layer}")
    logger.debug(f"Interpolation coefficient alpha\n "
                 f"start: {args.alpha_start}\n"
                 f"end: {args.alpha_end}\n"
                 f"steps: {args.alpha_steps}\n")

    alpha = np.linspace(0, 1, args.alpha_steps)

    train_loader, test_loader = nnvis.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)

    interpolate = nnvis.Linear(model, device, alpha, nnvis.final_state, nnvis.init_state)

    if args.trained:
        interpolate.interpolate_all_linear(test_loader)
    interpolate.layers_linear(test_loader, args.layer)


def run_individual(args, device):
    """
    Function executes experiment with individual parameter

    :param args: command line arguments
    :param device: device to be used
    """
    logger.info(f"Running linear path examination on the level of parameters {args.layer} {args.idxs}")
    logger.debug(f"Interpolation coefficient alpha\n "
                 f"start: {args.alpha_start}\n"
                 f"end: {args.alpha_end}\n"
                 f"steps: {args.alpha_steps}\n")

    alpha = np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps)  # setup interpolation coefficient

    train_loader, test_loader = nnvis.data_load()  # setup data loaders

    model = prep.get_net(device, train_loader, test_loader, args.epochs)  # setup model

    interpolate = nnvis.Linear(model, device, alpha, nnvis.final_state, nnvis.init_state)  # get interpolator instance

    interpolate.individual_param_linear(test_loader, args.layer, args.idxs)  # execute the experiment
