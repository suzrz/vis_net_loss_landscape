from lib import data_load, prep
from lib.examine1D import *


def run_complete(args, device):
    """
    Function runs a linear path experiment over all parameters of the model

    :param args: CLI arguments with experiment configuration
    :param device: device to be used
    """
    alpha = np.linspace(0, 1, args.alpha_steps)  # set interpolation start and end are disabled

    train_loader, test_loader = data_load.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)

    interpolate = Linear(model, device, alpha, final_state, init_state)

    interpolate.interpolate_all_linear(test_loader)


def run_layer(args, device):
    """
    Function setups and executes experiment of interpolation of parameters

    :param args: experiment configuration
    :param device: device to be used
    """
    alpha = np.linspace(0, 1, args.alpha_steps)
    logger.debug(f"Interpolation coefficient alpha "
                 f"start: {args.alpha_start}"
                 f"end: {args.alpha_end}"
                 f"steps: {args.alpha_steps}")

    train_loader, test_loader = data_load.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)

    interpolate = Linear(model, device, alpha, final_state, init_state)

    if args.trained:
        interpolate.interpolate_all_linear(test_loader)
    interpolate.layers_linear(test_loader, args.layer, args.trained)


def run_individual(args, device):
    """
    Function executes experiment with individual parameter

    :param args: command line arguments
    :param device: device to be used
    """
    alpha = np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps)  # setup interpolation coefficient
    logger.debug(f"Interpolation coefficient alpha "
                 f"start: {args.alpha_start}"
                 f"end: {args.alpha_end}"
                 f"steps: {args.alpha_steps}")

    train_loader, test_loader = data_load.data_load()  # setup data loaders

    model = prep.get_net(device, train_loader, test_loader, args.epochs)  # setup model

    interpolate = Linear(model, device, alpha, final_state, init_state)  # get interpolator instance

    interpolate.individual_param_linear(test_loader, args.layer, args.idxs)  # execute the experiment
