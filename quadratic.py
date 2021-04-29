import prep
import data_load
from examine1D import *


def run_complete(args, device):
    """
    Runs quadratic interpolation examination over all parameters of the model.

    :param args: experiment configuration
    :param device: device to be used
    """
    alpha = np.linspace(0, 1, args.alpha_steps)

    train_loader, test_loader = data_load.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)

    interpolate = Quadratic(model, device, alpha, final_state, init_state)

    interpolate.interpolate_all_quadratic(test_loader)


def run_layers(args, device):
    """
    Runs quadratic interpolation examination on the level of layers.

    :param args: experiment configuration
    :param device: device to be used
    """
    logger.info("Running quadratic interpolation on the level of layers")

    alpha = np.linspace(0, 1, args.alpha_steps)

    train_loader, test_loader = data_load.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)

    interpolate = Quadratic(model, device, alpha, final_state, init_state)

    interpolate.layers_quadratic(test_loader, args.layer)


def run_individual(args, device):
    """
    Runs quadratic interpolation examination on the level of parameters.

    :param args: experiment configuration
    :param device: device to be used
    """
    logger.info("Running quadratic interpolation of parameters")

    alpha = np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps)

    train_loader, test_loader = data_load.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)

    interpolate = Quadratic(model, device, alpha, final_state, init_state)

    interpolate.individual_param_quadratic(test_loader, args.layer, args.idxs)
