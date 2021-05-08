import prep
import numpy as np
import nnvis


def run_complete(args, device):
    """
    Runs quadratic interpolation examination over all parameters of the model.

    :param args: experiment configuration
    :param device: device to be used
    """
    alpha = np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps)

    train_loader, test_loader = nnvis.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)

    interpolate = nnvis.Quadratic(model, device, alpha, nnvis.final_state, nnvis.init_state)

    interpolate.interpolate_all_quadratic(test_loader)

    interpolate_l = nnvis.Linear(model, device, alpha, nnvis.final_state, nnvis.init_state)
    interpolate_l.interpolate_all_linear(test_loader)

    nnvis.plot_lin_quad_real(alpha)


def run_layers(args, device):
    """
    Runs quadratic interpolation examination on the level of layers.

    :param args: experiment configuration
    :param device: device to be used
    """

    alpha = np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps)

    train_loader, test_loader = nnvis.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)

    interpolate = nnvis.Quadratic(model, device, alpha, nnvis.final_state, nnvis.init_state)

    interpolate.layers_quadratic(test_loader, args.layer)


def run_individual(args, device):
    """
    Runs quadratic interpolation examination on the level of parameters.

    :param args: experiment configuration
    :param device: device to be used
    """

    alpha = np.linspace(args.alpha_start, args.alpha_end, args.alpha_steps)

    train_loader, test_loader = nnvis.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)

    interpolate = nnvis.Quadratic(model, device, alpha, nnvis.final_state, nnvis.init_state)

    interpolate.individual_param_quadratic(test_loader, args.layer, args.idxs)
