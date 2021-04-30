import logging
from lib import examine2D, plot, data_load, prep

logger = logging.getLogger("vis_net")


def run_pca_surface(args, device):
    """
    Runs the visualization of loss surface around trained model.

    :param args: experiment configuration
    :param device: device to be used
    """
    logger.info("Running loss function landscape visualization")

    train_loader, test_loader = data_load.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)

    examine = examine2D.Examinator2D(model, device)

    data = examine.get_loss_grid(test_loader, resolution=args.res)

    plot.contour_path(data["path_2d"], data["loss_grid"], data["coords"], data["pcvariances"])
    plot.surface_contour(data["loss_grid"], data["coords"])
