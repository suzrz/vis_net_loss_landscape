import logging
import nnvis
import prep

logger = logging.getLogger("vis_net")


def run_pca_surface(args, device):
    """
    Runs the visualization of loss surface around trained model.

    :param args: experiment configuration
    :param device: device to be used
    """
    logger.info("Running loss function landscape visualization using PCA directions")

    train_loader, test_loader = nnvis.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)

    examine = nnvis.Examinator2D(model, device)

    data = examine.get_loss_grid(test_loader, resolution=args.res)

    nnvis.contour_path(data["path_2d"], data["loss_grid"], data["coords"], data["pcvariances"])
    nnvis.surface_contour(data["loss_grid"], data["coords"])
