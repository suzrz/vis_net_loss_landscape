"""
Neural Network Training Progress Path Visualization using PCA directions

:author: Silvie Nemcova (xnemco06@stud.fit.vutbr.cz)
:year: 2021
"""
import nnvis
import prep


def run_pca_surface(args, device):
    """
    Runs the visualization of loss surface around trained model.

    :param args: experiment configuration
    :param device: device to be used
    """
    train_loader, test_loader = nnvis.data_load()

    model = prep.get_net(device, train_loader, test_loader, args.epochs)

    examine = nnvis.Examinator2D(model, device)

    data = examine.get_loss_grid(test_loader, resolution=args.res)

    nnvis.contour_path(data["path_2d"], data["loss_grid"], data["coords"], data["pcvariances"])
    nnvis.surface_contour(data["loss_grid"], data["coords"])
