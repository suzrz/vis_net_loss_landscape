import net
import h5py
import torch
import numpy as np
from paths import *
from sklearn.decomposition import PCA

logger = logging.getLogger("vis_net")


def get_random_direction(model, device):
    """
    Function generates random directions and normalizes them

    :param model: model to be respected when normalizing
    :param device: device to be used
    :return: normalized random direction
    """
    weights = [p.data for p in model.parameters()]
    direction = [torch.randn(w.size()).to(device) for w in weights]

    assert (len(direction) == len(weights))

    for d, w in zip(direction, weights):
        for dire, wei in zip(d, w):
            dire.mul_(wei.norm() / (dire.norm() + 1e-10))

    return direction


def get_directions(model, device):
    """
    Function prepares two random directions

    :param model: model
    :param device: device to be used
    :return: list of two random directions
    """
    x = get_random_direction(model, device)
    y = get_random_direction(model, device)

    return [x, y]


def set_surf_file(filename):
    """
    Function prepares h5py file for storing loss function values

    :param filename: Filename of a surface file
    """
    xmin, xmax, xnum = -1, 2, 20
    ymin, ymax, ynum = -1, 2, 20

    if filename.exists():
        return

    with h5py.File(filename, 'a') as fd:
        xcoord = np.linspace(xmin, xmax, xnum)
        fd["xcoordinates"] = xcoord

        ycoord = np.linspace(ymin, ymax, ynum)
        fd["ycoordinates"] = ycoord

        shape = (len(xcoord), len(ycoord))

        losses = -np.ones(shape=shape)
        fd["loss"] = losses


def get_indices(vals, xcoords, ycoords):
    """
    Function gets indices

    :param vals: values
    :param xcoords: x coordinates
    :param ycoords: y coordinates
    :return: indices
    """
    ids = np.array(range(vals.size))
    ids = ids[vals.ravel() <= 0]

    xcoord_mesh, ycoord_mesh = np.meshgrid(xcoords, ycoords)

    s1 = xcoord_mesh.ravel()[ids]
    s2 = ycoord_mesh.ravel()[ids]

    return ids, np.c_[s1, s2]


def overwrite_weights(model, init_weights, directions, step, device):
    """
    Function overwrite weights of the model according to actual step

    :param model: model which parameters are updated
    :param init_weights: initial parameters of the model
    :param directions: x, y directions
    :param step: step
    :param device: device
    """
    dx = directions[0]
    dy = directions[1]

    changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]

    for (p, w, d) in zip(model.parameters(), init_weights, changes):
        p.data = w.to(device) + torch.tensor(d).to(device)


def calc_loss(model, test_loader, directions, device):
    """
    Function iterates over surface file and calculates loss over the surface

    :param model: model to be evaluated
    :param test_loader: test dataset loader
    :param directions: random projection directions
    :param device: device
    """
    logger.info("Calculating loss function surface")
    filename = Path(os.path.join(random_dirs, "surf.h5"))
    logger.debug(f"Surface file: {filename}")

    set_surf_file(filename)

    init_weights = [p.data for p in model.parameters()]

    with h5py.File(filename, "r+") as fd:
        xcoords = fd["xcoordinates"][:]
        ycoords = fd["ycoordinates"][:]
        losses = fd["loss"][:]

        ids, coords = get_indices(losses, xcoords, ycoords)

        for count, idx in enumerate(ids):
            coord = coords[count]
            logger.debug(f"Index: {idx}")

            overwrite_weights(model, init_weights, directions, coord, device)

            loss, _ = net.test(model, test_loader, device)
            logger.debug(f"Loss: {loss}")

            losses.ravel()[idx] = loss

            fd["loss"][:] = losses

            fd.flush()


def sample_path(steps, n_samples=300):
    """
    Function takes n_samples from steps dictionary

    :param steps: dictionary of sgd steps with members flat_w [] and loss []
    :param n_samples: number of samples to take
    :return: sampled dict
    """
    samples = {"flat_w": [], "loss": []}

    if n_samples > len(steps):
        logger.warn(f"Less steps ({len(steps)} than final samples ({n_samples}). Using whole set of steps.")
        n_samples = len(steps)

    interval = len(steps) // n_samples
    count = 0
    for i in range(len(steps) - 1, -1, -1):
        if i % interval == 0 and count < n_samples:
            samples["flat_w"].append(steps["flat_w"][i])
            samples["loss"].append(steps["loss"][i])
            count += 1

    samples["flat_w"] = reversed(samples["flat_w"])
    samples["loss"] = reversed(samples["loss"])
    return samples


def pca_dim_reduction(params, directions):
    optim_path_np = []
    for tensor in params:
        optim_path_np.append(np.array(tensor.cpu()))

    pca = PCA(n_components=2)
    path_2d = pca.fit_transform(optim_path_np)
    reduced_dirs = pca.components_

    return {
        "optim_path": optim_path_np,
        "path_2d": path_2d,
        "reduced_dirs": reduced_dirs,
        "pcvariances": pca.explained_variance_ratio_
    }