import re
import sys
import net
import h5py
import torch
import pickle
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

################################################################################


def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text

    return retval


def natural_keys(text):
    return [atof(c) for c in re.split(r"[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)", text)]


def get_steps(directory=checkpoints):
    logger.debug(f"Loading checkpoints from directory {directory}")
    steps = {"flat_w": [], "loss": []}
    files = os.listdir(os.path.abspath(directory))
    files.sort(key=natural_keys)
    for filename in files:
        if re.search("step", filename):
            logger.debug(f"Loading from file {filename}")
            with open(os.path.join(os.path.abspath(directory), filename), "rb") as fd:
                try:
                    checkpoint = pickle.load(fd)
                    steps["flat_w"].append(checkpoint["flat_w"])
                    steps["loss"].append(checkpoint["loss"])
                except pickle.UnpicklingError:
                    continue

    return steps


def sample_path(steps, n_samples=300):
    """
    Function takes n_samples from steps dictionary

    :param steps: dictionary of sgd steps with members flat_w [] and loss []
    :param n_samples: number of samples to take
    :return: sampled dict
    """
    samples = {"flat_w": [], "loss": []}

    if n_samples > len(steps["flat_w"]):
        logger.warning(f"Less steps ({len(steps)} than final samples ({n_samples}). Using whole set of steps.")
        n_samples = len(steps)

    interval = len(steps["flat_w"]) // n_samples
    logger.debug(f"Samples interval: {interval}")
    count = 0
    for i in range(len(steps["flat_w"]) - 1, -1, -1):  # TODO investigate
        if i % interval == 0 and count < n_samples:
            samples["flat_w"].append(steps["flat_w"][i])
            samples["loss"].append(steps["loss"][i])
            count += 1

    #samples["flat_w"] = reversed(samples["flat_w"])
    #samples["loss"] = reversed(samples["loss"])
    return samples


def pca_dim_reduction(params):
    logger.debug("PCA dimension reduction")
    optim_path_np = []
    for tensor in params:
        optim_path_np.append(np.array(tensor[0].cpu()))

    pca = PCA(n_components=2)
    path_2d = pca.fit_transform(optim_path_np)
    reduced_dirs = pca.components_
    logger.debug(f"Reduced directions: {reduced_dirs}")
    logger.debug(f"PCA variances: {pca.explained_variance_ratio_}")

    return {
        "optim_path": optim_path_np,
        "path_2d": path_2d,
        "reduced_dirs": reduced_dirs,
        "pcvariances": pca.explained_variance_ratio_
    }


def calc_step(resolution, grid):
    dist_2d = grid[-1] - grid[0]
    dist = (dist_2d[0]**2 + dist_2d[1]**2)**0.5
    return dist * (1 + 0.3) / resolution


def compute_loss_2d(model, test_loader, device, params_grid):
    loss_2d = []
    n = len(params_grid)
    m = len(params_grid[0])
    loss_min = sys.float_info.max
    arg_min = ()

    logger.info("Calculating loss values for PCA directions")
    for i in range(n):
        loss_row = []
        for j in range(m):
            logger.debug(f"Calculating loss for coordinates: {i}, {j}")
            w_ij = torch.Tensor(params_grid[i][j].float()).to(device)

            model.load_from_flat_params(w_ij)
            loss, acc = net.test(model, test_loader, device)
            logger.debug(f"Loss for {i}, {j} = {loss}")
            if loss < loss_min:
                loss_min = loss
                logger.debug(f"New min loss {loss_min}")
                arg_min = (i, j)
            loss_row.append(loss)
        loss_2d.append(loss_row)

    loss_2darray = np.array(loss_2d).T
    return loss_2darray, arg_min, loss_min


def get_loss_grid(model, device, test_loader, resolution=50):
    grid_file = Path(os.path.join(pca_dirs, "loss_grid"))
    logger.info(f"Surface grid file {grid_file}")

    steps = get_steps(checkpoints)
    logger.debug(f"Steps len: {len(steps)}, type: {type(steps)}\n"
                 f"Steps flat_w len: {len(steps['flat_w'])}\n"
                 f"Steps loss len: {len(steps['loss'])}")
    sampled_optim_path = sample_path(steps)
    logger.debug(f"Sampled len: {len(sampled_optim_path)}, type: {type(sampled_optim_path)}\n"
                 f"Sample flat_w len: {len(sampled_optim_path['flat_w'])}\b"
                 f"Sample loss len: {len(sampled_optim_path['loss'])}")

    optim_path = sampled_optim_path["flat_w"]
    logger.debug(f"Optim path len: {len(optim_path)}")
    loss_path = sampled_optim_path["loss"]
    logger.debug(f"Loss path len : {len(loss_path)}")

    reduced_dict = pca_dim_reduction(optim_path)
    path_2d = reduced_dict["path_2d"]
    directions = reduced_dict["reduced_dirs"]
    pcvariances = reduced_dict["pcvariances"]

    d1 = directions[0]
    d2 = directions[1]

    optim_point = optim_path[-1]
    optim_point_2d = path_2d[-1]

    alpha = calc_step(resolution, path_2d)
    logger.debug(f"Step size: {alpha}")

    grid = []
    # prepare grid
    for i in range(-resolution, resolution):
        r = []
        for j in range(-resolution, resolution):
            updated = optim_point[0].cpu() + (i * d1 * alpha) + (j * d2 * alpha)
            r.append(updated)
        grid.append(r)

    if not grid_file.exists():
        loss, argmin, loss_min = compute_loss_2d(model, test_loader, device, grid)

        with open(grid_file, "wb") as fd:
            pickle.dump((loss, argmin, loss_min), fd)

    else:
        with open(grid_file, "rb") as fd:
            loss, argmin, loss_min = pickle.load(fd)

    coords = get_coords(alpha, resolution, optim_point_2d)

    return {
        "path_2d": path_2d,
        "loss_grid": loss,
        "argmin": argmin,
        "loss_min": loss_min,
        "coords": coords,
        "pcvariances": pcvariances
    }


def convert_coord(idx, ref_p, step_size):
    return idx * step_size + ref_p


def get_coords(step_size, resolution, optim_point2d):
    converted_x = []
    converted_y = []
    for i in range(-resolution, resolution):
        converted_x.append(convert_coord(i, optim_point2d[0], step_size))
        converted_y.append(convert_coord(i, optim_point2d[1], step_size))

    return converted_x, converted_y
