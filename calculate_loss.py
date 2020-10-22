import os
import net
import copy
import h5py
import torch
import pickle
import data_load
import numpy as np


def single(model, device, samples, optimizer):
    """
    Calculate losses for one interpolation coefficient and one scalar parameter of network.

    :param model:
    :param device:
    :param samples:
    :param optimizer:
    :return:
    """
    alpha = np.linspace(-0.25, 1.5, samples)  # set interpolation coefficient
    train_loss_list = []  # prepare clean list for train losses
    val_loss_list = []  # prepare clean list for validation losses

    if not os.path.isfile("trained_net_loss.txt"):
        print("No fil ")
        model.load_state_dict(torch.load("final_state.pt"))
        trained_loss = net.test(model, data_load.test_loader, device)
        trained_loss = np.broadcast_to(trained_loss, alpha.shape)
        with open("trained_net_loss.txt", "wb") as fd:
            pickle.dump(trained_loss, fd)

    if not os.path.isfile("v_loss_list.txt") or not os.path.isfile("t_loss_list.txt"):
        theta = copy.deepcopy(torch.load("final_state.pt"))
        theta_f = copy.deepcopy(torch.load("final_state.pt"))
        theta_i = copy.deepcopy(torch.load("init_state.pt"))

        for alpha_act in alpha:  # interpolate
            theta["conv2.weight"][4][0][0][0] = copy.copy(torch.add(
                theta_i["conv2.weight"][4][0][0][0] * (1.0 - alpha_act),
                theta_f["conv2.weight"][4][0][0][0] * alpha_act))
            if not model.load_state_dict(theta):
                print("Something went wrong.")  # loading parameters in model failed

            print("ALPHA: ", alpha_act)
            print("Getting train loss")
            train_loss = net.train(model, data_load.train_loader, optimizer, device, 0)
            train_loss_list.append(train_loss)

            print("Getting val loss")
            val_loss = net.test(model, data_load.test_loader, device)  # get loss with new parameters
            val_loss_list.append(val_loss)  # save obtained loss into list

        with open("v_loss_list.txt", "wb") as fd:
            pickle.dump(val_loss_list, fd)

        with open("t_loss_list.txt", "wb") as fd:
            pickle.dump(train_loss_list, fd)


def set_surf_file(filename):
    """
    Prepare surface file for 3D loss

    :param filename:
    :return:
    """
    xmin, xmax, xnum = -1, 1, 51
    ymin, ymax, ynum = -1, 1, 51

    if os.path.isfile(filename):
        return

    with h5py.File(filename, 'a') as fd:

        xcoord = np.linspace(xmin, xmax, xnum)
        fd["xcoordinates"] = xcoord

        ycoord = np.linspace(ymin, ymax, ynum)
        fd["ycoordinates"] = ycoord

        shape = (len(xcoord), len(ycoord))
        # print(shape)
        losses = -np.ones(shape=shape)

        fd["val_loss"] = losses

        return


def get_indices(vals, xcoords, ycoords):
    """

    :param vals:
    :param xcoords:
    :param ycoords:
    :return:
    """
    inds = np.array(range(vals.size))
    print(type(inds))
    print(inds)  # 0 ... 2600
    inds = inds[vals.ravel() <= 0]
    print(vals.ravel())
    print(inds)

    xcoord_mesh, ycoord_mesh = np.meshgrid(xcoords, ycoords)
    s1 = xcoord_mesh.ravel()[inds]
    s2 = ycoord_mesh.ravel()[inds]

    return inds, np.c_[s1, s2]


def overwrite_weights(model, init_weights, directions, step, device):
    """

    :param model:
    :param init_weights:
    :param directions:
    :param step:
    :param device:
    :return:
    """
    dx = directions[0]
    dy = directions[1]

    changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]

    for (p, w, d) in zip(model.parameters(), init_weights, changes):
        p.data = w.to(device) + torch.Tensor(d).to(device)


def double(model, directions, device):
    """
    Calculate loss for two interpolation coefficients and whole set of network parameters

    :param model:
    :param directions:
    :param device:
    :return:
    """
    filename = "3D_surf.h5"
    set_surf_file(filename)
    init_weights = [p.data for p in model.parameters()]

    if not os.path.isfile(filename):
        with h5py.File(filename, "r+") as fd:
            xcoords = fd["xcoordinates"][:]
            ycoords = fd["ycoordinates"][:]
            losses = fd["val_loss"][:]
            # print(losses, xcoords, ycoords)  # losses = ones, x, y ... ok

            inds, coords = get_indices(losses, xcoords, ycoords)

            print(inds, coords)  # [] [] HERE TODO
            for count, ind in enumerate(inds):
                coord = coords[count]
                print("COORD: ", coord)
                overwrite_weights(model, init_weights, directions, coord, device)

                loss = net.test(model, data_load.test_loader, device)

                losses.ravel()[ind] = loss

                fd["val_loss"][:] = losses

                fd.flush()
