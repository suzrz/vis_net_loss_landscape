import os
import net
import h5py
import torch
import data_load
import numpy as np


def set_surf_file(filename):
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
        losses = np.ones(shape=shape)

        fd["val_loss"] = losses

        return


def get_indices(vals, xcoords, ycoords):
    inds = np.array(range(vals.size))
    inds = inds[vals.ravel() <= 0]

    xcoord_mesh, ycoord_mesh = np.meshgrid(xcoords, ycoords)
    s1 = xcoord_mesh.ravel()[inds]
    s2 = ycoord_mesh.ravel()[inds]

    return inds, np.c_[s1, s2]


def overwrite_weights(model, init_weights, directions, step, device):
    dx = directions[0]
    dy = directions[1]

    changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]

    for (p, w, d) in zip(model.parameters(), init_weights, changes):
        p.data = w.to(device) + torch.Tensor(d).to(device)


def calculate_loss(model, directions, device):
    filename = "3D_surf.h5"
    set_surf_file(filename)
    init_weights = [p.data for p in model.parameters()]

    with h5py.File(filename, "r+") as fd:
        xcoords = fd["xcoordinates"][:]
        ycoords = fd["ycoordinates"][:]
        losses = fd["val_loss"][:]

        inds, coords = get_indices(losses, xcoords, ycoords)

        for count, ind in enumerate(inds):
            coord = coords[count]
            print(coord[0])
            overwrite_weights(model, init_weights, directions, coord, device)

            loss = net.test(model, data_load.test_loader, device)

            losses.ravel()[ind] = loss

            fd["val_loss"][:] = losses

            fd.flush()

       # print(losses)