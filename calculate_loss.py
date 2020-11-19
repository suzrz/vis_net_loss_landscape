import os
import net
import copy
import h5py
import torch
import data_load
import numpy as np
from pathlib import Path

directory = "results"
trained_loss_path = Path(os.path.join(directory, "trained_loss"))
trained_accuracy_path = Path(os.path.join(directory, "trained_accuracy"))
validation_loss_path = Path(os.path.join(directory, "val_loss"))
training_loss_path = Path(os.path.join(directory, "training_loss"))
accuracy_path = Path(os.path.join(directory, "accuracy"))

def get_diff_in_success_for_subsets(model, optimizer, scheduler, device, directory, epochs=14):
    n_tr_samples = [60000, 50000, 40000, 30000, 20000, 10000]
    loss_list = []
    acc_list = []
    subset_losses_path = Path(os.path.join(directory, "subset_losses"))
    subset_accuracies_path = Path(os.path.join(directory, "subset_accs"))

    if not subset_losses_path.exists() or not subset_accuracies_path.exists():
        for samples in n_tr_samples:
            train_loader, test_loader = data_load.data_load(train_samples=samples)
            for epoch in range(1, epochs):
                net.train(model, train_loader, optimizer, device, epoch)
                net.test(model, test_loader, device)
                scheduler.step()
                print("Finished epoch no. ", epoch)
            loss, acc = net.test(model, test_loader, device)
            loss_list.append(loss)
            acc_list.append(acc)

        loss_list = np.array(loss_list, dtype=float)
        acc_list = np.array(acc_list, dtype=float)

        np.savetxt(subset_losses_path, loss_list)
        np.savetxt(subset_accuracies_path, acc_list)


def single(model, train_loader, test_loader, device, alpha, optimizer, final_state_path, init_state_path):
    """
    Calculate losses for one interpolation coefficient and one scalar parameter of network.

    :param test_loader:
    :param train_loader:
    :param init_state_path:
    :param final_state_path:
    :param directory:
    :param model:
    :param device:
    :param samples:
    :param optimizer:
    :return:
    """
    train_loss_list = []  # prepare clean list for train losses
    val_loss_list = []  # prepare clean list for validation losses
    accuracy_list = []  # prepare clean list for accuracy



    if not trained_loss_path.exists() or not trained_accuracy_path.exists():
        print("No trained loss and accuracy files found.\nGetting loss and accuracy...")
        if not model.load_state_dict(torch.load(final_state_path)):  # load final state of model
             print("[single: get trained loss and acc] Model parameters loading failed.")
        trained_loss, trained_accuracy = net.test(model, test_loader, device)  # get trained model loss and accuracy
        # broadcast to list for easier plotting
        trained_loss = np.broadcast_to(trained_loss, alpha.shape)
        trained_accuracy = np.broadcast_to(trained_accuracy, alpha.shape)

        trained_loss = np.array(trained_loss, dtype=float)
        trained_accuracy = np.array(trained_accuracy, dtype=float)

        np.savetxt(trained_loss_path, trained_loss)
        np.savetxt(trained_accuracy_path, trained_accuracy)

    if not validation_loss_path.exists() or not training_loss_path.exists() or \
            not accuracy_path.exists():
        theta = copy.deepcopy(torch.load(final_state_path))
        theta_f = copy.deepcopy(torch.load(final_state_path))
        theta_i = copy.deepcopy(torch.load(init_state_path))

        """INTERPOLATION"""
        for alpha_act in alpha:
            theta["conv2.weight"][4][0][0][0] = torch.add(
                theta_i["conv2.weight"][4][0][0][0] * (1.0 - alpha_act),
                theta_f["conv2.weight"][4][0][0][0] * alpha_act)

            if not model.load_state_dict(theta):
                print("Loading parameters to model failed.")  # loading parameters in model failed

            print("Getting train loss for alpha: ", alpha_act)
            train_loss = net.train(model, train_loader, optimizer, device, 0)
            train_loss_list.append(train_loss)

            print("Getting validation loss for alpha: ", alpha_act)
            val_loss, accuracy = net.test(model, test_loader, device)  # get loss with new parameters
            val_loss_list.append(val_loss)  # save obtained loss into list
            accuracy_list.append(accuracy)

        val_loss_list = np.array(val_loss_list, dtype=float)
        train_loss_list = np.array(train_loss_list, dtype=float)
        accuracy_list = np.array(accuracy_list, dtype=float)

        np.savetxt(validation_loss_path, val_loss_list)
        np.savetxt(training_loss_path, train_loss_list)
        np.savetxt(accuracy_path, accuracy_list)


def set_surf_file(filename):
    """
    Prepare surface file for 3D loss

    :param filename:
    :return:
    """
    print(filename)
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


def double(model, test_loader, directions, device, directory):
    """
    Calculate loss for two interpolation coefficients and whole set of network parameters

    :param directory:
    :param test_loader:
    :param model:
    :param directions:
    :param device:
    :return:
    """

    filename = "surf_3d.h5"
    file = Path(os.path.join(directory, filename))
    print("FUNC DOUBLE: ", file)
    set_surf_file(file)
    init_weights = [p.data for p in model.parameters()]

    if file.exists():
        print("FUNC DOUBLE: in if")
        with h5py.File(file, "r+") as fd:
            xcoords = fd["xcoordinates"][:]
            ycoords = fd["ycoordinates"][:]
            losses = fd["val_loss"][:]
            # print(losses, xcoords, ycoords)  # losses = ones, x, y ... ok

            inds, coords = get_indices(losses, xcoords, ycoords)

            print(inds, coords)  # [] [] HERE TODO
            for count, ind in enumerate(inds):
                coord = coords[count]

                overwrite_weights(model, init_weights, directions, coord, device)

                loss, _ = net.test(model, test_loader, device)
                print("COORD: ", coord, loss)

                losses.ravel()[ind] = loss

                fd["val_loss"][:] = losses

                fd.flush()
    else:
        print("H5 file doesn't exist")