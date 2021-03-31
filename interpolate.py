import net
import copy
import h5py
import plot
import torch
import logging
import directions
import numpy as np
import scipy.optimize
from paths import *
from pathlib import Path


def convert_list2str(int_list):
    res = int(''.join(map(str, int_list)))

    return res


class Interpolator:
    def __init__(self, model, device, alpha, final_state_path, init_state_path):
        self.model = model
        self.device = device
        self.alpha = alpha
        self.theta = copy.deepcopy(torch.load(final_state_path))
        self.theta_f = copy.deepcopy(torch.load(final_state_path))
        self.theta_i = copy.deepcopy(torch.load(init_state_path))
        logging.debug("[interpolate]: model:"
                      "{}".format(model))
        logging.debug("[interpolate]: device: {}".format(device))
        logging.debug("[interpolate]: alpha: {}".format(alpha))
        logging.debug("[interpolate]: final state path: {}".format(final_state_path))
        logging.debug("[interpolate]: init state path: {}".format(init_state_path))

    def calc_distance(self, layer, idxs=None):
        """
        Method calculates distance between parameters

        :param layer: layer
        :return: distance
        """
        if not idxs:
            return torch.dist(self.theta_f[layer], self.theta_i[layer])
        else:
            return torch.dist(self.theta_f[layer][idxs], self.theta_i[layer][idxs])

    def calc_theta_single(self, layer, idxs, alpha):
        """
        Method calculates interpolation of a single parameter with respect to interpolation coefficient alpha

        :param layer: layer of parameter
        :param idxs: position of parameter
        :param alpha: interpolation coefficient
        """
        logging.debug("[interpolate]: Calculating value of: {} {} for alpha = {}".format(
            layer, idxs, alpha
        ))
        logging.debug("[interpolate]: {} {}".format(layer, idxs))
        logging.debug("[interpolate]: Theta:\n{}".format(self.theta[layer][idxs]))

        self.theta[layer][idxs] = (self.theta_i[layer][idxs] * (1.0 - alpha)) + (
                    self.theta_f[layer][idxs] * alpha)

    def calc_theta_vec(self, layer, alpha):
        """
        Method calculates interpolation of parameters of one layer with respect to interpolation coefficient alpha

        :param layer: layer
        :param alpha: interpolation coefficient
        """
        logging.debug("[interpolate]: Calculating value of: {} for alpha = {}".format(
            layer, alpha
        ))

        self.theta[layer] = torch.add((torch.mul(self.theta_i[layer], (1.0 - alpha))),
                                      torch.mul(self.theta_f[layer], alpha))

    @staticmethod
    def parabola(self, x, a, b, c):
        return a*x**2 + b*x + c

    def quadr(self, alpha, epochs, data):
        fit_params, pcov = scipy.optimize.curve_fit(self.parabola, epochs, data)
        approx = self.parabola(alpha, *fit_params)
        print(approx)

        np.savetxt("quadr", approx)

    def interpolate_all(self, test_loader):
        """
        Method interpolates all parameters of the model and after each interpolation step evaluates the
        performance of the model

        :param test_loader: test loader object
        """

        if not loss_path.exists() or not acc_path.exists():
            v_loss_list = []
            acc_list = []
            layers = ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias", "fc1.weight",
                      "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"]

            self.model.load_state_dict(self.theta_f)
            for alpha_act in self.alpha:
                for layer in layers:
                    self.calc_theta_vec(layer, alpha_act)
                    self.model.load_state_dict(self.theta)

                loss, acc = net.test(self.model, test_loader, self.device)
                v_loss_list.append(loss)
                acc_list.append(acc)

            np.savetxt(loss_path, v_loss_list)
            np.savetxt(acc_path, acc_list)

    def single_acc_vloss(self, test_loader, layer, idxs, trained=False):
        """
        Method interpolates individual parameter of the model and evaluates the model after each interpolation
        step

        :param test_loader: test loader
        :param layer: layer
        :param idxs: position of the parameter
        :param trained: show trained state TODO: to be deleted
        """

        loss_res = Path("{}_{}_{}".format(svloss_path, layer, convert_list2str(idxs)))
        loss_img = Path("{}_{}_{}".format(svloss_img_path, layer, convert_list2str(idxs)))
        acc_res = Path("{}_{}_{}".format(sacc_path, layer, convert_list2str(idxs)))
        acc_img = Path("{}_{}_{}".format(sacc_img_path, layer, convert_list2str(idxs)))
        dist = Path("{}_{}_{}_{}".format(svloss_path, layer, convert_list2str(idxs), "distance"))

        logging.debug("[interpolator]: Result files:\n{}\n{}".format(loss_res, acc_res))
        logging.debug("[interpolator]: Img files:\n{}\n{}".format(loss_img, acc_img))
        logging.debug("[interpolator]: Dist file:\n{}".format(dist))

        if not loss_res.exists() or not acc_res.exists():
            logging.debug("[interpolator.single_acc_vloss]: Files with results not found - beginning interpolation.")
            v_loss_list = []
            acc_list = []

            self.model.load_state_dict(self.theta_f)
            for alpha_act in self.alpha:
                self.calc_theta_single(layer + ".weight", idxs, alpha_act)

                self.model.load_state_dict(self.theta)
                logging.debug("[interpolator.single_acc_vloss]: Getting validation loss "
                              "and accuracy for alpha = {}".format(alpha_act))
                val_loss, acc = net.test(self.model, test_loader, self.device)
                acc_list.append(acc)
                v_loss_list.append(val_loss)

            logging.debug("[interpolator.single_acc_vloss]: Saving results to "
                          "files. ({}, {})".format(loss_res, acc_res))
            np.savetxt(loss_res, v_loss_list)
            np.savetxt(acc_res, acc_list)

        if not dist.exists():
            logging.info("[interpolate]: Calculating distance for: {} {}".format(layer, idxs))
            distance = self.calc_distance(layer + ".weight", idxs)
            logging.info("[interpolate]: Distance: {}".format(distance))
            with open(dist, 'w') as f:
                f.write("{}".format(distance))



        logging.debug("[interpolator.single_acc_vloss]: Saving results to figure {}, {} ...".format(loss_img, acc_img))
        plot.plot_one_param(self.alpha, np.loadtxt(loss_res), np.loadtxt(acc_res), loss_img, acc_img, trained=False)
        self.model.load_state_dict(self.theta_f)

        return

    def vec_acc_vlos(self, test_loader, layer, trained=False):
        """
        Method interpolates parameters of selected layer of the model and evaluates the model after each interpolation
        step

        :param test_loader: test loader
        :param layer: layer to be interpolated
        :param trained: show trained state
        """

        loss_res = Path("{}_{}".format(vvloss_path, layer))
        loss_img = Path("{}_{}".format(vvloss_img_path, layer))
        acc_res = Path("{}_{}".format(vacc_path, layer))
        acc_img = Path("{}_{}".format(vacc_img_path, layer))
        dist = Path("{}_{}_{}".format(vvloss_path, layer, "distance"))

        logging.debug("[interpolator]: Result files:\n{}\n{}".format(loss_res, acc_res))
        logging.debug("[interpolator]: Img files:\n{}\n{}".format(loss_img, acc_img))
        logging.debug("[interpolator]: Dist file:\n{}".format(dist))

        if not loss_res.exists() or not acc_res.exists():
            logging.debug("[interpolator.vec_acc_vloss]: Result files not found - beginning interpolation.")
            v_loss_list = []
            acc_list = []

            self.model.load_state_dict(self.theta_f)
            for alpha_act in self.alpha:
                self.calc_theta_vec(layer + ".weight", alpha_act)
                self.calc_theta_vec(layer + ".bias", alpha_act)

                self.model.load_state_dict(self.theta)
                logging.debug("[interpolator.vec_acc_vloss]: Getting "
                              "validation loss and accuracy for alpha = {}".format(alpha_act))
                vloss, acc = net.test(self.model, test_loader, self.device)
                v_loss_list.append(vloss)
                acc_list.append(acc)

            logging.debug("[interpolator.vec_acc_vloss]: Saving results to fiels. ({}, {})".format(loss_res, acc_res))
            np.savetxt(loss_res, v_loss_list)
            np.savetxt(acc_res, acc_list)

        if not dist.exists():
            logging.debug("[interpolator]: Calculating distance for {}".format(layer))
            distance = self.calc_distance(layer + ".weight")
            with open(dist, 'w') as f:
                f.write("{}".format(distance))

        logging.debug("[interpolator.vec_acc_vloss]: Saving results to figure {}, {} ...".format(loss_img, acc_img))
        plot.plot_one_param(self.alpha, np.loadtxt(loss_res), np.loadtxt(acc_res), loss_img, acc_img, trained=True)
        self.model.load_state_dict(self.theta_f)

        return

    def set_surf_file(self):
        """
        Method prepares file with loss function values in point of view of set directions
        """

        xmin, xmax, xnum = self.alpha[0], self.alpha[-1], len(self.alpha)
        ymin, ymax, ynum = self.alpha[0], self.alpha[-1], len(self.alpha)

        with h5py.File(surf, 'a') as fd:
            xcoord = np.linspace(xmin, xmax, xnum)
            fd["xcoordinates"] = xcoord

            ycoord = np.linspace(ymin, ymax, ynum)
            fd["ycoordinates"] = ycoord

            shape = (len(xcoord), len(ycoord))
            losses = -np.ones(shape=shape)

            fd["val_loss"] = losses

            return

    def get_indices(self, vals, xcoords, ycoords):
        idxs = np.array(range(vals.size))
        idxs = idxs[vals.ravel() <= 0]

        xcoords_mesh, ycoords_mesh = np.meshgrid(xcoords, ycoords)
        s1 = xcoords_mesh.ravel()[idxs]
        s2 = ycoords_mesh.ravel()[idxs]

        return idxs, np.c_[s1, s2]

    def update_weights(self, directions, step):
        dx = directions[0]
        dy = directions[1]
        theta_i = [p.data for p in self.model.parameters()]

        changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]

        for (p, w, d) in zip(self.model.parameters(), theta_i, changes):
            p.data = w.to(self.device) + d.clone().detach().requires_grad_(True)

    def rand_dirs(self, test_loader):
        if not surf.exists():
            self.set_surf_file()

        self.model.load_state_dict(self.theta_f)
        if surf.exists():
                dirs = directions.random_directions(self.model, self.device)

                with h5py.File(surf, "r+") as fd:
                    xcoords = fd["xcoordinates"][:]
                    ycoords = fd["ycoordinates"][:]
                    losses = fd["val_loss"][:]

                    idxs, coords = self.get_indices(losses, xcoords, ycoords)

                    for count, idx in enumerate(idxs):
                        coord = coords[count]

                        self.update_weights(dirs, coord)
                        loss, _ = net.test(self.model, test_loader, self.device)
                        losses.ravel()[idx] = loss
                        fd["val_loss"][:] = losses
                        fd.flush()

    def get_final_loss_acc(self, test_loader):
        """
        Method gets final validation loss and accuracy of the model

        :param test_loader: test loader
        :return: final validation loss, final accuracy
        """
        if sf_loss_path.exists() and sf_acc_path.exists():
            return np.loadtxt(sf_loss_path), np.loadtxt(sf_acc_path)

        if not self.model.load_state_dict(self.theta_f):
            print("[interpolator.get_final_loss_acc]: loading final state parameters has failed")
            return None

        loss, acc = net.test(self.model, test_loader, self.device)
        loss = np.broadcast_to(loss, self.alpha.shape)
        acc = np.broadcast_to(acc, self.alpha.shape)

        np.savetxt(sf_loss_path, loss)
        np.savetxt(sf_acc_path, acc)

        return loss, acc

    def print_theta(self, layer, idxs):
        """
        Method prints theta
        """
        layer = layer + ".weight"
        print(self.theta[layer][idxs])
