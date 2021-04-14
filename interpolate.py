import itertools
import net
import copy
import h5py
import plot
import torch
import surface
import numpy as np
import scipy.interpolate
from paths import *
from pathlib import Path
from numpy.polynomial import Polynomial


logger = logging.getLogger("vis_net")


def convert_list2str(int_list):
    res = int(''.join(map(str, int_list)))

    return res


def parabola(x, a, b, c):
    return a*x**2 + b*x + c


class Interpolator:
    def __init__(self, model, device, alpha, final_state_path, init_state_path):
        self.model = model
        self.device = device
        self.alpha = alpha
        self.theta = copy.deepcopy(torch.load(final_state_path))
        self.theta_f = copy.deepcopy(torch.load(final_state_path))
        self.theta_i = copy.deepcopy(torch.load(init_state_path))

        logger.debug(f"Model: "
                     f"{model}")
        logger.debug(f"Device: {device}")
        logger.debug(f"Alpha: {alpha}")
        logger.debug(f"Final state path: {final_state_path}")
        logger.debug(f"Init state path: {init_state_path}")

    def calc_distance(self, layer, idxs=None):
        """
        Method calculates distance between parameters

        :param layer: layer
        :param idxs: position of parameter
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
        logger.debug(f"Calculating: {layer} {idxs} for alpha = {alpha}")

        self.theta[layer][idxs] = (self.theta_i[layer][idxs] * (1.0 - alpha)) + (
                    self.theta_f[layer][idxs] * alpha)

        logger.debug(f"Modified theta:\n"
                     f"{self.theta[layer][idxs]}")

    def calc_theta_single_q(self, layer, idxs, alpha, start, mid, end):
        """
        Method calculates quadratic interpolation of a single parameter with respect to interpolation coefficient
        alpha

        :param layer: layer of parameter
        :param idxs: position of parameter
        :param alpha: interpolation coefficient value
        :param start: first point
        :param mid: second point
        :param end: ending point
        """
        logger.debug(f"Calculating quadr: {layer} {idxs} for alpha = {alpha}")
        xdata = np.array([start[0], mid[0], end[0]])
        logger.debug(f"XDATA: {xdata}")
        ydata = np.array([start[1], mid[1], end[1]])
        logger.debug(f"YDATA: {ydata}")
        #self.fit_params, self.p_cov = scipy.optimize.curve_fit(parabola, xdata, ydata)

        poly = scipy.interpolate.lagrange(xdata, ydata)

        self.fit_params = Polynomial(poly).coef
        logger.debug(f"Coefficients: {self.fit_params}")

        self.theta[layer][idxs] = torch.tensor((self.fit_params[0]*(alpha**2) + self.fit_params[1]*alpha +
                                                   self.fit_params[2])).to(self.device)
        logger.debug(f"Modified theta:\n"
                     f"{self.theta[layer][idxs]}")


    def calc_theta_vec(self, layer, alpha):
        """
        Method calculates interpolation of parameters of one layer with respect to interpolation coefficient alpha

        :param layer: layer
        :param alpha: interpolation coefficient
        """
        logger.debug(f"Calculating: {layer} for alpha = {alpha}")

        self.theta[layer] = torch.add((torch.mul(self.theta_i[layer], (1.0 - alpha))),
                                      torch.mul(self.theta_f[layer], alpha))

    def calc_theta_vec_q(self, layer, alpha, start, mid, end):
        logger.debug(f"Calculating quadr: {layer} for alpha = {alpha}")
        xdata = np.array([start[0], mid[0], end[0]])
        logger.debug(f"XDATA: {xdata}")
        ydata = np.array([start[1], mid[1], end[1]])
        logger.debug(f"YDATA: {ydata}")
        #self.fit_params, self.p_cov = scipy.optimize.curve_fit(parabola, xdata, ydata)

        poly = scipy.interpolate.lagrange(xdata, ydata)

        self.fit_params = Polynomial(poly).coef
        logger.debug(f"Coefficients: {self.fit_params}")

        self.theta[layer] = torch.tensor(((1.0 - alpha)*self.fit_params[0]**2 + alpha*self.fit_params[1] +
                                                self.fit_params[2]) / 100).to(self.device)
        logger.debug(f"Modified theta:\n"
                     f"{self.theta[layer]}")

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
            self.model.load_state_dict(self.theta_f)

    def single_acc_vloss(self, test_loader, layer, idxs):
        """
        Method interpolates individual parameter of the model and evaluates the model after each interpolation
        step

        :param test_loader: test loader
        :param layer: layer
        :param idxs: position of the parameter
        """

        loss_res = Path("{}_{}_{}".format(svloss_path, layer, convert_list2str(idxs)))
        loss_img = Path("{}_{}_{}".format(svloss_img_path, layer, convert_list2str(idxs)))

        acc_res = Path("{}_{}_{}".format(sacc_path, layer, convert_list2str(idxs)))
        acc_img = Path("{}_{}_{}".format(sacc_img_path, layer, convert_list2str(idxs)))

        dist = Path("{}_{}_{}_{}".format(svloss_path, layer, convert_list2str(idxs), "distance"))

        logger.debug(f"Result files:\n"
                     f"{loss_res}\n"
                     f"{acc_res}\n")
        logger.debug(f"Img files:\n"
                     f"{loss_img}\n"
                     f"{acc_img}\n")
        logger.debug(f"Dist file:\n"
                     f"{dist}\n")

        if not loss_res.exists() or not acc_res.exists():
            logger.debug("Files with results not found - beginning interpolation.")

            v_loss_list = []
            acc_list = []

            self.model.load_state_dict(self.theta_f)
            for alpha_act in self.alpha:
                self.calc_theta_single(layer + ".weight", idxs, alpha_act)

                self.model.load_state_dict(self.theta)

                logger.debug(f"Getting validation loss and accuracy for alpha = {alpha_act}")
                val_loss, acc = net.test(self.model, test_loader, self.device)
                acc_list.append(acc)
                v_loss_list.append(val_loss)

            logger.debug(f"Saving results to files ({loss_res}, {acc_res})")

            np.savetxt(loss_res, v_loss_list)
            np.savetxt(acc_res, acc_list)
            self.model.load_state_dict(self.theta_f)

        if not dist.exists():
            logger.info(f"Calculating distance for: {layer} {idxs}")

            distance = self.calc_distance(layer + ".weight", idxs)
            logger.info(f"Distance: {distance}")

            with open(dist, 'w') as f:
                f.write("{}".format(distance))

        logger.debug(f"Saving results to figures {loss_img}, {acc_img} ...")
        plot.plot_one_param(self.alpha, np.loadtxt(loss_res), np.loadtxt(acc_res), loss_img, acc_img)

        self.model.load_state_dict(self.theta_f)

        return

    def single_acc_vloss_q(self, test_loader, layer, idxs):
        """
        Method interpolates individual parameter of the model and evaluates the performance of the model when the
        interpolated parameter replaces its original in the parameters of the model

        :param test_loader: test dataset loader
        :param layer: layer of parameter
        :param idxs: position of parameter
        """

        loss_res = Path("{}_{}_{}_q".format(svloss_path, layer, convert_list2str(idxs)))
        loss_img = Path("{}_{}_{}_q".format(svloss_img_path, layer, convert_list2str(idxs)))

        acc_res = Path("{}_{}_{}_q".format(sacc_path, layer, convert_list2str(idxs)))
        acc_img = Path("{}_{}_{}_q".format(sacc_img_path, layer, convert_list2str(idxs)))

        logger.debug(f"Result files:\n"
                     f"{loss_res}\n"
                     f"{acc_res}\n")
        logger.debug(f"Img files:\n"
                     f"{loss_img}\n"
                     f"{acc_img}\n")

        if not loss_res.exists() or not acc_res.exists():
            logger.debug("Files with results not found - beginning interpolation.")

            v_loss_list = []
            acc_list = []

            start_a = 0
            mid_a = 0.5
            end_a = 1
            logger.debug(f"Start: {start_a}\n"
                         f"Mid: {mid_a}\n"
                         f"End: {end_a}")


            start_p = self.theta_i[layer + ".weight"][idxs].cpu()
            mid_p = copy.deepcopy(torch.load(Path(os.path.join(results, "state_7"))))[layer + ".weight"][idxs].cpu()
            end_p = self.theta_f[layer + ".weight"][idxs].cpu()
            #start_loss = np.loadtxt(actual_loss_path)[0]
            #mid_loss = np.loadtxt(actual_loss_path)[6]
            #end_loss = np.loadtxt(actual_loss_path)[-1]
            logger.debug(f"Start loss: {start_p}\n"
                         f"Mid loss: {mid_p}\n"
                         f"End loss: {end_p}")

            start = [start_a, start_p]
            mid = [mid_a, mid_p]
            end = [end_a, end_p]
            logger.debug(f"Start: {start}\n"
                         f"Mid: {mid}\n"
                         f"End: {end}")

            self.model.load_state_dict(self.theta_f)
            for alpha_act in self.alpha:
                self.calc_theta_single_q(layer + ".weight", idxs, alpha_act, start, mid, end)

                self.model.load_state_dict(self.theta)

                logger.debug(f"Getting validation loss and accuracy for alpha = {alpha_act}")
                val_loss, acc = net.test(self.model, test_loader, self.device)
                acc_list.append(acc)
                v_loss_list.append(val_loss)

            logger.debug(f"Saving results to files ({loss_res}, {acc_res})")

            np.savetxt(loss_res, v_loss_list)
            np.savetxt(acc_res, acc_list)
            self.model.load_state_dict(self.theta_f)

        logger.debug(f"Saving results to figures {loss_img}, {acc_img} ...")
        plot.plot_one_param(self.alpha, np.loadtxt(loss_res), np.loadtxt(acc_res), loss_img, acc_img, show=False)

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

        logger.debug(f"Result files:\n"
                     f"{loss_res}\n"
                     f"{acc_res}")
        logger.debug(f"Img files:\n"
                     f"{loss_img}\n"
                     f"{acc_img}")
        logger.debug(f"Dist file:\n"
                     f"{dist}")

        if not loss_res.exists() or not acc_res.exists():
            logger.debug("Result files not found - beginning interpolation.")

            v_loss_list = []
            acc_list = []

            self.model.load_state_dict(self.theta_f)
            for alpha_act in self.alpha:
                self.calc_theta_vec(layer + ".weight", alpha_act)
                self.calc_theta_vec(layer + ".bias", alpha_act)

                self.model.load_state_dict(self.theta)
                logger.debug(f"Getting validation loss and accuracy for alpha = {alpha_act}")

                vloss, acc = net.test(self.model, test_loader, self.device)
                v_loss_list.append(vloss)
                acc_list.append(acc)

            logger.debug(f"Saving results to files ({loss_res}, {acc_res})")
            np.savetxt(loss_res, v_loss_list)
            np.savetxt(acc_res, acc_list)

        if not dist.exists():
            logger.info(f"Calculating distance for: {layer}")

            distance = self.calc_distance(layer + ".weight")
            logger.info(f"Distance: {distance}")

            with open(dist, 'w') as f:
                f.write("{}".format(distance))

        logger.debug(f"Saving results to figures {loss_img}, {acc_img} ...")
        plot.plot_one_param(self.alpha, np.loadtxt(loss_res), np.loadtxt(acc_res), loss_img, acc_img, trained=trained)

        self.model.load_state_dict(self.theta_f)

        return

    def vec_acc_vloss_q(self, test_loader, layer, trained=False):
        loss_res = Path("{}_{}_q".format(vvloss_path, layer))
        loss_img = Path("{}_{}_q".format(vvloss_img_path, layer))

        acc_res = Path("{}_{}_q".format(vacc_path, layer))
        acc_img = Path("{}_{}_q".format(vacc_img_path, layer))

        logger.debug(f"Result files:\n"
                     f"{loss_res}\n"
                     f"{acc_res}")
        logger.debug(f"Img files:\n"
                     f"{loss_img}\n"
                     f"{acc_img}")

        if not loss_res.exists() or not acc_res.exists():
            logger.debug("Result files not found - beginning interpolation.")

            v_loss_list = []
            acc_list = []

            start_a = 0
            mid_a = 0.5
            end_a = 1
            logger.debug(f"Start: {start_a}\n"
                         f"Mid: {mid_a}\n"
                         f"End: {end_a}")

            if layer == "conv1":
                aux = [list(np.arange(0, 6)), [0], list(np.arange(0, 3)), list(np.arange(0, 3))]
                idxs = list(itertools.product(*aux))
            elif layer == "conv2":
                aux = [list(np.arange(0, 6)), list(np.arange(0, 6)), list(np.arange(0, 3)), list(np.arange(0, 3))]
                idxs = list(itertools.product(*aux))
            elif layer == "fc1":
                aux = [list(np.arange(0, 120)), list(np.arange(0, 576))]
                idxs = list(itertools.product(*aux))
            elif layer == "fc2":
                aux = [list(np.arange(0, 84)), list(np.arange(0, 120))]
                idxs = list(itertools.product(*aux))
            elif layer == "fc3":
                aux = [list(np.arange(0, 10)), list(np.arange(0, 84))]
                idxs = list(itertools.product(*aux))

            self.model.load_state_dict(self.theta_f)

            for alpha_act in self.alpha:
                for i in idxs:
                    try:
                        start_p = self.theta_i[layer + ".weight"][i].cpu()
                        mid_p = copy.deepcopy(torch.load(Path(os.path.join(results, "state_7"))))[layer + ".weight"][i].cpu()
                        end_p = self.theta_f[layer + ".weight"][i].cpu()
                        # start_loss = np.loadtxt(actual_loss_path)[0]
                        # mid_loss = np.loadtxt(actual_loss_path)[6]
                        # end_loss = np.loadtxt(actual_loss_path)[-1]
                        logger.debug(f"Start loss: {start_p}\n"
                                     f"Mid loss: {mid_p}\n"
                                     f"End loss: {end_p}")

                        start = [start_a, start_p]
                        mid = [mid_a, mid_p]
                        end = [end_a, end_p]
                        logger.debug(f"Start: {start}\n"
                                     f"Mid: {mid}\n"
                                     f"End: {end}")

                        print(idxs)

                        self.calc_theta_single_q(layer + ".weight", i, alpha_act, start, mid, end)
                    #self.calc_theta_vec_q(layer + ".bias", alpha_act)  # TODO start, mid, end pro bias
                    except IndexError:
                        continue

                self.model.load_state_dict(self.theta)
                logger.debug(f"Getting validation loss and accuracy for alpha = {alpha_act}")

                vloss, acc = net.test(self.model, test_loader, self.device)
                v_loss_list.append(vloss)
                acc_list.append(acc)

            logger.debug(f"Saving results to files ({loss_res}, {acc_res})")
            np.savetxt(loss_res, v_loss_list)
            np.savetxt(acc_res, acc_list)

        logger.debug(f"Saving results to figures {loss_img}, {acc_img} ...")
        plot.plot_one_param(self.alpha, np.loadtxt(loss_res), np.loadtxt(acc_res), loss_img, acc_img, trained=trained)

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
            dirs = surface.random_directions(self.model, self.device)

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
