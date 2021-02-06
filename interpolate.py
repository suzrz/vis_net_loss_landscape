import net
import copy
import plot
import torch
import logging
import numpy as np
from paths import *
from pathlib import Path

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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

    def calc_theta_single(self, layer, idxs, alpha):
        self.theta[layer][idxs] = (self.theta_i[layer][idxs] * (1.0 - alpha)) + (
                    self.theta_f[layer][idxs] * alpha)

    def calc_theta_vec(self, layer, alpha):
        self.theta[layer] = torch.add((torch.mul(self.theta_i[layer], (1.0 - alpha))), 
                                      torch.mul(self.theta_f[layer], alpha))

    def single_acc_vloss(self, test_loader, layer, idxs):
        loss_res = Path("{}_{}_{}".format(svloss_path, layer, convert_list2str(idxs)))
        loss_img = Path("{}_{}_{}".format(svloss_img_path, layer, convert_list2str(idxs)))
        acc_res = Path("{}_{}_{}".format(sacc_path, layer, convert_list2str(idxs)))
        acc_img = Path("{}_{}_{}".format(sacc_img_path, layer, convert_list2str(idxs)))

        if not loss_res.exists() or not acc_res.exists():
            logging.debug("[interpolator.single_acc_vloss]: Files with results not found - beginning interpolation.")
            v_loss_list = []
            acc_list = []

            self.model.load_state_dict(self.theta_f)
            for alpha_act in self.alpha:
                self.calc_theta_single(layer + ".weight", idxs, alpha_act)
                self.calc_theta_single(layer + ".bias", idxs, alpha_act)

                if not (self.model.load_state_dict(self.theta)):
                    logging.debug("[interpolator.single_acc_vloss]: Loading parameters into model failed (alpha = {}).".format(alpha_act))
                    return None

                logging.debug("[interpolator.single_acc_vloss]: Getting validation loss and accuracy for alpha = {}".format(alpha_act))
                val_loss, acc = net.test(self.model, test_loader, self.device)
                acc_list.append(acc)
                v_loss_list.append(val_loss)

            logging.debug("[interpolator.single_acc_vloss]: Saving results to files. ({}, {})".format(loss_res, acc_res))
            np.savetxt(loss_res, v_loss_list)
            np.savetxt(acc_res, acc_list)

        logging.debug("[interpolator.single_acc_vloss]: Plotting {} {}...".format(layer, idxs))
        plot.plot_one_param(self.alpha, np.loadtxt(loss_res), np.loadtxt(acc_res), loss_img, acc_img)
        self.model.load_state_dict(self.theta_f)

        return

    def vec_acc_vlos(self, test_loader, layer):
        if vvloss_path.exists() or vacc_path.exists():
            return np.loadtxt(vvloss_path), np.loadtxt(vacc_path)

        v_loss_list = []
        acc_list = []

        self.model.load_state_dict(self.theta_f)
        for alpha_act in self.alpha:
            self.calc_theta_vec(layer + ".weight", alpha_act)

            vloss, acc = net.test(self.model, test_loader, self.device)
            v_loss_list.append(vloss)
            acc_list.append(acc)

        np.savetxt(vvloss_path, v_loss_list)
        np.savetxt(vacc_path, acc_list)
        self.model.load_state_dict(self.theta_f)

        return v_loss_list, vacc_path

    def get_final_loss_acc(self, test_loader):
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