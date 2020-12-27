import net
import copy
import torch
import numpy as np
import data_load  # only for testing, delete after program validation
from torch import optim as optim  # only for testing, delete after program validation
from paths import *
from torch.optim.lr_scheduler import StepLR


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

    def single_vloss(self, test_loader, layer, idxs):
        v_loss_list = []

        if not svloss_path.exists():
            for alpha_act in self.alpha:
                self.calc_theta_single(layer + ".weight", idxs, alpha_act)

                if not (self.model.load_state_dict(self.theta)):
                    print("LOADING PARAMETERS INTO MODEL FAILED")
                    return None

                print("Validation loss for alpha:", alpha_act, "...")
                val_loss, _ = net.test(self.model, test_loader, self.device)
                print(val_loss)
                v_loss_list.append(val_loss)

            np.savetxt(svloss_path, v_loss_list)

        self.model.load_state_dict(self.theta_f)
        return v_loss_list

    def single_acc(self, test_loader, layer, idxs):
        acc_list = []

        if not sacc_path.exists():
            for alpha_act in self.alpha:
                self.calc_theta_single(layer + ".weight", idxs, alpha_act)

                if not (self.model.load_state_dict(self.theta)):
                    print("LOADING PARAMETERS INTO MODEL FAILED")
                    return None

                print("Validation loss for alpha:", alpha_act, "...")
                _, acc = net.test(self.model, test_loader, self.device)
                print(acc)
                acc_list.append(acc)

            np.savetxt(sacc_path, acc_list)

        self.model.load_state_dict(self.theta_f)
        return acc_list

    def single_acc_vloss(self, test_loader, layer, idxs):
        acc_list = []
        v_loss_list = []

        if not sacc_path.exists() or not svloss_path.exists():
            for alpha_act in self.alpha:
                self.calc_theta_single(layer + ".weight", idxs, alpha_act)

                if not (self.model.load_state_dict(self.theta)):
                    print("LOADING PARAMETERS INTO MODEL FAILED")
                    return None

                print("ALPHA:", alpha_act)
                val_loss, acc = net.test(self.model, test_loader, self.device)
                acc_list.append(acc)
                v_loss_list.append(val_loss)

            np.savetxt(sacc_path, acc_list)
            np.savetxt(svloss_path, v_loss_list)

        self.model.load_state_dict(self.theta_f)
        return acc_list, v_loss_list

    def single_tloss(self, train_loader, optimizer, layer, idxs):
        t_loss_list = []

        if not stloss_path.exists():
            for alpha_act in self.alpha:
                self.calc_theta_single(layer + ".weight", idxs, alpha_act)

                if not self.model.load_state_dict(self.theta):
                    print("LOADING PARAMETERS INTO MODEL FAILED")
                    return None

                print("Training loss for alpha:", alpha_act, "...")
                t_loss = net.train(self.model, train_loader, optimizer, self.device, 0)
                print(t_loss)
                t_loss_list.append(t_loss)

            np.savetxt(stloss_path, t_loss_list)

        self.model.load_state_dict(self.theta_f)
        return t_loss_list

    def vec_vloss(self, test_loader, layer):
        v_loss_list = []

        if not vvloss_path.exists():
            for alpha_act in self.alpha:
                self.calc_theta_vec(layer + ".weight", alpha_act)

                if not self.model.load_state_dict(self.theta):
                    print("LOADING PARAMETERS INTO MODEL FAILED")
                    return None
                vloss, _ = net.test(self.model, test_loader, self.device)
                print(vloss)
                v_loss_list.append(vloss)
            np.savetxt(vvloss_path, v_loss_list)

        self.model.load_state_dict(self.theta_f)
        return v_loss_list

    def vec_acc(self, test_loader, layer):
        acc_list = []

        if not vvloss_path.exists():
            for alpha_act in self.alpha:
                self.calc_theta_vec(layer + ".weight", alpha_act)

                if not self.model.load_state_dict(self.theta):
                    print("LOADING PARAMETERS INTO MODEL FAILED")
                    return None
                _, acc = net.test(self.model, test_loader, self.device)
                print(acc)
                acc_list.append(acc)
            np.savetxt(vacc_path, acc_list)

        self.model.load_state_dict(self.theta_f)
        return acc_list

    def vec_tloss(self, train_loader, optimizer, layer):
        t_loss_list = []

        if not stloss_path.exists():
            for alpha_act in self.alpha:
                self.calc_theta_vec(layer + ".weight", alpha_act)

                if not self.model.load_state_dict(self.theta):
                    print("LOADING PARAMETERS INTO MODEL FAILED")
                    return None

                print("Training loss for alpha:", alpha_act, "...")
                t_loss = net.train(self.model, train_loader, optimizer, self.device, 0)
                print(t_loss)
                t_loss_list.append(t_loss)

            np.savetxt(vtloss_path, t_loss_list)

        self.model.load_state_dict(self.theta_f)
        return t_loss_list

    def get_final_loss(self, test_loader):
        self.model.load_state_dict(self.theta_f)

        loss, _ = net.test(self.model, test_loader, self.device)
        loss = np.broadcast_to(loss, self.alpha.shape)

        np.savetxt(sf_loss_path, loss)

        return loss

    def get_final_acc(self, test_loader):
        self.model.load_state_dict(self.theta_f)

        _, acc = net.test(self.model, test_loader, self.device)
        acc = np.broadcast_to(acc, self.alpha.shape)

        np.savetxt(sf_acc_path, acc)

        return acc

    def get_final_loss_acc(self, test_loader, loss_only=False, acc_only=False):
        if not self.model.load_state_dict(self.theta_f):
            print("[interpolator] - get_final_loss_acc: loading final state parameters has failed")
            return None

        loss, acc = net.test(self.model, test_loader, self.device)
        loss = np.broadcast_to(loss, self.alpha.shape)
        acc = np.broadcast_to(acc, self.alpha.shape)

        np.savetxt(sf_loss_path, loss)
        np.savetxt(sf_acc_path, acc)

        return loss, acc

    def get_train_subset_impact(self, subset_list, epochs, test_loader):
        loss_list = []
        acc_list = []

        for n_samples in subset_list:
            self.model.load_state_dict(self.theta_i)

            optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5)  # set optimizer
            scheduler = StepLR(optimizer, step_size=1, gamma=0.7)  # set scheduler

            for epoch in range(1, epochs):
                train_loader, test_loader = data_load.data_load(train_samples=n_samples)

                net.train(self.model, train_loader, optimizer, self.device, epoch)
                net.test(self.model, test_loader, self.device)

                scheduler.step()
                print("[interpolator] - get_subset_perf : Finished epoch", epoch, "for training set of size:", n_samples)

            loss, acc = net.test(self.model, test_loader, self.device)

            loss_list.append(loss)
            acc_list.append(acc)

        np.savetxt(train_subs_loss, loss_list)
        np.savetxt(train_subs_acc, acc_list)

        self.model.load_state_dict(self.theta_f)

    def get_test_subset_impact(self, subset_list):
        loss_list = []
        acc_list = []

        self.model.load_state_dict(self.theta_f)


        for n_samples in subset_list:
            loss_s = []
            acc_s = []
            for x in range(int((10000/n_samples)*2)):
                _, test_loader = data_load.data_load(test_samples=n_samples)
                loss, acc = net.test(self.model, test_loader, self.device)

                loss_s.append(loss)
                acc_s.append(acc)

            loss_avg = sum(loss_s)/len(loss_s)
            acc_avg = sum(acc_s)/len(acc_s)

            loss_list.append(loss_avg)
            acc_list.append(acc_avg)
            #print("[interpolator] - get_stability : subset", n_samples, "| loss", loss, "| acc", acc)

        np.savetxt(test_subs_loss, loss_list)
        np.savetxt(test_subs_acc, acc_list)


"""
init_state = Path(os.path.join(directory, "init_state.pt"))
final_state = Path(os.path.join(directory, "final_state.pt"))

model = net.Net().to("cuda")
model.load_state_dict(torch.load(final_state))

train_loader, test_loader = data_load.data_load()

interpolate = Interpolator(model, "cuda", np.linspace(-1, 1, 20), final_state, init_state)
v = interpolate.single_vloss(test_loader, "conv2", [4, 0, 10, 0])
print("VALIDATION LOSS:", v)

a = interpolate.single_acc(test_loader, "conv2", [4, 0, 10, 0])
print("ACCURACY:", a)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # set optimizer
t = interpolate.single_tloss(train_loader, optimizer, "conv2", [4, 0, 10, 0])
print("TRAINING LOSS:", t)

vv = interpolate.vec_vloss(test_loader, "conv2")
print("VECTOR VLOSS:", vv)

va = interpolate.vec_acc(test_loader, "conv2")
print("VECTOR ACC:", va)

vt = interpolate.vec_tloss(train_loader, optimizer, "conv2")
print("VECTOR TLOSS:", vt)
"""