import os
import net
import copy
import torch
import numpy as np
from pathlib import Path
import data_load  # only for testing, delete after program validation
from torch import optim as optim  # only for testing, delete after program validation
from calculate_loss import directory, trained_loss_path, trained_accuracy_path, validation_loss_path, training_loss_path, accuracy_path

class Interpolator():
    def __init__(self, model, device,  final_state_path, init_state_path):
        self.model = model
        self.device = device
        self.theta = copy.deepcopy(torch.load(final_state_path))
        self.theta_f = copy.deepcopy(torch.load(final_state_path))
        self.theta_i = copy.deepcopy(torch.load(init_state_path))

    def calc_theta_single(self, layer, idxs, alpha):
        self.theta[layer][idxs] = (self.theta_i[layer][idxs] * (1.0 - alpha)) + (
                    self.theta_f[layer][idxs] * (alpha))


    def calc_theta_vec(self, layer, alpha):
        self.theta[layer] = torch.add((torch.mul(self.theta_i[layer], (1.0 - alpha))), 
                                      torch.mul(self.theta_f[layer], alpha))


    def single_vloss(self, test_loader, alpha, layer, idxs):
        v_loss_list = []

        if not validation_loss_path.exists():
            for alpha_act in alpha:
                self.calc_theta_single(layer + ".weight", idxs, alpha_act)

                if not (self.model.load_state_dict(self.theta)):
                    print("LOADING PARAMETERS INTO MODEL FAILED")
                    return None

                print("Validation loss for alpha:", alpha_act, "...")
                val_loss, _ = net.test(self.model, test_loader, self.device)
                print(val_loss)
                v_loss_list.append(val_loss)

            np.savetxt(validation_loss_path, v_loss_list)
        return v_loss_list


    def single_acc(self, test_loader, alpha, layer, idxs):
        acc_list = []

        if not accuracy_path.exists():
            for alpha_act in alpha:
                self.calc_theta_single(layer + ".weight", idxs, alpha_act)

                if not (self.model.load_state_dict(self.theta)):
                    print("LOADING PARAMETERS INTO MODEL FAILED")
                    return None

                print("Validation loss for alpha:", alpha_act, "...")
                _, acc = net.test(self.model, test_loader, self.device)
                print(acc)
                acc_list.append(acc)

            np.savetxt(accuracy_path, acc_list)
        return acc_list


    def single_acc_vloss(self, test_loader, alpha, layer, idxs):
        acc_list = []
        v_loss_list = []

        if not accuracy_path.exists() or not validation_loss_path.exists():
            for alpha_act in alpha:
                self.calc_theta_single(layer + ".weight", idxs, alpha_act)

                if not (self.model.load_state_dict(self.theta)):
                    print("LOADING PARAMETERS INTO MODEL FAILED")
                    return None

                print("ALPHA:", alpha_act)
                val_loss, acc = net.test(self.model, test_loader, self.device)
                acc_list.append(acc)
                v_loss_list.append(val_loss)

            np.savetxt(accuracy_path, acc_list)
            np.savetxt(validation_loss_path, v_loss_list)

        return acc_list, v_loss_list


    def single_tloss(self, train_loader, optimizer, alpha, layer, idxs):
        t_loss_list = []

        if not training_loss_path.exists():
            for alpha_act in alpha:
                self.calc_theta_single(layer + ".weight", idxs, alpha_act)

                if not self.model.load_state_dict(self.theta):
                    print("LOADING PARAMETERS INTO MODEL FAILED")
                    return None

                print("Training loss for alpha:", alpha_act, "...")
                t_loss = net.train(self.model, train_loader, optimizer, self.device, 0)
                print(t_loss)
                t_loss_list.append(t_loss)

            np.savetxt(training_loss_path, t_loss_list)
        return t_loss_list


    def vec_vloss(self):
        v_loss_list = []


init_state = Path(os.path.join(directory, "init_state.pt"))
final_state = Path(os.path.join(directory, "final_state.pt"))

model = net.Net().to("cuda")
model.load_state_dict(torch.load(final_state))

train_loader, test_loader = data_load.data_load()

interpolate = Interpolator(model, "cuda", final_state, init_state)
v = interpolate.single_vloss(test_loader, np.linspace(-1, 1, 20), "conv2", [4, 0, 10, 0])
print("VALIDATION LOSS:", v)

a = interpolate.single_acc(test_loader, np.linspace(-1, 1, 20), "conv2", [4, 0, 10, 0])
print("ACCURACY:", a)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # set optimizer
t = interpolate.single_train_loss(train_loader, optimizer, np.linspace(-1, 1, 20), "conv2", [4, 0, 10, 0])
print("TRAINING LOSS:", t)