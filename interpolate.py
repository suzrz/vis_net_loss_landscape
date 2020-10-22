import os
import net
import pickle
import data_load
import torch
import numpy as np
import copy


"""INTERPOLATION"""
alpha = np.linspace(-0.25, 1.5, 13)  # set interpolation coefficient
train_loss_list = []  # prepare clean list for train losses
val_loss_list = []  # prepare clean list for validation losses


if not os.path.isfile("trained_net_loss.txt"):
    net.model.load_state_dict(torch.load("final_state.pt"))
    trained_loss = net.test(net.model, data_load.test_loader, net.device)
    trained_loss = np.broadcast_to(trained_loss, alpha.shape)
    with open("trained_net_loss.txt", "wb") as fd:
        pickle.dump(trained_loss, fd)


if not os.path.isfile("v_loss_list.txt") or not os.path.isfile("t_loss_list.txt"):
    theta = copy.deepcopy(net.theta_f)
    for alpha_act in alpha:  # interpolate
        theta["conv2.weight"][4][0][0][0] = copy.copy(torch.add(net.theta_i["conv2.weight"][4][0][0][0] * (1.0 - alpha_act),
                                                                net.theta_f["conv2.weight"][4][0][0][0] * alpha_act))
        if not net.model.load_state_dict(theta):
            print("Something went wrong.")  # loading parameters in model failed

        print("ALPHA: ", alpha_act)
        print("Getting train loss")
        train_loss = net.train(net.model, data_load.train_loader, net.optimizer, net.device, 0)
        train_loss_list.append(train_loss)

        print("Getting val loss")
        val_loss = net.test(net.model, data_load.test_loader, net.device)  # get loss with new parameters
        val_loss_list.append(val_loss)  # save obtained loss into list

    with open("v_loss_list.txt", "wb") as fd:
        pickle.dump(val_loss_list, fd)

    with open("t_loss_list.txt", "wb") as fd:
        pickle.dump(train_loss_list, fd)

