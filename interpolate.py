import net
import data_load
import torch
import numpy as np
from collections import OrderedDict


"""INTERPOLATION"""
alpha = np.linspace(-0.25, 1.5, 13)  # set interpolation coefficient
train_loss_list = []  # prepare clean list for train losses
val_loss_list = []  # prepare clean list for validation losses
theta = OrderedDict()  # prepare clean parameter dict

"""
for alpha_act in alpha:  # interpolate
    for param_name0, param_name1 in zip(net.theta_i, net.theta_f):
        net.theta_0[param_name0] = torch.mul(net.theta_i[param_name0],
                                         (1.0 - alpha_act))
        net.theta_1[param_name1] = torch.mul(net.theta_f[param_name1],
                                         alpha_act)
        theta[param_name0] = torch.add(net.theta_0[param_name0],
                                       net.theta_1[param_name1])

    if not net.model.load_state_dict(theta):
        print("Something went wrong.")  # loading parameters in model failed
    print("ALPHA: ", alpha_act)
    print("Getting train_loss")
    train_loss = net.train(net.model, data_load.train_loader, net.optimizer, net.device, 0)
    train_loss_list.append(train_loss)
    print("Getting val loss")
    val_loss = net.test(net.model, data_load.test_loader, net.device)  # get loss with new parameters
    val_loss_list.append(val_loss)  # save obtained loss into list
"""

model_i = net.Net().to(net.device)
model_i.load_state_dict(net.theta_i)

model_f = net.Net().to(net.device)
model_f.load_state_dict(net.theta_f)

model = net.Net().to(net.device)
model.load_state_dict(net.theta_f) # can change.. depends on what I want to change
"""
print(model.conv1.weight[0][0][0][0])
model.conv1.weight[0][0][0][0] = model.conv1.weight[0][0][0][0] * 2
print(model.conv1.weight[0][0][0][0])
"""
for alpha_act in alpha:

    model.conv1.weight[0][0][0][0] = (model_i.conv1.weight[0][0][0][0] * (1.0 - alpha_act)) + (model_f.conv1.weight[0][0][0][0] * alpha_act)
    print("ALPHA: ", alpha_act)
    #train_loss = net.train(model, data_load.train_loader, net.optimizer, net.device, 0)
    train_loss = net.train_vis(model, data_load.train_loader,net.optimizer, net.device)
    print("Train loss: ", train_loss)
    train_loss_list.append(train_loss)
    val_loss = net.test(model, data_load.test_loader, net.device)  # get loss with new parameters
    print("Val loss: ", val_loss)
    val_loss_list.append(val_loss)  # save obtained loss into list