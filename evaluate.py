import net
import torch
import numpy as np
import matplotlib.pyplot as plt


net.model.load_state_dict(torch.load("init_state.pt"))
theta_i = net.model.state_dict()  # old theta_i does not change
theta_0 = net.model.state_dict()  # new theta_i (theta_0) does change

net.model.load_state_dict(torch.load("final_state.pt"))
theta_f = net.model.state_dict()  # old theta_f does not change
theta_1 = net.model.state_dict()  # new theta_f (theta_1) does change

"""INTERPOLATION"""
alpha = np.linspace(0, 1, 10)
loss_list = []
theta = dict()

for alpha_act in alpha:
    print(alpha_act)
    for param_tensor0, param_tensor1 in zip(theta_i, theta_f):
        # print(theta_0[param_tensor0]) #<class 'str'>
        theta_0[param_tensor0] = torch.mul(theta_i[param_tensor0],
                                           (1 - alpha_act))
        theta_1[param_tensor1] = torch.mul(theta_f[param_tensor1],
                                           alpha_act)
        theta[param_tensor0] = torch.add(theta_0[param_tensor0],
                                         theta_1[param_tensor1])

    loss = 0.
    net.model.load_state_dict(theta)  # load parameters in model
    # theta_0 = model.state_dict()
    loss = net.test(net.model, net.test_loader, net.device)
    loss_list.append(loss)

print(loss_list)

plt.plot(alpha, loss_list, "x-")
plt.xlabel("alpha")
plt.ylabel("loss")
plt.show()

