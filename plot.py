import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("v_loss_list.txt", "rb") as fd:
    val_loss_list = pickle.load(fd)

with open("t_loss_list.txt", "rb") as fd:
    train_loss_list = pickle.load(fd)

alpha = np.linspace(-0.25, 1.5, 13)

# plot
fig, axe = plt.subplots()
axe.plot(alpha, val_loss_list, "x-", label="validation loss")
axe.plot(alpha, train_loss_list, "o-", color="orange", label="train loss")  # not normalized! should be lower than validation loss but because it is measured on more samples, it looks worse
axe.spines['right'].set_visible(False)
axe.spines['top'].set_visible(False)
plt.legend()
plt.xlabel("alpha")
plt.ylabel("loss")
plt.tight_layout()
plt.show()
