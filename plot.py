import interpolate
import matplotlib.pyplot as plt


# plot
fig, axe = plt.subplots()
axe.plot(interpolate.alpha, interpolate.val_loss_list, "x-", label="validation loss")
axe.plot(interpolate.alpha, interpolate.train_loss_list, "o-", color="orange", label="train loss")  # not normalized! should be lower than validation loss but because it is measured on more samples, it looks worse
axe.spines['right'].set_visible(False)
axe.spines['top'].set_visible(False)
plt.legend()
plt.xlabel("alpha")
plt.ylabel("loss")
plt.show()