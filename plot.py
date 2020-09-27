import interpolate
import matplotlib.pyplot as plt


# plot
fig, axe = plt.subplots(nrows=1, ncols=2)
axe[0].plot(interpolate.alpha, interpolate.val_loss_list, "x-", label="validation loss")
axe[1].plot(interpolate.alpha, interpolate.train_loss_list, "o-", color="orange", label="train loss")  # not normalized! should be lower than validation loss but because it is measured on more samples, it looks worse
plt.legend()
plt.xlabel("alpha")
plt.ylabel("loss")
plt.tight_layout()
plt.show()
