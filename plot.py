import interpolate
import matplotlib.pyplot as plt


# plot
fig, axe = plt.subplots(nrows=1, ncols=2)
axe[0].plot(interpolate.alpha, interpolate.val_loss_list, "x-", label="mod val loss")
axe[0].plot(interpolate.alpha, interpolate.val_loss_final, "o-", label="val loss")
axe[0].set_title("validation loss")
axe[1].plot(interpolate.alpha, interpolate.train_loss_list, "x-", label="mod train loss")
axe[1].plot(interpolate.alpha, interpolate.train_loss_final, "o-", label="train loss")
axe[1].set_title("train loss")
plt.legend()
plt.xlabel("alpha")
plt.ylabel("loss")
plt.tight_layout()
plt.show()
