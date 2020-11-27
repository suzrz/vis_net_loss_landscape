import h5py
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from paths import *
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

color_loss = "teal"
color_trained = "orange"
color_acc = "royalblue"


def plot_one_param(alpha, loss_only=False, acc_only=False):
    if not acc_only:
        fig, ax = plt.subplots()
        losses = np.loadtxt(svloss_path)
        trained_loss = np.loadtxt(sf_loss_path)

        ax.plot(alpha, losses, "x-", color=color_loss, label="Validation loss with one parameter modified")
        ax.plot(alpha, trained_loss, "-", color=color_trained, label="Validation loss of trained neural network")
        ax.legend()
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Validation loss")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        plt.show()

    if not loss_only:
        fig, ax = plt.subplots()
        accs = np.loadtxt(sacc_path)
        trained_accuracy = np.loadtxt(sf_acc_path)

        ax.plot(alpha, accs, "x-", color=color_acc, label="Accuracy with one parameter modified")
        ax.plot(alpha, trained_accuracy, "-", color=color_trained, label="Accuracy of trained neural network")
        ax.legend()
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Validation loss")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        plt.show()


def plot_impact_of_subset_size(subsets, losses, accs):
    fig, (ax1, ax2) = plt.subplots(2)
    ax1.set_xlabel("Size of test subset")
    ax1.set_ylabel("Validation loss")
    ax1.plot(subsets, losses, color=color_loss)

    ax2.set_xlabel("Size of subset")
    ax2.set_ylabel("Accuracy")
    ax2.plot(subsets, accs, color=color_acc)

    fig.tight_layout()
    plt.show()


""" Template may be needed later
def plot_subset_hist(n_tr_samples):
    losses = np.loadtxt(train_subs_loss)
    accs = np.loadtxt(train_subs_acc)

    ind = np.arange(len(losses))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 6)
    loss = ax1.bar(ind-width/2, losses, width, color=color_loss, label="validation loss")
    acc = ax2.bar(ind + width/2, accs, width, color=color_acc, label="accuracy")

    ax1.set_ylabel("validation loss")
    ax1.set_xlabel("number of samples")
    ax1.set_xticks(ind)
    ax1.set_xticklabels(n_tr_samples)
    ax1.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               borderaxespad=0, ncol=3)

    ax2.set_ylabel("accuracy")
    ax2.set_xlabel("number of samples")
    ax2.set_xticks(ind)
    ax2.set_xticklabels(n_tr_samples)
    ax2.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
               borderaxespad=0, ncol=3)

    for rect in loss:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() * 0.5, 1.01 * height, '{:.3f}'.format(height),
                 ha="center", va="bottom")

    for rect in acc:
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width() * 0.5, 1.01 * height, '{:.2f}'.format(height),
                 ha="center", va="bottom")

    plt.show()
"""


def surface3d_rand_dirs():
    # vmin = 0
    # vmax = 100

    # vlevel = 0.5
    filename = "surf_3d.h5"
    file = os.path.join(directory, filename)
    surf_name = "val_loss"

    with h5py.File(file, 'r') as fd:
        x = np.array(fd["xcoordinates"][:])
        y = np.array(fd["ycoordinates"][:])

        X, Y = np.meshgrid(x, y)
        Z = np.array(fd[surf_name][:])

        """3D"""
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x, y)")
        surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False, cmap=cm.jet)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

        """COUNTOURS"""
        """
        fig = plt.figure()
        CS = plt.contour(X, Y, Z, cmap="summer",
                         levels=np.arange(vmin, vmax, vlevel))
        plt.clabel(CS, inline=1, fontsize=8)
        #plt.savefig(result_base + '_' + surf_name + "_2D_contour" + ".pdf",
        #            dpi=300, bbox_inches="tight", format="pdf")
        plt.show()
        """

        """HEAT MAP"""
        """
        fig = plt.figure()
        sns_plot = sns.heatmap(Z, cmap="viridis", cbar=True, vmin=vmin,
                               vmax=vmax, xticklabels=False, yticklabels=False)
        sns_plot.invert_yaxis()
        #sns_plot.get_figure().savefig(result_base + '_' + surf_name + "_2D_heat.pdf",
        #                              dpi=300, bbox_inches="tight", format="pdf")
        plt.show()
        """
        """SAVE 3D"""
        """
        plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
        #fig.savefig(result_base + '_' + surf_name + "_3D_surface.pdf",
        #            dpi=300, bbox_inches="tight", format="pdf")
        """
