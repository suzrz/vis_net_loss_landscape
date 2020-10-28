import h5py
import pickle
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def plot_accuracy(samples):
    with open("accuracy_list.txt", "rb") as fd:
        acc = pickle.load(fd)
    with open("trained_accuracy.txt","rb") as fd:
        trained_acc = pickle.load(fd)

    alpha = np.linspace(-0.25, 1.5, samples)

    plt.plot(alpha, acc, "x-", color="purple", label="Accuracy with interpolated parameters")
    plt.plot(alpha, trained_acc, "-", color="orange", label="Accuracy with trained parameters")
    plt.legend()
    plt.xlabel("alpha")
    plt.ylabel("accuracy")
    plt.show()

def plot_2D_loss(samples):
    with open("v_loss_list.txt", "rb") as fd:
        val_loss_list = pickle.load(fd)
    with open("trained_net_loss.txt", "rb") as fd:
        trained = pickle.load(fd)

    alpha = np.linspace(-0.25, 1.5, samples)

    plt.plot(alpha, val_loss_list, "x-", color="blue", label="Loss with one param modified")
    plt.plot(alpha, trained, "-", color="orange", label="Trained net loss")
    plt.legend()
    plt.xlabel("alpha")
    plt.ylabel("loss")
    plt.show()


def line2D_single_parameter():
    with open("v_loss_list.txt", "rb") as fd:
        val_loss_list = pickle.load(fd)

    with open("t_loss_list.txt", "rb") as fd:
        train_loss_list = pickle.load(fd)

    with open("accuracy_list.txt", "rb") as fd:
        accuracy = pickle.load(fd)

    with open("trained_net_loss.txt", "rb") as fd:
        trained = pickle.load(fd)

    with open("trained_accuracy.txt","rb") as fd:
        trained_acc = pickle.load(fd)

    alpha = np.linspace(-0.25, 1.5, 13)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(alpha, val_loss_list, "x-", label="validation loss")
    ax1.plot(alpha, train_loss_list, "o-", color="orange", label="train loss")  # not normalized! should be lower than validation loss but because it is measured on more samples, it looks worse
    ax1.plot(alpha, trained, "-", color="green", label="loss of trained net")
    ax2.plot(alpha, accuracy, "*-", color="purple", label="accuracy")
    ax2.plot(alpha, trained_acc, "-", color="orange", label="trained accuracy")

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    plt.legend()
    plt.xlabel("alpha")
    plt.ylabel("loss")

    plt.tight_layout()

    plt.show()

def surface3D_rand_dirs():
    # vmin = 0
    # vmax = 100

    # vlevel = 0.5

    filename = "3D_surf.h5"
    result_base = "./results/res_3D_surf"
    surf_name = "val_loss"

    with h5py.File(filename, 'r') as fd:
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
