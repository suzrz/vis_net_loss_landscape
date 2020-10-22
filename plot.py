import h5py
import pickle
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D





def line2D_single_parameter():
    with open("v_loss_list.txt", "rb") as fd:
        val_loss_list = pickle.load(fd)

    with open("t_loss_list.txt", "rb") as fd:
        train_loss_list = pickle.load(fd)

    with open("trained_net_loss.txt", "rb") as fd:
        trained = pickle.load(fd)

    alpha = np.linspace(-0.25, 1.5, 13)

    fig, axe = plt.subplots()

    axe.plot(alpha, val_loss_list, "x-", label="validation loss")
    axe.plot(alpha, train_loss_list, "o-", color="orange", label="train loss")  # not normalized! should be lower than validation loss but because it is measured on more samples, it looks worse
    axe.plot(alpha, trained, "-", color="green", label="loss of trained net")

    axe.spines['right'].set_visible(False)
    axe.spines['top'].set_visible(False)

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
