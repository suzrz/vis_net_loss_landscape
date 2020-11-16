import os
import h5py
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D


def plot_subset_hist():
    n_tr_samples = [10000, 20000, 30000, 40000, 50000, 60000]

    losses = np.load("results/subset_losses.txt")
    accs = np.load("results/subset_accs.txt")

    ind = np.arange(len(losses))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 6)
    loss = ax1.bar(ind-width/2, losses, width, color="orange", label="validation loss")
    acc = ax2.bar(ind + width/2, accs, width, color="purple", label="accuracy")

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


def plot_accuracy(alpha, directory):
    trained_accuracy_path = Path(os.path.join(directory, "trained_accuracy"))
    accuracy_path = Path(os.path.join(directory, "accuracy"))

    trained_acc = np.load(trained_accuracy_path)
    acc = np.load(accuracy_path)

    plt.plot(alpha, acc, "x-", color="purple", label="Accuracy with interpolated parameters")
    plt.plot(alpha, trained_acc, "-", color="orange", label="Accuracy with trained parameters")
    plt.legend()
    plt.xlabel("alpha")
    plt.ylabel("accuracy")
    plt.show()


def plot_2d_loss(alpha, directory):
    trained_loss_path = Path(os.path.join(directory, "trained_loss"))
    validation_loss_path = Path(os.path.join(directory, "val_loss"))

    val_loss_list = np.load(validation_loss_path)
    trained = np.load(trained_loss_path)

    plt.plot(alpha, val_loss_list, "x-", color="blue", label="Loss with one param modified")
    plt.plot(alpha, trained, "-", color="orange", label="Trained net loss")
    plt.legend()
    plt.xlabel("alpha")
    plt.ylabel("loss")
    plt.show()


def line2d_single_parameter(directory, alpha):
    trained_loss_path = Path(os.path.join(directory, "trained_loss"))
    trained_accuracy_path = Path(os.path.join(directory, "trained_accuracy"))
    validation_loss_path = Path(os.path.join(directory, "val_loss"))
    training_loss_path = Path(os.path.join(directory, "training_loss"))
    accuracy_path = Path(os.path.join(directory, "accuracy"))

    val_loss_list = np.load(validation_loss_path)
    train_loss_list = np.load(training_loss_path)
    accuracy = np.load(accuracy_path)
    trained = np.load(training_loss_path)
    trained_acc = np.load(trained_accuracy_path)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(alpha, val_loss_list, "x-", label="validation loss")
    ax1.plot(alpha, train_loss_list, "o-", color="orange", label="train loss")
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


def surface3d_rand_dirs(directory):
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
