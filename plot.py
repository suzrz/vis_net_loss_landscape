import re
import h5py
import numpy as np
# import seaborn as sns
import matplotlib.pyplot as plt
from paths import *
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

color_loss = "red"
color_trained = "black"
color_acc = "blue"

def plot_line(x, y, xlabel, ylabel, annotate=False, color="blue"):
    fig, ax = plt.subplots(figsize=(6.4, 2))
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.plot(x, y, ".-", color=color, linewidth=1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if annotate:
        #ax.annotate("{:.4f}".format(y[-1]), xy=(x[-1], y[-1]),
        #            xytext=(y[-1] - 0.01 * x[-1], y[-1] + 0.01 * y[-1]))
        #ax.annotate("{:.4f}".format(y[-2]), xy=(x[-2], y[-2]),
        #            xytext=(x[-2] - 0.01 * x[-2], y[-2] + 0.01 * y[-2]))
        #ax.annotate("{:.4f}".format(y[-3]), xy=(x[-3], y[-3]),
        #            xytext=(x[-3] - 0.01 * x[-3], y[-3] + 0.01 * y[-3]))
        if y[-1] < 1:
            # loss
            k = 2
            if x[-1] < 1000:
                k = 0.25
        else:
            # accuracy
            k = 0.02
            if x[-1] < 1000:
                k = 0.002

        ax.annotate("{:.3f}".format(y[-1]), xy=(x[-1], y[-1]), xytext=(x[-1], y[-1] + y[-1]*k))
        ax.annotate("{:.3f}".format(y[-2]), xy=(x[-2], y[-2]), xytext=(x[-2], y[-2] + y[-2]*k))
        ax.annotate("{:.3f}".format(y[-3]), xy=(x[-3], y[-3]), xytext=(x[-3], y[-3] + y[-3]*k))

    fig.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(os.path.join(imgs), "{}.pdf".format(ylabel)), format="pdf")

def plot_impact(x, loss=None, acc=None, loss_only=False, acc_only=False, annotate=True, xlabel=None):
    if not acc_only:
        if not loss.all():
            print("No loss data found.")
            return
        plot_line(x, loss, xlabel, "Validation loss", annotate, color_loss)


    if not loss_only:
        if not acc.all():
            print("No accuracy data found.")
            return
        plot_line(x, acc, xlabel, "Accuracy", annotate, color_acc)


def plot_box(x, loss_only=False, acc_only=False, show=False, xlabel=None):
    if not acc_only:
        fig, ax = plt.subplots()

        loss = np.loadtxt(epochs_loss)

        ax.set_ylabel("Validation loss")
        ax.set_xlabel(xlabel)
        ax.set_xticklabels(x)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.boxplot(loss)

        if show:
            plt.show()
        plt.savefig(os.path.join(os.path.join(imgs, "subsets_imp"), "test_loss.pdf"), format="pdf")

    if not loss_only:
        fig, ax = plt.subplots()

        acc = np.loadtxt(epochs_acc)

        ax.set_ylabel("Accuracy")
        ax.set_xlabel(xlabel)
        ax.set_xticklabels(x)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.boxplot(acc)

        if show:
            plt.show()
        plt.savefig(os.path.join(os.path.join(imgs, "subsets_imp"), "test_acc.pdf"), format="pdf")



def plot_one_param(alpha, loss, acc, loss_img_path, acc_img_path, loss_only=False, acc_only=False, show=False, trained=False):
    if not acc_only:
        fig = plt.figure()
        ax = fig.add_subplot(111, label="1")
        trained_loss = np.loadtxt(os.path.join(results, "actual_loss"))

        ax.plot(alpha, loss, "x-", color=color_loss, label="Validation loss with one parameter modified", linewidth=1, markersize=3)
        #ax.legend(loc="upper right", fontsize="small")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Validation loss")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis='x', colors=color_loss)
        #ax.tick_params(axis='y', color=color_loss)

        if trained:
            ax.spines["right"].set_visible(True)
            ax.spines["top"].set_visible(True)
            ax2 = fig.add_subplot(111, label="2", frame_on=False)
            ax2.plot(range(len(trained_loss)), trained_loss, "-", color=color_trained, linewidth=1, linestyle="dashed")
            ax2.xaxis.tick_top()
            #ax2.yaxis.tick_right()
            ax2.set_xlabel("Epochs")
            #ax2.set_ylabel("Validation loss")
            ax2.set_yticks([])
            ax2.set_yticks([], minor=True)
            ax2.xaxis.set_label_position("top")
            #ax2.yaxis.set_label_position("right")
            ax2.tick_params(axis='x', colors=color_trained)
            #ax2.tick_params(axis='y', color=color_trained)


        #if show:
        #    plt.show()
        plt.savefig("{}.pdf".format(loss_img_path), format="pdf")

    if not loss_only:
        fig = plt.figure()
        ax = fig.add_subplot(111, label="1")

        trained_accuracy = np.loadtxt(os.path.join(results, "actual_acc"))

        ax.plot(alpha, acc, ".-", color=color_acc, label="Accuracy with one parameter modified", linewidth=1)
        #ax.legend(loc="lower right", fontsize="small")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Accuracy")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis='x', colors=color_acc)
        #ax.tick_params(axis='y', color=color_acc)

        if trained:
            ax.spines["right"].set_visible(True)
            ax.spines["top"].set_visible(True)
            ax2 = fig.add_subplot(111, label="2", frame_on=False)
            ax2.plot(range(len(trained_accuracy)), trained_accuracy, "-", color=color_trained, label="Accuracy of trained neural network", linewidth=1, linestyle="dashed")
            ax2.xaxis.tick_top()
            #ax2.yaxis.tick_right()
            ax2.set_yticks([])
            ax2.set_yticks([], minor=True)
            ax2.set_xlabel("Epochs")
            #ax2.set_ylabel("Accuracy")
            ax2.xaxis.set_label_position("top")
            #ax2.yaxis.set_label_position("right")
            ax2.tick_params(axis='x', colors=color_trained)
            #ax2.tick_params(axis='y', color=color_trained)

        #if show:
        #    plt.show()
        plt.savefig("{}.pdf".format(acc_img_path), format="pdf")

def plot_single(x, layer, show=False):
    files = os.listdir(single)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    count = 0

    print("layer:", layer)

    for file in files:
        if re.search(layer, file) and re.search("loss", file):
            count = count + 1
            print(count)
            #print(file)
            lab = file.split("_")
            ax.plot(x, np.loadtxt(os.path.join(single, file)), label=lab[-1])
        if count >= 10:
            plt.legend()
            plt.savefig("{}.pdf".format(os.path.join(single_img, layer)), format="pdf")
            plt.show()
            return

    plt.savefig("{}.pdf".format(os.path.join(single_img, layer)), format="pdf")
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
    surf_name = "val_loss"

    with h5py.File(surf, 'r') as fd:
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
        surface = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=False, cmap=cm.jet)
        fig.colorbar(surface, shrink=0.5, aspect=5)
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

"""
x = np.linspace(-1.0, 2.0, 40)
plot_single(x, "conv1")
plot_single(x, "conv2")
plot_single(x, "fc1")
plot_single(x, "fc2")
plot_single(x, "fc3")
"""