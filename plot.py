import re
import h5py
import copy
import numpy as np
import matplotlib.pyplot as plt
from paths import *
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

color_loss = "red"
color_trained = "dimgrey"
color_acc = "blue"


def plot_line(x, y, xlabel, ylabel, annotate=False, color="blue"):
    logging.debug("[plot]: plotting line")
    fig, ax = plt.subplots(figsize=(6.4, 2))

    if xlabel:
        logging.debug("[plot]: xlabel = {}".format(xlabel))
        ax.set_xlabel(xlabel)
    if ylabel:
        logging.debug("[plot]: ylabel = {}".format(ylabel))
        ax.set_ylabel(ylabel)

    ax.plot(x, y, ".-", color=color, linewidth=1)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if annotate:
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
    plt.savefig(os.path.join(os.path.join(imgs), "{}.pdf".format(ylabel)), format="pdf")
    plt.close("all")


def plot_impact(x, loss, acc, loss_only=False, acc_only=False, annotate=True, xlabel=None):
    logging.debug("[plot]: Plotting preliminary experiments results")
    if not acc_only:
        if not loss.exists():
            logging.error("[plot]: No loss data found")
            return
        plot_line(x, np.loadtxt(loss), xlabel, "Validation loss", annotate, color_loss)

    if not loss_only:
        if not acc.exists():
            logging.error("[plot]: No accuracy data found")
            return
        plot_line(x, np.loadtxt(acc), xlabel, "Accuracy", annotate, color_acc)


def plot_box(x, loss_only=False, acc_only=False, show=False, xlabel=None):
    logging.info("[plot]: Plotting preliminary experiments results (test subset size)")

    if not acc_only:
        fig, ax = plt.subplots()

        if not epochs_loss.exists():
            logging.error("[plot]: No loss data found")
            return

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

        if not epochs_acc.exists():
            logging.error("[plot]: No accuracy data found")
            return

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


def plot_one_param(alpha, loss, acc, loss_img_path, acc_img_path, loss_only=False,
                   acc_only=False, show=False, trained=False):
    """
    Plots interpolation progress of parameter

    :param alpha: interpolation coefficient
    :param loss: validation loss data
    :param acc: accuracy data
    :param loss_img_path: path of validation loss image
    :param acc_img_path: path of accuracy image
    :param loss_only: plot only loss TODO: consider to delete
    :param acc_only: plot only accuracy TODO: considet to delete
    :param show: show the plots
    :param trained: plot actual state of the model
    """
    if not acc_only:
        fig, ax = plt.subplots()
        trained_loss = np.loadtxt(os.path.join(results, "actual_loss"))

        ax.plot(alpha, loss, ".-", color=color_loss, label="Validation loss with one parameter modified",
                linewidth=1)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Validation loss")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis='x', colors=color_loss)

        if trained:
            ax.spines["right"].set_visible(True)
            ax.spines["top"].set_visible(True)

            ax2 = ax.twiny()
            ax2.plot(range(len(trained_loss)), trained_loss, "-", color=color_trained, linewidth=1, linestyle="dashed")
            ax2.xaxis.tick_top()
            ax2.set_xlabel("Epochs")
            ax2.xaxis.set_label_position("top")
            ax2.tick_params(axis='x', colors=color_trained)

        if show:
            plt.show()
        plt.savefig("{}.pdf".format(loss_img_path), format="pdf")

    if not loss_only:
        fig, ax = plt.subplots()
        trained_accuracy = np.loadtxt(os.path.join(results, "actual_acc"))

        ax.plot(alpha, acc, ".-", color=color_acc, label="Accuracy with one parameter modified", linewidth=1)
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Accuracy")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis='x', colors=color_acc)

        if trained:
            ax.spines["right"].set_visible(True)
            ax.spines["top"].set_visible(True)
            ax2 = ax.twiny()
            ax2.plot(range(len(trained_accuracy)), trained_accuracy, "-", color=color_trained, linewidth=1,
                     linestyle="dashed")
            ax2.xaxis.tick_top()
            ax2.set_xlabel("Epochs")
            ax2.xaxis.set_label_position("top")
            ax2.tick_params(axis='x', colors=color_trained)

        if show:
            plt.show()
        plt.savefig("{}.pdf".format(acc_img_path), format="pdf")

        plt.close("all")


def map_distance(directory):
    """
    Maps calculated distances to values from interval <0, 1>

    :param directory: directory with distance files
    :return: dictionary of mapped distances assigned to according names
    """
    a_files = os.listdir(directory)
    distances = {}
    for file in a_files:
        # get all distances and associate them to right values in dictionary
        if re.search("distance", file):
            f = open(os.path.join(directory, file), 'r')
            distances[file] = float(f.readline())

    result = copy.deepcopy(distances)

    # get max and min values
    mx = distances[max(distances, key=lambda key: distances[key])]
    mn = distances[min(distances, key=lambda key: distances[key])]

    for key, value in distances.items():
        # map the distances
        result[key] = (value - mn)/(mx - mn) * (1 - 0) + 0
        if result[key] < 0.1:
            # low cap
            result[key] = 0.1

    return result


def plot_single(x, layer, opacity_dict, show=False):
    """
    Function plots all examined parameter of a layer in one plot

    :param x: data for x-axis (usually interpolation coefficient)
    :param layer: examined layer
    :param opacity_dict: dictionary with travelled distances of each parameter
    :param show: show the plots
    """
    files = os.listdir(single)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    for file in files:
        if re.search(layer, file) and re.search("loss", file) and not re.search("distance", file) and not re.search("q", file):
            k = file + "_distance"  # key for opacity dictionary
            lab = file.split("_")  # get label (parameter position)
            ax.plot(x, np.loadtxt(os.path.join(single, file)), label=lab[-1], alpha=opacity_dict[k], color="blueviolet")

    ax.set_ylabel("Validation loss")
    ax.set_xlabel(r"$\alpha$")

    plt.savefig("{}.pdf".format(os.path.join(single_img, layer)), format="pdf")

    if show:
        plt.show()

    plt.close("all")


def plot_vec_in_one(x, metric, opacity_dict, show=False):
    """
    Function plots selected metric of all layers and actual performance of the model in one plot

    :param x: data for x-axis (interpolation coefficient)
    :param metric: metric to be observed ("loss" or "acc)
    :param opacity_dict: dictionary with travelled distance of parameters of a layer
    :param show: show plot
    """
    files = os.listdir(vec)

    fig = plt.figure()
    ax = fig.add_subplot()

    label = "Validation loss" if metric == "loss" else "Accuracy"

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    for file in files:
        if re.search(metric, file) and not re.search("distance", file):
            k = file + "_distance"
            lab = file.split('_')
            ax.plot(x, np.loadtxt(os.path.join(vec, file)), label=lab[-1], alpha=opacity_dict[k])
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(label)

    actual = actual_loss_path if metric == "loss" else actual_acc_path
    actual = np.loadtxt(actual)

    ax2 = ax.twiny()
    ax2.plot(range(len(actual)), actual, '-', color=color_trained, label="actual", linewidth=1, linestyle="dashed")
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.set_xticks([])
    ax2.set_xticks([], minor=True)

    fig.legend(loc="lower center", ncol=6, mode="expand")
    fig.subplots_adjust(bottom=0.17)

    plt.savefig("{}_{}.pdf".format(vec_img, "all"), format="pdf")

    if show:
        plt.show()
    plt.close("all")


def plot_vec_all_la(x, distance_dict, show=False):
    """
    Function plots all performance of the model with modified layers in one figure

    :param x: data for x-axis (interpolation coefficient)
    :param show: show plot
    """
    files = os.listdir(vec)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8, 3))

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Validation loss")
    ax.set_xlabel(r"$\alpha$")

    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.set_ylabel("Accuracy")
    ax2.set_xlabel(r"$\alpha$")

    for file in files:
        if not re.search("distance", file) and not re.search("q", file):
            lab = file.split('_')
            k = file + "_distance"
            try:
                if re.search("loss", file):
                    ax.plot(x, np.loadtxt(os.path.join(vec, file)), label=lab[-1], lw=1, alpha=distance_dict[k])
                if re.search("acc", file):
                    ax2.plot(x, np.loadtxt(os.path.join(vec, file)), lw=1, alpha=distance_dict[k])
            except KeyError:
                logger.warning(f"Missing key {k} in opacity dict, will not plot line for {file}")
                continue

    ax.plot(x, np.loadtxt(loss_path), label="all", color=color_trained, linewidth=1)
    ax2.plot(x, np.loadtxt(acc_path), color=color_trained, linewidth=1)

    fig.legend()
    fig.subplots_adjust(bottom=0.17)

    plt.savefig("{}_{}.pdf".format(vec_img, "all_la"), format="pdf")

    if show:
        plt.show()
    plt.close("all")


def plot_lin_quad_real(show=False):
    alpha = np.linspace(0, 1, 40)
    epochs = np.arange(0, 14)

    lin = np.loadtxt(loss_path)
    quadr = np.loadtxt(q_loss_path)
    real = np.loadtxt(actual_loss_path)

    fig, ax1 = plt.subplots()

    ax2 = ax1.twiny()

    c1, = ax1.plot(alpha, lin, label="Linear interpolation", color="orange")
    c2, = ax1.plot(alpha, quadr, label="Quadratic interpolation", color="blue")
    c3, = ax2.plot(epochs, real, label="Real values", color="black")

    curves = [c1, c2, c3]
    ax2.legend(curves, [curve.get_label() for curve in curves])

    ax1.set_xlabel(r"$\alpha$")
    ax2.set_xlabel("Epochs")
    ax1.set_ylabel("Validation Loss")

    plt.savefig(os.path.join(vec_img, "lin_quadr_real.pdf"), format="pdf")
    if show:
        plt.show()

def plot_surface_contours(data, levels=50, show=False):
    plt.contour(data, levels)
    plt.title("Loss Function around trained model")

    if show:
        plt.show()

    plt.savefig(Path(os.path.join(random_dirs_img, "contour.pdf"), format="pdf"))


def contour_path(steps, loss_grid, coords, pcvariances):
    f = Path(os.path.join(pca_dirs_img, "loss_contour_path.pdf"))

    _, ax = plt.subplots()
    coords_x, coords_y = coords

    im = ax.contourf(coords_x, coords_y, loss_grid, levels=35, alpha=0.9)
    w1s = [step[0] for step in steps]
    w2s = [step[1] for step in steps]
    (pathline,) = ax.plot(w1s, w2s, color='r', lw=1)
    (point, ) = ax.plot(steps[0][0], steps[0][1], "ro")
    plt.colorbar(im)
    plt.show()

    plt.savefig(f, format="pdf")


def surface_3d(data, steps, show=False):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    X = np.array([[j for j in range(steps)] for i in range(steps)])
    Y = np.array([[i for _ in range(steps)] for i in range(steps)])

    ax.plot_surface(X, Y, data, rstride=1, cstride=1, cmap="viridis", edgecolor="none")

    ax.set_title("Surface of the loss function")
    if show:
        fig.show()

    plt.savefig(Path(os.path.join(random_dirs_img, f"surface_{steps}.pdf"), format="pdf"))

def surface3d_rand_dirs():
    # vmin = 0
    # vmax = 100

    # vlevel = 0.5
    surf = Path(os.path.join(random_dirs, "surf.h5"))
    surf_name = "loss"

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
        im = plt.imshow(Z, cmap="jet")
        plt.colorbar(im)
        plt.show()
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
