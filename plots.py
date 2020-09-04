import matplotlib.cm as cm
import matplotlib.plt as plt

def plot2D(x, y, figure_index=0, label=None):
    plt.figure(num=figure_index)
    plt.plot(x, y, '-', label=label)
    if label is not None:
        plt.legend()

def plot_levels(x, y, z, figure_index=0):
    plt.figure(num=figure_index)
    CS = plt.contour(x, y, z)
    plt.clabel(CS, inline=1, fontsize=10)

def show_plots():
    plt.show()
