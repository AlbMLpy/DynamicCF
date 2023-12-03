import matplotlib as mpl
import matplotlib.pyplot as plt

PARAMS = {
 'figure.figsize': (10, 5),
 'figure.constrained_layout.use': True,
 'figure.facecolor': 'white',
 'font.size': 10,
 'axes.labelsize': 14,
 'legend.fontsize': 12,
 'xtick.labelsize': 12,
 'ytick.labelsize': 12,
 'axes.titlesize': 16,
 'figure.max_open_warning': 50,
}

def get_fig_ax():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    return fig, ax

def set_ax(ax, xlabel, ylabel, title) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid()

def add_plot_ax(ax, x, y, label, marker='*') -> None:
    ax.plot(x, y, marker=marker, label=label)

def save_fig(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi)

def plot_results(x, ys, labels, xlabel, ylabel, title, save_path=None, dpi=600, markers=None):
    if markers is None:
        markers = ['*'] * len(labels)
    with mpl.rc_context(PARAMS):
        fig, ax = get_fig_ax()
        for y, label, marker in zip(ys, labels, markers):
            add_plot_ax(ax, x, y, label, marker)
        set_ax(ax, xlabel, ylabel, title) 
        if save_path is not None:
            save_fig(fig, save_path, dpi=dpi)
