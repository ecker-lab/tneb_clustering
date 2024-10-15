from os.path import exists, join
import os
import pickle
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import numpy as np


def cond_mkdir(path):
    """create a folder if there is none.

    Parameters
    ----------
    path : str
        path or foldername
    """
    if not exists(path):
        os.makedirs(path)


def save(data, filename, outdir):
    """save data using pickle in folder under given name

    Parameters
    ----------
    data : ndarray
        data that should be saved
    filename : str
        name under which data is saved
    outdir : str
        folder name where to save data
    """
    with open(join(outdir, f"{filename}.pkl"), "wb") as f:
        pickle.dump(data, f)
    with open(join(outdir, f"{filename}.txt"), "w") as f:
        f.write(f"{data}")


def generate_overview_lineplot(log_dir):
    with open(join(log_dir, "config.yaml")) as f:
        opt = yaml.safe_load(f)

    df = pd.DataFrame(columns=["std", "value", "metric"])

    for metric in opt["metric"]["type"]:
        df_metric = pd.read_pickle(join(log_dir, f"{metric}.pkl"))
        df_metric["metric"] = [metric] * len(opt["metric"]["stds"])
        df_metric = df_metric.rename(columns={metric: "value"})
        df = pd.concat([df, df_metric])

    fig = plt.figure()
    sns.lineplot(data=df[df["metric"] != "CH"], x="std", y="value", hue="metric")
    filename = "overview_lineplot"
    fig.savefig(join(log_dir, f"{filename}.png"))
    fig.clear(True)

    sns.lineplot(data=df[df["metric"] == "CH"], x="std", y="value", hue="metric")
    filename = "overview_lineplot_CH"
    fig.savefig(join(log_dir, f"{filename}.png"))
    fig.clear(True)


def generate_overview_visplot(log_dir):

    from mpl_toolkits.axes_grid1 import ImageGrid

    with open(join(log_dir, "config.yaml")) as f:
        opt = yaml.safe_load(f)

    png_dir = join(log_dir, "png")
    n_stds = len(opt["metric"]["stds"])

    plot_types = sorted(list(set([t.split("_")[0] for t in os.listdir(png_dir)])))
    n_plots = len(plot_types)

    imgs, titles = [], []

    for std in opt["metric"]["stds"]:
        for t in plot_types:
            imgs.append(plt.imread(join(png_dir, f"{t}_{std:.2f}.png")))
            titles.append(f"plot {t} with std={std}")

    axes_pad = 0.1

    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(n_stds, n_plots),  # creates 2x2 grid of axes
        axes_pad=axes_pad,  # pad between axes in inch.
    )

    for ax, im, title in zip(grid, imgs, titles):
        ax.set_title(title, fontdict={"fontsize": 7}, pad=-0.5)
        ax.imshow(im)
        ax.axis("off")
    plt.axis("off")
    plt.tight_layout()

    filename = "overview_visplot"
    fig.savefig(join(log_dir, f"{filename}.pdf"))
    fig.savefig(join(log_dir, f"{filename}.png"))
    fig.clear(True)


def reorder_colors(y_pred, y_true):
    cm = confusion_matrix(y_true, y_pred)
    _, col_ind = linear_sum_assignment(
        -cm
    )  # col_ind returns how to reorder the columns (colors of y_pred)
    y_pred_permuted = np.argsort(col_ind)[
        y_pred
    ]  # we need the inverse mapping which we get through argsort
    return y_pred_permuted
