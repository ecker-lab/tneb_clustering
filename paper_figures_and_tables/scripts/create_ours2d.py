import pickle
import matplotlib.pyplot as plt
import numpy as np
import corc
import corc.utils
import corc.graph_metrics
import corc.our_datasets
import os
import sklearn
import tqdm

# set parameters
cache_path = "cache"
figure_path = "figures"
figure_name = "ours2d.png"
datasets = corc.our_datasets.DATASETS2D
algorithm_name = "TMM-NEB"
ari_fontsize = 19


# load datasets and algorithm
fig, axs = plt.subplots(1, len(datasets), figsize=(len(datasets) * 3, 3))

for i, dataset_name in tqdm.tqdm(enumerate(datasets)):
    X, y, tsne = corc.utils.load_dataset(dataset_name, cache_path=cache_path)
    if tsne is None:
        tsne = X
    y = np.array(y, dtype=int)

    # load algorithm objects
    alg_filename = os.path.join(cache_path, f"{dataset_name}_{algorithm_name}.pickle")
    with open(alg_filename, "rb") as f:
        algorithms = pickle.load(f)

    # select the best model
    best_model = None
    best_ari = -1
    for algorithm in algorithms:
        y_pred = algorithm.predict_with_target(X, len(np.unique(y)))
        ari = sklearn.metrics.adjusted_rand_score(y, y_pred)
        if ari > best_ari:
            best_ari = ari
            best_model = algorithm

    # plot points
    y_pred = best_model.predict_with_target(X, len(np.unique(y)))
    colors = corc.visualization.get_color_scheme(int(max(max(y), max(y_pred)) + 1))
    y_pred_permuted = corc.visualization.reorder_colors(y_pred, y)
    axs[i].scatter(tsne[:, 0], tsne[:, 1], s=10, color=colors[y_pred_permuted])

    # plot graph
    best_model.data = X
    best_model.plot_graph(X2D=tsne, target_num_clusters=len(np.unique(y)), ax=axs[i])

    # add ARI scores to the plot
    axs[i].text(
        0.01,
        0.89,
        f"{best_ari:.2f}",
        transform=axs[i].transAxes,
        fontweight="bold",
        size=ari_fontsize,
        bbox=dict(facecolor="white", alpha=0.5, lw=0),
    )

    # remove border
    corc.visualization.remove_border(axs[i])
    axs[i].set_xticks(())
    axs[i].set_yticks(())

plt.subplots_adjust(wspace=0.02)

# save plot
plt.savefig(
    os.path.join(figure_path, figure_name),
    bbox_inches="tight",
    dpi=200,
)
plt.clf()
