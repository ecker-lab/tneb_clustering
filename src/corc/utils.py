from os.path import exists, join
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import sklearn
import studenttmixture
import corc.mixture
import diptest
import random
import time

os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# import jax


def set_seed(random_state):
    os.environ["PYTHONHASHSEED"] = str(random_state)
    random.seed(random_state)
    np.random.seed(random_state)


def compute_projection(data, cluster1, cluster2, means, predictions):
    c = means[cluster1] - means[cluster2]
    unit_vector = c / np.linalg.norm(c)

    points1 = data[predictions == cluster1]
    points2 = data[predictions == cluster2]
    cluster1_proj = np.dot(points1, unit_vector)
    cluster2_proj = np.dot(points2, unit_vector)

    mean = (np.mean(cluster1_proj) + np.mean(cluster2_proj)) / 2

    cluster1_proj -= mean
    cluster2_proj -= mean

    return cluster1_proj, cluster2_proj


def mixture_center_locations(mixture_model):
    # extract cluster centers
    if isinstance(mixture_model, sklearn.mixture.GaussianMixture):
        locations = mixture_model.means_
    elif isinstance(mixture_model, studenttmixture.EMStudentMixture):
        locations = mixture_model.location
    elif isinstance(mixture_model, corc.mixture.MixtureModel):
        locations = mixture_model.centers
    else:
        raise ValueError("Unknown mixture model type")
    return locations


def best_possible_labels_from_overclustering(y_true, y_pred):
    confusion = confusion_matrix(y_true, y_pred)

    # Create a mapping of predicted clusters to the majority true label
    best_labels = np.zeros(confusion.shape[1], dtype=int)

    for predicted_label in range(confusion.shape[1]):
        if (
            confusion[:, predicted_label].sum() > 0
        ):  # Check if there are any samples for this predicted label
            best_labels[predicted_label] = np.argmax(confusion[:, predicted_label])
        else:
            best_labels[predicted_label] = -1  # Handle cases where there's no count

    # Map the new labels back to the original predicted labels
    y_best = np.array(
        [best_labels[label] if label < len(best_labels) else -1 for label in y_pred]
    )

    return y_best


def predict_by_joining_closest_clusters(
    centers,
    y_pred,
    num_classes,
    data,
    dip_stat=False,
    recompute_distances=False,
    debug=False,
):

    def find_root(mapping, class_index):
        if mapping[class_index] != class_index:
            mapping[class_index] = find_root(mapping, mapping[class_index])
        return mapping[class_index]

    def merge_classes(mapping, class_i, class_j):
        root_i = find_root(mapping, class_i)
        root_j = find_root(mapping, class_j)

        # Merge by attaching root_j to root_i
        if root_i != root_j:
            mapping[root_j] = root_i
        return mapping

    def update_centers_after_merge(centers, mapping, i, j):
        # Get all points that belong to either cluster
        mask_i = y_pred == i
        mask_j = y_pred == j
        combined_points = data[mask_i | mask_j]

        # Compute new center as center of mass
        new_center = np.mean(combined_points, axis=0)

        # Update the center for cluster i (which is the root)
        centers[i] = new_center
        return centers

    def update_distances_after_merge(
        distances, mapping, i, j, centers, data, y_pred, dip_stat
    ):
        root_i = find_root(mapping, i)

        # Set distances between merged clusters to infinity
        distances[i, j] = distances[j, i] = np.inf

        if dip_stat:
            # Recompute dip statistics for the merged cluster with all others
            for k in range(len(centers)):
                if (
                    k != i and k != j and find_root(mapping, k) == k
                ):  # Only update for active clusters
                    # Compute new projections with merged cluster
                    pr1, pr2 = compute_projection(data, root_i, k, centers, y_pred)
                    dip, _ = diptest.diptest(np.concatenate([pr1, pr2]))
                    distances[min(root_i, k), max(root_i, k)] = -dip
        else:
            # Recompute Euclidean distances from merged cluster to all others
            for k in range(len(centers)):
                if (
                    k != i and k != j and find_root(mapping, k) == k
                ):  # Only update for active clusters
                    dist = np.linalg.norm(centers[root_i] - centers[k])
                    distances[min(root_i, k), max(root_i, k)] = dist

        # Set all distances involving j to infinity since it's now merged
        distances[j, :] = distances[:, j] = np.inf

        return distances

    mapping = np.array(range(len(centers)))
    centers = centers.copy()  # Create a copy to modify
    distances = np.ones((len(centers), len(centers))) * np.inf
    if dip_stat:
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                pr1, pr2 = compute_projection(data, i, j, centers, y_pred)
                dip, _ = diptest.diptest(np.concatenate([pr1, pr2]))
                distances[i, j] = -dip
    else:
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                distances[i, j] = np.linalg.norm(centers[i] - centers[j])

    if debug:
        plt.figure(figsize=(8, 8))
        sns.heatmap(distances.T * (-1), annot=True, fmt=".2g")
        plt.show()
    num_classes_to_join = len(centers) - num_classes
    for _ in range(num_classes_to_join):
        i, j = np.unravel_index(np.argmin(distances), distances.shape)
        mapping = merge_classes(mapping, i, j)
        if recompute_distances:
            # Update centers and distances
            centers = update_centers_after_merge(centers, mapping, i, j)
            distances = update_distances_after_merge(
                distances, mapping, i, j, centers, data, y_pred, dip_stat
            )
            # print(f"joined {i} and {j} (both now in class {find_root(mapping,i)})")
            if debug:
                plt.figure(figsize=(8, 8))
                sns.heatmap(distances.T * (-1), annot=True, fmt=".2g")
                plt.show()

    final_mapping = np.zeros(len(centers))
    for i in range(len(centers)):
        final_mapping[i] = find_root(mapping, i)

    joined_predictions = final_mapping[y_pred]

    # Relabel such that the output is in range(num_classes)
    joined_predictions = np.searchsorted(
        np.unique(joined_predictions), joined_predictions
    )

    return np.array(joined_predictions, dtype=int)


def load_dataset(dataset_name, cache_path="../cache", return_params=False):

    dataset_name = dataset_name.replace(" ", "_")
    dataset_filename = os.path.join(cache_path, "datasets", f"{dataset_name}.pickle")
    # Check if the dataset file exists
    if os.path.exists(dataset_filename):
        with open(dataset_filename, "rb") as f:
            dataset = pickle.load(f)
    else:
        print(f"Dataset {dataset_name} not found. Creating it...", end="")
        starttime = time.time()
        dataset = create_dataset_pickle(
            dataset_name=dataset_name,
            dataset_filename=dataset_filename,
            cache_path=cache_path,
        )
        print(f" done. {time.time()-starttime:.2f}s")

    X, y = dataset["dataset"]
    dimension = X.shape[-1]
    if "X2D" in dataset.keys():
        transformed_points = dataset["X2D"]
    elif dimension > 2:
        transformed_points = corc.visualization.get_TSNE_embedding(X)
    else:
        transformed_points = X

    if return_params:
        return X, y, transformed_points, dataset["dataset_info"]
    else:
        return X, y, transformed_points


def load_tmms(dataset_name, cache_path="../cache"):
    return load_algorithms(dataset_name, algorithm="TMM-NEB", cache_path=cache_path)


def load_algorithms(dataset_name, algorithm="TMM-NEB", cache_path="../cache"):
    dataset_filename = os.path.join(cache_path, f"{dataset_name}_{algorithm}.pickle")
    # Check if the dataset file exists
    if os.path.exists(dataset_filename):
        with open(dataset_filename, "rb") as f:
            tmms = pickle.load(f)
        return tmms
    else:
        return None


def create_dataset_pickle(dataset_name, dataset_filename=None, cache_path="../cache"):
    import corc.our_datasets
    import corc.visualization

    # obtain dataset
    X, y = corc.our_datasets.our_datasets(
        dataset_folder=cache_path + "/../datasets"
    ).get_single_dataset(dataset_name)
    tsne = corc.visualization.get_TSNE_embedding(X)

    dataset = {
        "dataset": (X, y),
        "X2D": tsne,
        "dataset_name": dataset_name,
    }

    # Save the dataset to a pickle file
    if dataset_filename is None:
        dataset_filename = f"{cache_path}/{dataset_name}.pickle"
    with open(dataset_filename, "wb") as f:
        pickle.dump(dataset, f)
    return dataset


def create_folder(folder_path):
    # ask the user if they want to create the directory
    if not exists(folder_path):
        create_dir = input(
            f"Directory {folder_path} does not exist. Do you want to create it? (Y/n): "
        )
        if create_dir.lower() == "y" or create_dir == "":
            os.makedirs(folder_path)
        else:
            print("Exiting...")
            exit(-1)
