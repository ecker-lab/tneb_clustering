import functools
import itertools

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import numpy as np
import scipy
import tqdm
import optax
from typing import Tuple


# evaluates multivariate T dist at x, return logpdf - this version does not check the input
# from https://gist.github.com/yuneg11/5b493ab689f2c46fce50f19aec2df5b4
@jax.jit
def t_logpdf(X, mean, cov, df):
    # we trust that everything comes in the right shape
    n_dims = mean.shape[-1]
    u = 1 / 2 * (df + n_dims)
    L = lax.linalg.cholesky(cov)
    y = lax.linalg.triangular_solve(L, X - mean, lower=True, transpose_a=True)
    return (
        -u * jnp.log(1 + 1 / df * jnp.einsum("...i,...i->...", y, y))
        - n_dims / 2 * jnp.log(df * np.pi)
        + jax.scipy.special.gammaln(u)
        - jax.scipy.special.gammaln(1 / 2 * df)
        - jnp.log(L.diagonal(axis1=-1, axis2=-2)).sum(-1)
    )


t_logpdf_batched = jax.vmap(
    jax.vmap(t_logpdf, in_axes=(None, 0, 0, 0)), in_axes=(0, None, None, None)
)


@jax.jit
def tmm_jax_batched(paths, means, covs, weights, df=1.0):
    """
    Batched computation of log probabilities for multiple NEB paths using a t-mixture model.

    Args:
        paths: Batched paths. Shape: (batch_size, num_points, dim)
        means: TMM means. Shape: (n_components, dim)
        covs: TMM covariances. Shape: (n_components, dim, dim)
        weights: TMM weights. Shape: (n_components,)

    Returns:
        Log probabilities. Shape: (batch_size, num_points)
    """
    df_expanded = jnp.zeros(weights.shape[-1]) + df  # ensure dimensions of df
    # Compute logpdf for each path and each component
    logpdfs = t_logpdf_batched(
        paths, means, covs, df_expanded
    )  # Shape: (batch_size, num_points, n_components)

    # Add log weights to all paths (batch dim) and all points (last dim)
    logpdfs += jnp.log(weights)[None, :, None]

    # Perform logsumexp over the mixture components
    log_probs = jax.scipy.special.logsumexp(
        jnp.nan_to_num(logpdfs, nan=-jnp.inf), axis=1
    )  # Shape: (batch_size, num_points)

    return log_probs


@jax.jit
def tmm_jax(X, means, covs, weights, df=1.0):
    """
    Computes log probabilities for a single NEB path using a t-mixture model.
    """
    data_X = jnp.expand_dims(jnp.array(X), 0)
    return tmm_jax_batched(data_X, means, covs, weights, df=df)[0]


@jax.jit
def predict_tmm_batched(paths, means, covs, weights, df=1.0):
    # this functions assumes a batch dimension to be present.
    # paths: (batch_size, num_points, dim)
    df_expanded = jnp.zeros(weights.shape[-1]) + df  # ensure dimensions of df
    logprobs_raw = t_logpdf_batched(paths, means, covs, df_expanded)
    logprobs = logprobs_raw + jnp.log(weights)[None, :, None]
    probs = jnp.exp(logprobs)
    probs = jnp.nan_to_num(probs, nan=0.0)
    predictions = jnp.argmax(probs, axis=1)
    return predictions, probs


@jax.jit
def predict_tmm_jax(X, means, covs, weights, df=1.0):
    # no batch dimension present
    batched_X = X[None, ...]  # add batch dimension
    predictions, probs = predict_tmm_batched(batched_X, means, covs, weights, df)
    return predictions[0], probs[0]  # remove batch dimension


@jax.jit
def gmm_jax(X, means, covs, weights):

    logpdf_function = jax.vmap(jstats.multivariate_normal.logpdf, in_axes=(None, 0, 0))

    raw_logprobs = logpdf_function(X, means, covs)
    weighted_logprobs = raw_logprobs + jnp.log(weights)[:, None]
    log_probs = jax.scipy.special.logsumexp(
        jnp.nan_to_num(weighted_logprobs, nan=-jnp.inf), axis=0
    )

    return log_probs


gmm_jax_batched = jax.vmap(gmm_jax, in_axes=(0, None, None, None))


@jax.jit
def predict_gmm_jax(X, means, covs, weights):
    # Initialize an array to hold log probabilities for each component
    logprobs_jax = jnp.zeros((len(weights), X.shape[0]))  # (n_components, n_samples)

    for i, (mean, cov, weight) in enumerate(zip(means, covs, weights)):
        # Calculate logpdf for each component and add the log weight
        logprobs_jax = logprobs_jax.at[i].set(
            jstats.multivariate_normal.logpdf(X, mean, cov) + jnp.log(weight)
        )

    # Find the index of the maximum log probability for each sample
    probs = jnp.exp(logprobs_jax)
    probs = jnp.nan_to_num(probs, nan=0.0)
    predictions_jax = jnp.argmax(probs, axis=0)
    return predictions_jax, probs


# the loss of the interpolation
def loss(paths, means, covs, weights, gmm=False, df=1.0):
    # maximize the negative log likelihood
    if not gmm: # tmm (the default case)
        nll = -jnp.sum(tmm_jax_batched(paths, means, covs, weights, df=df))
    else:
        nll = -jnp.sum(gmm_jax_batched(paths, means, covs, weights))

    # the loss is just the tmm/gmm value
    return nll


# reinterpolate the points to have a uniform distance along the same path
@jax.jit
def reinterpolate(X):
    diffs = jnp.diff(X, axis=0)
    distances = (diffs**2).sum(
        axis=1
    ) ** 0.5  # euclidean dist between successive points
    path_lengths = jnp.cumsum(distances)
    path_lengths = jnp.concatenate([jnp.array([0]), path_lengths])  # add leading 0
    path_lengths = path_lengths / path_lengths[-1]  # normalize

    ts = jnp.linspace(0, 1, X.shape[0])
    interpolated_path = jax.vmap(jnp.interp, in_axes=(None, None, -1))(
        ts, path_lengths, X
    )  # the actual interpolation
    return interpolated_path.T


batch_interpolate = jax.vmap(reinterpolate, in_axes=(0))


# reinterpolate the points to have a uniform distance along the same path
# this function can specify the number of points of the output path
def equidistant_interpolate(path, target_num_points=None):
    num_points = len(path)
    if target_num_points is None:
        target_num_points = num_points

    # Calculate arc lengths
    diffs = jnp.diff(path, axis=0)
    arc_lengths = jnp.cumsum(jnp.linalg.norm(diffs, axis=1))
    arc_lengths = jnp.pad(arc_lengths, (1, 0)) / arc_lengths[-1]

    # Interpolate
    t = jnp.linspace(0, 1, target_num_points)
    interpolated_path = jax.vmap(
        lambda t_val: jnp.apply_along_axis(
            lambda x: jnp.interp(t_val, arc_lengths, x), 0, path
        )
    )(t)

    return interpolated_path


interpolate_paths_batched = jax.vmap(
    functools.partial(equidistant_interpolate, target_num_points=1024)
)


def compute_interpolation_batch(
    pairs: Tuple[int, int],
    means: jnp.ndarray,
    covs: jnp.ndarray,
    weights: jnp.ndarray,
    df: float = 1.0,
    iterations: int = 500,
    num_points: int = 1024,
    gmm: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vectorized computation of NEB paths for a batch of index pairs.

    Args:
        pairs: A tuple of two arrays, each containing indices for means.
        means: Array of means.
        covs: Array of covariances.
        weights: Array of mixture weights.
        iterations: Number of optimization iterations.
        num_points: Number of points in NEB path.
        gmm: Whether to use TMM (default) or GMM.

    Returns:
        paths: Batched array of interpolated paths.
        probabilities: Batched array of log probabilities along paths.
    """
    i_indices, j_indices = pairs  # Each of shape: (batch_size,)

    batch_size, dim = means.shape

    # Reshape for broadcasting
    temperatures = jnp.linspace(0, 1, num_points).reshape(1, num_points, 1)
    means_i = means[i_indices].reshape(-1, 1, dim)  # Shape: (batch_size, 1, dim)
    means_j = means[j_indices].reshape(-1, 1, dim)  # Shape: (batch_size, 1, dim)
    initial_paths = (
        1 - temperatures
    ) * means_i + temperatures * means_j  # Shape: (batch_size, num_points, dim)

    # Define the optimizer
    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(initial_paths)

    def optimization_step(paths, opt_state):
        grads = jax.grad(loss)(paths, means, covs, weights, df=df, gmm=False)
        updates, opt_state = optimizer.update(grads, opt_state)
        paths = optax.apply_updates(paths, updates)
        paths = batch_interpolate(paths)
        return paths, opt_state

    # Run the optimization loop
    paths, _ = jax.lax.fori_loop(
        0,
        iterations,
        lambda i, val: optimization_step(*val),
        (initial_paths, opt_state),
    )

    # Compute log probabilities
    interpolated_paths = interpolate_paths_batched(paths)
    if not gmm:
        logprobs = tmm_jax_batched(interpolated_paths, means, covs, weights)
    else:
        logprobs = gmm_jax_batched(interpolated_paths, means, covs, weights)
    distances = jnp.min(logprobs, axis=1)

    # return paths, distances
    return paths, distances


def compute_neb_paths_batch(
    means: jnp.ndarray,
    covs: jnp.ndarray,
    weights: jnp.ndarray,
    df: float = 1.0,
    iterations: int = 500,
    num_NEB_points: int = 100,
    knn: int = None,
    gmm: bool = False,
    batch_size: int = 1000,
):
    n_components = len(means)

    # Determine all pairs based on knn
    if knn is not None:
        distances = jnp.linalg.norm(means[:, None, :] - means[None, :, :], axis=-1)
        indices = jnp.argsort(distances, axis=-1)[:, 1 : (knn + 1)]
        pair_i, pair_j = jnp.meshgrid(
            jnp.arange(n_components), jnp.arange(knn), indexing="ij"
        )
        pairs = (pair_i.flatten(), indices.flatten())
    else:
        pairs = list(itertools.combinations(range(n_components), r=2))
        pair_i = jnp.array([i for i, _ in pairs])
        pair_j = jnp.array([j for _, j in pairs])
        pairs = (pair_i, pair_j)

    total_pairs = len(pairs[0])
    adjacency = jnp.zeros((n_components, n_components))
    paths = dict()

    # Batch processing
    description_tqdm = "g-NEB" if gmm else "t-NEB"
    for start in tqdm.tqdm(range(0, total_pairs, batch_size), desc=description_tqdm):
        end = start + batch_size
        batch_pairs = (pairs[0][start:end], pairs[1][start:end])

        # Compute interpolations in batch
        batch_paths, distances = compute_interpolation_batch(
            batch_pairs,
            means,
            covs,
            weights,
            df=df,
            iterations=iterations,
            num_points=num_NEB_points,
            gmm=gmm,
        )

        # Update adjacency and result dictionaries
        for idx, (i, j) in enumerate(zip(batch_pairs[0], batch_pairs[1])):
            i, j = int(i), int(j)  # Ensure indices are integers
            paths[(i, j)] = paths[(j, i)] = batch_paths[idx]
            adjacency = adjacency.at[i, j].set(distances[idx])
            adjacency = adjacency.at[j, i].set(distances[idx])

    raw_adjacency = adjacency.copy()
    adjacency = compute_mst_distances(raw_adjacency)

    return adjacency, raw_adjacency, paths


def compute_mst_distances(adjacency):
    """
    Takes a distance matrix from NEB, computes an MST, and outputs a corrected distance
    especially useful in cases where NEB did not fully converge for all paths. This is already used in compute_neb_paths
    """
    mst = -scipy.sparse.csgraph.minimum_spanning_tree(-adjacency)
    dist_matrix, predecessors = scipy.sparse.csgraph.shortest_path(
        mst, directed=False, unweighted=True, return_predecessors=True, method="BF"
    )  # note that we use unweighted because we are only interested in the pairwise paths
    # which are encoded in "predecessors". The dist_matrix would sum up the path instead of
    # returning the longest path segment.

    # Check whether the MST is "whole" (dist_matrix contains inf values for disconnected graphs)
    if not np.isfinite(dist_matrix).all():
        print(
            "Warning: MST contains multiple connected components. Please consider increasing knn."
        )

    # compute pairwise distances using shortest path information from the MST
    distances = -np.ones_like(adjacency) * float("inf")
    num_nodes = adjacency.shape[0]
    for start_node, target_node in itertools.combinations(range(num_nodes), 2):
        max_step_length = np.inf

        if np.isfinite(dist_matrix[start_node, target_node]):
            # Walk along path from start to target using "predecessors"
            current_node = target_node
            while current_node != start_node:
                next_node = predecessors[start_node, current_node]
                max_step_length = min(
                    max_step_length, adjacency[current_node, next_node]
                )
                current_node = next_node

        distances[start_node, target_node] = distances[target_node, start_node] = (
            max_step_length
        )

    # set self-loops
    for i in range(num_nodes):
        distances[i, i] = 0

    return distances
