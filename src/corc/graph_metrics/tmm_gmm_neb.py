import functools
import itertools

import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import jax.example_libraries.optimizers as optimizers
import numpy as np
import scipy
import sklearn
import studenttmixture
import tqdm
import optax


# evaluates multivariate T dist at x, return logpdf - checks input
# from https://gist.github.com/yuneg11/5b493ab689f2c46fce50f19aec2df5b4
@jax.jit
def t_logpdf2(X, mean, cov, df):
    # TODO: Properly handle df == np.inf
    # if df == np.inf:
    #   return multivariate_normal.logpdf(x, loc, shape)
    # x, loc, shape, df = jnp._promote_dtypes_inexact(x, loc, shape, df)
    if not mean.shape:
        return jstats.t.logpdf(X, df, loc=mean, scale=jnp.sqrt(cov))
    else:
        n_dims = mean.shape[-1]
        if not np.shape(cov):
            y = X - mean
            # TODO: Implement this
            raise NotImplementedError(
                "multivariate_t.logpdf doesn't support scalar shape"
            )
        else:
            if cov.ndim < 2 or cov.shape[-2:] != (n_dims, n_dims):
                raise ValueError("multivariate_t.logpdf got incompatible shapes")

            # actual computation starts here
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


# @functools.partial(jax.vmap, in_axes=(0, None, None, None))
@jax.jit
def gmm_jax(X, means, covs, weights):
    p = []

    # pdf of the individual components
    for mean, cov, weight in zip(means, covs, weights):
        p.append(jstats.multivariate_normal.logpdf(X, mean, cov) + jnp.log(weight))

    # logsumexp (note that logpdf might return NaN because jax only operates on 32bit precision)
    p = jnp.stack(p, axis=-1)
    p = jax.scipy.special.logsumexp(jnp.nan_to_num(p, nan=-jnp.inf), axis=-1)
    return p


@jax.jit
def tmm_jax(X, means, covs, weights):
    p = []

    # pdfs of the individual components
    for mean, cov, weight in zip(means, covs, weights):
        p.append(t_logpdf(X=X, df=1.0, mean=mean, cov=cov) + jnp.log(weight))

    # adding them all up (note that logpdf might return NaN because jax only operates on 32bit precision)
    p = jnp.stack(p, axis=-1)
    p = jax.scipy.special.logsumexp(jnp.nan_to_num(p, nan=-jnp.inf), axis=-1)
    return p


@jax.jit
def predict_tmm_jax(X, means, covs, weights):
    # Initialize an array to hold log probabilities for each component
    logprobs_jax = jnp.zeros((len(weights), X.shape[0]))  # (n_components, n_samples)

    for i, (mean, cov, weight) in enumerate(zip(means, covs, weights)):
        # Calculate logpdf for each component and add the log weight
        logprobs_jax = logprobs_jax.at[i].set(
            t_logpdf(X=X, df=1.0, mean=mean, cov=cov) + jnp.log(weight)
        )

    # Find the index of the maximum log probability for each sample
    probs = jnp.exp(logprobs_jax)
    probs = jnp.nan_to_num(
        probs, nan=0.0
    )  # NaNs might occur in logpdf because jax uses 32bits
    predictions_jax = jnp.argmax(probs, axis=0)
    return predictions_jax, probs


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
def loss(X, means, covs, weights, dif_refrence, mixture_model="tmm"):
    # maximize the negative log likelihood
    if mixture_model == "tmm":
        nll = -tmm_jax(X, means, covs, weights).sum()
    elif mixture_model == "gmm":
        nll = -gmm_jax(X, means, covs, weights).sum()
    else:
        raise ValueError("mixture_model must be either 'tmm' or 'gmm'")

    # the loss is just the tmm/gmm value
    return nll


# perform gradient descent on the points
# @jax.jit
def step(X, means, covs, weights, dif_refrence, mixture_model="tmm", learning_rate=0.1):
    gradient = jax.grad(loss)(
        X, means, covs, weights, dif_refrence, mixture_model=mixture_model
    )
    X_new = X - learning_rate * gradient
    X = X.at[1:-1].set(X_new[1:-1])
    return X


# reinterpolate the points to have a uniform distance along the same path
# this function seems to be no longer in use and is superseded by equidistant_interpolate
@jax.jit
def reinterpolate(X):
    dif = jnp.diff(X, axis=0)
    dif = (dif**2).sum(axis=1) ** 0.5  # euclidean dist between successive points
    dif = jnp.cumsum(dif)
    dif = jnp.concatenate([jnp.array([0]), dif])
    dif = dif / dif[-1]

    ts = jnp.linspace(0, 1, X.shape[0])
    rms = jax.vmap(jnp.interp, in_axes=(None, None, -1))(ts, dif, X)
    return rms.T


@jax.jit
# reinterpolate the points to have a uniform distance along the same path
def equidistant_interpolate(path):
    num_points = 1024
    # Calculate arc lengths
    diffs = jnp.diff(path, axis=0)
    arc_lengths = jnp.cumsum(jnp.linalg.norm(diffs, axis=1))
    arc_lengths = jnp.pad(arc_lengths, (1, 0)) / arc_lengths[-1]

    # Interpolate
    t = jnp.linspace(0, 1, num_points)
    interpolated_path = jax.vmap(
        lambda t_val: jnp.apply_along_axis(
            lambda x: jnp.interp(t_val, arc_lengths, x), 0, path
        )
    )(t)

    return interpolated_path


# given two indices i and j find interpolation between the ith and jth means
def compute_interpolation(
    i,
    j,
    means,
    covs,
    weights,
    iterations=400,
    mixture_model="tmm",
):
    mean_1, mean_2 = means[i], means[j]
    num_points = 1024  # hardcoded since otherwise JIT does not work.

    # linear interpolation between m1 and m2
    temperatures = jnp.linspace(0, 1, num_points)[..., None]
    path = (1 - temperatures) * mean_1 + temperatures * mean_2

    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(path)
    for i in range(iterations):
        grads = jax.grad(loss)(
            path, means, covs, weights, None, mixture_model=mixture_model
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        path = optax.apply_updates(path, updates)

        path = equidistant_interpolate(path)

    # compute logprobs with the given mixture model
    if mixture_model == "tmm":
        probabilities = tmm_jax(path, means, covs, weights)
    elif mixture_model == "gmm":
        probabilities = gmm_jax(path, means, covs, weights)
    else:
        raise ValueError("mixture_model must be either 'tmm' or 'gmm'")
    return path, temperatures, probabilities


def compute_neb_paths(means, covs, weights, model_type, iterations=1000, knn=None):
    n_components = len(means)

    adjacency = np.zeros(
        (n_components, n_components)
    )  # will hold the scores, i.e. the minimum value encountered on a path
    paths = dict()  # the bent path positions from one center to another
    temps = (
        dict()
    )  # "procentual" values of the path (used for plotting logprob against distance)
    logprobs = dict()  # actual values at path positions

    if knn is not None:
        assert isinstance(knn, int) and knn > 0, "knn must be a positive integer"

        # compute k nearest neighbors for each node
        distances = jnp.linalg.norm(means[:, None, :] - means[None, :, :], axis=-1)
        indices = jnp.argsort(distances)[
            :, 1 : (knn + 1)
        ]  # skipping (i,i) which is always is the closest
        pairs = [
            (i, int(j))  # the order does not matter
            for i, row in enumerate(indices)
            for j in row
        ]
        total_pairs = len(pairs)

    else:
        # compute paths for all pairs
        pairs = itertools.combinations(range(n_components), r=2)
        total_pairs = n_components * (n_components - 1) // 2

    for i, j in tqdm.tqdm(
        pairs,
        total=total_pairs,
        desc=model_type,
    ):
        # compute nudged elastic band
        path_positions, temperatures, interpolation_probs = compute_interpolation(
            i,
            j,
            means=means,
            covs=covs,
            weights=weights,
            iterations=iterations,
            mixture_model=model_type,
        )
        # store results
        paths[(i, j)] = paths[(j, i)] = path_positions
        temps[(i, j)] = temps[(j, j)] = temperatures
        logprobs[(i, j)] = logprobs[(j, i)] = interpolation_probs

        # evaluate score of elastic band (minimum value along the path)
        adjacency[i, j] = adjacency[j, i] = jnp.min(interpolation_probs)

    raw_adjacency = adjacency.copy()
    adjacency = compute_mst_distances(raw_adjacency)

    return adjacency, raw_adjacency, paths, temps, logprobs


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


def evaluate_equidistance(paths):
    """
    looks at distances between points in paths.
    First computes the longest path and the corresponding granularity (path_length/num_points).
    Then compares all path-segments and computes the worst factor (segment/granularity) in each path.
    returns the worst overall factor and a dictionary with factors for all paths where a segment is longer than the granularity.
    """
    worst_factor = 1

    # first compute the "general granularity" (to ignore inconsistencies below that)
    longest_equidistant_length = 0
    for path_index in paths:
        path = paths[path_index]
        distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
        total_length = np.sum(distances)
        equidistant_length = total_length / len(distances)
        if equidistant_length > longest_equidistant_length:
            longest_equidistant_length = equidistant_length

    # go through all paths and check whether their segments are equidistant
    all_factors = dict()
    for path_index in paths:
        path = paths[path_index]
        distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
        longest_segment = max(distances)
        total_length = np.sum(distances)
        equidistant_length = total_length / len(distances)
        if longest_segment > longest_equidistant_length:
            factor = longest_segment / longest_equidistant_length
            all_factors[path_index] = factor
            if factor > worst_factor:
                worst_factor = factor

    if worst_factor > 10:
        print(
            f"WARNING: the path is not equidistant! longest segment {equidistance_factor} times too long"
        )
        for edge in all_factors.keys():
            if all_factors[edge] > 10:
                print(f"{edge} has factor {all_factors[edge]}")
    return worst_factor, all_factors
