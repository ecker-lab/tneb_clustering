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
def t_logpdf2(x, loc, shape, df):
    # TODO: Properly handle df == np.inf
    # if df == np.inf:
    #   return multivariate_normal.logpdf(x, loc, shape)
    # x, loc, shape, df = jnp._promote_dtypes_inexact(x, loc, shape, df)
    if not loc.shape:
        return jstats.t.logpdf(x, df, loc=loc, scale=jnp.sqrt(shape))
    else:
        n = loc.shape[-1]
        if not np.shape(shape):
            y = x - loc
            # TODO: Implement this
            raise NotImplementedError(
                "multivariate_t.logpdf doesn't support scalar shape"
            )
        else:
            if shape.ndim < 2 or shape.shape[-2:] != (n, n):
                raise ValueError("multivariate_t.logpdf got incompatible shapes")

            # actual computation starts here
            u = 1 / 2 * (df + n)
            L = lax.linalg.cholesky(shape)
            y = lax.linalg.triangular_solve(L, x - loc, lower=True, transpose_a=True)
            return (
                -u * jnp.log(1 + 1 / df * jnp.einsum("...i,...i->...", y, y))
                - n / 2 * jnp.log(df * np.pi)
                + jax.scipy.special.gammaln(u)
                - jax.scipy.special.gammaln(1 / 2 * df)
                - jnp.log(L.diagonal(axis1=-1, axis2=-2)).sum(-1)
            )


# evaluates multivariate T dist at x, return logpdf - this version does not check the input
# from https://gist.github.com/yuneg11/5b493ab689f2c46fce50f19aec2df5b4
@jax.jit
def t_logpdf(x, loc, shape, df):
    # we trust that everything comes in the right shape
    n = loc.shape[-1]
    u = 1 / 2 * (df + n)
    L = lax.linalg.cholesky(shape)
    y = lax.linalg.triangular_solve(L, x - loc, lower=True, transpose_a=True)
    return (
        -u * jnp.log(1 + 1 / df * jnp.einsum("...i,...i->...", y, y))
        - n / 2 * jnp.log(df * np.pi)
        + jax.scipy.special.gammaln(u)
        - jax.scipy.special.gammaln(1 / 2 * df)
        - jnp.log(L.diagonal(axis1=-1, axis2=-2)).sum(-1)
    )


@jax.jit
@functools.partial(jax.vmap, in_axes=(0, None, None, None))
def gmm_jax(x, means, covs, weights):
    p = []

    # pdf of the individual components
    for mean, cov, weight in zip(means, covs, weights):
        p.append(jstats.multivariate_normal.logpdf(x, mean, cov) + jnp.log(weight))

    # logsumexp
    p = jnp.stack(p, axis=-1)
    p = jax.scipy.special.logsumexp(p, axis=-1)
    return p


@jax.jit
def tmm_jax(x, means, scales, weights):
    p = []

    # pdfs of the individual components
    for mean, scale, weight in zip(means, scales, weights):
        p.append(t_logpdf(x=x, df=1.0, loc=mean, shape=scale) + jnp.log(weight))

    # adding them all up
    p = jnp.stack(p, axis=-1)
    p = jax.scipy.special.logsumexp(p, axis=-1)
    return p


# the loss of the interpolation
def loss(ms, means, covs, weights, dif_refrence, mixture_model="tmm"):
    # maximize the negative log likelihood
    if mixture_model == "tmm":
        nll = -tmm_jax(ms, means, covs, weights).sum()
    elif mixture_model == "gmm":
        nll = -gmm_jax(ms, means, covs, weights).sum()
    else:
        raise ValueError("mixture_model must be either 'tmm' or 'gmm'")

    # # calculate the difference between successive points
    # dif = jnp.diff(ms, axis=0)
    # dif = (dif**2).sum(axis=1)
    #
    # # penalize the difference between the points compared to the reference
    # # difference from the linear interpolation
    # dif = (dif - dif_refrence) ** 2
    # dif = dif.sum()

    # the loss is just the tmm/gmm value
    return nll * 1e-1  # + dif * 1e-2


# perform gradient descent on the points
# @jax.jit
def step(ms, means, covs, weights, dif_refrence, mixture_model="tmm"):
    g = jax.grad(loss)(
        ms, means, covs, weights, dif_refrence, mixture_model=mixture_model
    )
    ms_new = ms - 1e-1 * g
    ms = ms.at[1:-1].set(ms_new[1:-1])
    return ms


# reinterpolate the points to have a uniform distance along the same path
@jax.jit
def reinterpolate(ms):
    dif = jnp.diff(ms, axis=0)
    dif = (dif**2).sum(axis=1) ** 0.5  # euclidean dist between successive points
    dif = jnp.cumsum(dif)
    dif = jnp.concatenate([jnp.array([0]), dif])
    dif = dif / dif[-1]

    ts = jnp.linspace(0, 1, ms.shape[0])
    rms = jax.vmap(jnp.interp, in_axes=(None, None, -1))(ts, dif, ms)
    return rms.T


# given two indices i and j find interpolation between the ith and jth means
def compute_interpolation(
    i, j, means, covs, weights, iterations=4000, mixture_model="tmm"
):
    m1, m2 = means[i], means[j]

    # linear interpolation between m1 and m2
    ts = jnp.linspace(0, 1, 1024)[..., None]
    ms = (1 - ts) * m1 + ts * m2

    # euclidean distance
    dif_refrence = ((ms[0] - ms[1]) ** 2).sum()

    optimizer = optax.adam(0.01)
    opt_state = optimizer.init(ms)
    for i in range(iterations):
        grads = jax.grad(loss)(
            ms, means, covs, weights, dif_refrence, mixture_model=mixture_model
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        ms = optax.apply_updates(ms, updates)

        # the following line effectively resamples the line through linear interpolation. Without it, all of the 1000 points will be pushed away from the low-density regions instead of finding the best possible line between the two dense regions.
        ms = scipy.signal.savgol_filter(ms, window_length=100, polyorder=1, axis=0)

    # this is where the band becomes elastic and where the work is done.
    # for _ in range(iterations):
    #     ms, opt_state = step(
    #         ms,
    #         means,
    #         covs,
    #         weights,
    #         dif_refrence,
    #         mixture_model=mixture_model,
    #     )
    #     ms = reinterpolate(ms)

    # compute logprobs with the given mixture model
    if mixture_model == "tmm":
        ps = tmm_jax(ms, means, covs, weights)
    elif mixture_model == "gmm":
        ps = gmm_jax(ms, means, covs, weights)
    else:
        raise ValueError("mixture_model must be either 'tmm' or 'gmm'")
    return ms, ts, ps


def compute_neb_paths(mixture_model, iterations=1000):
    if isinstance(mixture_model, sklearn.mixture.GaussianMixture):
        locations = mixture_model.means_
        covs = mixture_model.covariances_
        weights = mixture_model.weights_
        model_type = "gmm"
    elif isinstance(mixture_model, studenttmixture.EMStudentMixture):
        locations = mixture_model.location
        covs = np.transpose(mixture_model.scale, axes=(2, 0, 1))
        weights = mixture_model.mix_weights
        model_type = "tmm"
    n_components = len(locations)

    adjacency = np.zeros(
        (n_components, n_components)
    )  # will hold the scores, i.e. the minimum value encountered on a path
    paths = dict()  # the bent path positions from one center to another
    temps = (
        dict()
    )  # "procentual" values of the path (used for plotting logprob against distance)
    logprobs = dict()  # actual values at path positions

    for i, j in tqdm.tqdm(
        itertools.combinations(range(n_components), r=2),
        total=n_components * (n_components - 1) // 2,
        desc=model_type,
    ):
        # compute nudged elastic band
        path_positions, temperatures, interpolation_probs = compute_interpolation(
            i,
            j,
            means=locations,
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
        adjacency[i, j] = adjacency[j, i] = min(interpolation_probs)

        raw_adjacency = adjacency.copy()
        adjacency = compute_mst_distances(raw_adjacency)

    return adjacency, raw_adjacency, paths, temps, logprobs


""" Takes a distance matrix from NEB, computes an MST, and outputs a corrected distance 
especially useful in cases where NEB did not fully converge for all paths. This is already used in compute_neb_paths
"""


def compute_mst_distances(adjacency):
    mst = -scipy.sparse.csgraph.minimum_spanning_tree(-adjacency)
    num_nodes = adjacency.shape[0]

    node_array, predecessors = scipy.sparse.csgraph.breadth_first_order(
        mst, 0, directed=False, return_predecessors=True
    )

    # initialize distance matrix, -'inf' is neutral
    distances = np.ones_like(adjacency) * -float("inf")

    # self-loops
    for i in range(num_nodes):
        distances[i][i] = float("inf")

    # insert values
    mst = mst.tocoo()
    for u, v, weight in zip(mst.row, mst.col, mst.data):
        distances[u][v] = distances[v][u] = weight

    marked = np.zeros(adjacency.shape[0])
    for node in node_array:
        for other_node in range(num_nodes):
            if marked[other_node]:
                # distance is either "direct" or using the parent information
                direct_dist = distances[node][
                    other_node
                ]  # not touched before, possibly -inf
                indirect_dist = min(
                    distances[node][predecessors[node]],
                    distances[predecessors[node]][other_node],
                )  # the latter is already computed and the first has a value
                # assign to both directions
                distances[node][other_node] = distances[other_node][node] = max(
                    direct_dist, indirect_dist
                )

        marked[node] = 1

    # reset self-loops
    for i in range(num_nodes):
        distances[i, i] = 0

    return distances


# def get_thresholds_and_cluster_numbers(adjacency):
#     thresholds, counts = np.unique(adjacency, return_counts=True)
#     thresholds = sorted(thresholds.tolist())
#
#     cluster_numbers = list()
#     clusterings = list()
#     for threshold in thresholds:
#         tmp_adj = np.array(adjacency >= threshold, dtype=int)
#         n_components, clusters = scipy.sparse.csgraph.connected_components(
#             tmp_adj, directed=False
#         )
#         cluster_numbers.append((threshold, n_components))
#         clusterings.append((n_components, threshold, clusters))
#
#     cluster_numbers = np.array(cluster_numbers)
#
#     return thresholds, cluster_numbers, clusterings
