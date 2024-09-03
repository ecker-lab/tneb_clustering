import jax
import jax.numpy as jnp
import jax.scipy.stats as jstats
import jax.lax as lax
import functools
import numpy as np

# @_wraps(osp_stats.multivariate_t.logpdf, update_doc=False, lax_description="""
# In the JAX version, the `allow_singular` argument is not implemented.
# """)
# from https://gist.github.com/yuneg11/5b493ab689f2c46fce50f19aec2df5b4
def t_logpdf(x, loc, shape, df, allow_singular=None):
  if allow_singular is not None:
    raise NotImplementedError("allow_singular argument of multivariate_t.logpdf")
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
      raise NotImplementedError("multivariate_t.logpdf doesn't support scalar shape")
    else:
      if shape.ndim < 2 or shape.shape[-2:] != (n, n):
        raise ValueError("multivariate_t.logpdf got incompatible shapes")
      u = 1/2 * (df + n)
      L = lax.linalg.cholesky(shape)
      y = lax.linalg.triangular_solve(L, x - loc, lower=True, transpose_a=True)
      return (-u * jnp.log(1 + 1/df * jnp.einsum('...i,...i->...', y, y))
              - n/2*jnp.log(df*np.pi) + jax.scipy.special.gammaln(u) - jax.scipy.special.gammaln(1/2 * df)
              - jnp.log(L.diagonal(axis1=-1, axis2=-2)).sum(-1))


@jax.jit
@functools.partial(jax.vmap, in_axes=(0, None, None, None))
def gmm_jax(x, means, covs, weights):
    p = []

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
def loss(ms, means, covs, weights, dif_refrence):
    # maximize the negative log likelihood
    nll = -tmm_jax(ms, means, covs, weights).sum()

    # calculate the difference between the points
    dif = jnp.diff(ms, axis=0)
    dif = (dif**2).sum(axis=1)

    # penalize the difference between the points compared to the reference
    # difference from the linear interpolation
    dif = (dif - dif_refrence) ** 2
    dif = dif.sum()

    return nll * 1e-1  # + dif * 1e-2


# perform gradient descent on the points
@jax.jit
def step(ms, means, covs, weights, dif_refrence):
    g = jax.grad(loss)(ms, means, covs, weights, dif_refrence)
    ms_new = ms - 1e-1 * g
    ms = ms.at[1:-1].set(ms_new[1:-1])
    return ms


# reinterpolate the points to have a uniform distance along the same path
@jax.jit
def reinterpolate(ms):
    dif = jnp.diff(ms, axis=0)
    dif = (dif**2).sum(axis=1) ** 0.5
    dif = jnp.cumsum(dif)
    dif = jnp.concatenate([jnp.array([0]), dif])
    dif = dif / dif[-1]

    ts = jnp.linspace(0, 1, ms.shape[0])
    rms = jax.vmap(jnp.interp, in_axes=(None, None, -1))(ts, dif, ms)
    return rms.T


# given two indicies i and j find interpolation between the ith and jth means
def compute_interpolation(i, j, means, covs, weights, iterations=4000):
    m1, m2 = means[i], means[j]

    # linear interpolation between m1 and m2
    ts = jnp.linspace(0, 1, 1024)[..., None]
    ms = (1 - ts) * m1 + ts * m2

    # euclidean distance
    dif_refrence = ((ms[0] - ms[1]) ** 2).sum()

    # this is where the band becomes elastic and where the work is done.
    for _ in range(iterations):
        ms = step(ms, means, covs, weights, dif_refrence)
        ms = reinterpolate(ms)

    # compute logprobs with the given mixture model
    ps = tmm_jax(ms, means, covs, weights)
    return ms, ts, ps


# score the interpolation for the final graph
@jax.jit
def score_interpolation(xs):
    # approximate by the decreasing sequence
    min_ = lax.cummin(xs)
    # find largest difference to decreasing approximation
    min_ = jnp.abs(min_ - xs).max()

    # the same but for increasing
    max_ = lax.cummax(xs)
    max_ = jnp.abs(max_ - xs).max()

    # take the better approximation
    return jnp.minimum(min_, max_)

