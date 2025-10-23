"""
Various probability distributions implemented using NumPy and Numba.

LASER based models generally move from pure NumPy to Numba-accelerated version of the core dynamics, e.g., transmission.

It would be a hassle to re-implement these functions for each desired distribution, so we provide these Numba-wrapped distributions here which can be passed in to other Numba compiled functions.

For example, a simple SIR model may want to parameterize the infectious period using a distribution. By passing in a Numba-wrapped distribution function, we can pick and parameterize a distribution based on configuration and sample from that distribution within the Numba-compiled SIR model without needing to re-implement the distribution logic within the SIR model itself.

A simple example of usage::

    import laser.core.distributions as dist

    # Create a Numba-wrapped beta distribution
    beta_dist = dist.beta(2.0, 5.0)

    # Assign the distribution to the model's infectious period distribution
    # so the transmission component can sample from it during simulation
    model.infectious_period_dist = beta_dist

"""

import numba as nb
import numpy as np


# beta(a, b, size=None)
def beta(a, b):
    r"""
    Beta distribution.
    $$f(x; a, b) = \\frac {x^{a-1} (1-x)^{b-1}} {B(a, b)}$$
    where B(a, b) is the beta function.
    """

    @nb.njit(nogil=True, cache=True)
    def _beta():
        return np.float32(np.random.beta(a, b))

    return _beta


# binomial(n, p, size=None)
def binomial(n, p):
    r"""
    Binomial distribution.
    $$f(k,n,p) = Pr(X = k) = \\binom {n} {k} p^k (1-p)^{n-k}$$
    where *n* is the number of trials and *p* is the probability of success [0, 1].
    """

    @nb.njit(nogil=True, cache=True)
    def _binomial():
        return np.int32(np.random.binomial(n, p))

    return _binomial


def constant_float(value):
    """
    Constant distribution.
    Always returns the same floating point value.
    """

    @nb.njit(nogil=True, cache=True)
    def _constant():
        return np.float32(value)

    return _constant


def constant_int(value):
    """
    Constant distribution.
    Always returns the same integer value.
    """

    @nb.njit(nogil=True, cache=True)
    def _constant():
        return np.int32(value)

    return _constant


# exponential(scale=1.0, size=None)
def exponential(scale):
    r"""
    Exponential distribution.
    $$f(x; \\frac {1} {\\beta}) = \\frac {1} {\\beta} e^{-\\frac {x} {\\beta}}$$
    where *β* is the scale parameter (β = 1 / λ).
    """

    @nb.njit(nogil=True, cache=True)
    def _exponential():
        return np.float32(np.random.exponential(scale))

    return _exponential


# gamma(shape, scale=1.0, size=None)
def gamma(shape, scale):
    r"""
    Gamma distribution.
    $$p(x) = x^{k-1} \\frac {e^{- x / \\theta}}{\\theta^k \\Gamma(k)}$$
    where *k* is the shape, *θ* is the scale, and *Γ(k)* is the gamma function.
    """

    @nb.njit(nogil=True, cache=True)
    def _gamma():
        return np.float32(np.random.gamma(shape, scale))

    return _gamma


# logistic(loc=0.0, scale=1.0, size=None)
def logistic(loc, scale):
    r"""
    Logistic distribution.
    $$P(x) = \\frac {e^{-(x - \\mu) / s}} {s (1 + e^{-(x - \\mu) / s})^2}$$
    where *μ* is the location parameter and *s* is the scale parameter.
    """

    @nb.njit(nogil=True, cache=True)
    def _logistic():
        return np.float32(np.random.logistic(loc, scale))

    return _logistic


# lognormal(mean=0.0, sigma=1.0, size=None)
def lognormal(mean, sigma):
    r"""
    Log-normal distribution.
    $$p(x) = \\frac {1} {\\sigma x \\sqrt {2 \\pi}} e^{- \\frac {(\ln x - \\mu)^2} {2 \\sigma^2}}$$
    where *μ* is the mean and *σ* is the standard deviation of the underlying normal distribution.
    """

    @nb.njit(nogil=True, cache=True)
    def _lognormal():
        return np.float32(np.random.lognormal(mean, sigma))

    return _lognormal


# # multinomial(n, pvals, size=None)
# def multinomial(n, pvals):
#     @nb.njit(nogil=True, cache=True)
#     def _multinomial():
#         return np.int32(np.random.multinomial(n, pvals))
#
#     return _multinomial


# negative_binomial(n, p, size=None)
def negative_binomial(n, p):
    r"""
    Negative binomial distribution.
    $$P(N; n, p) = \\frac {\\Gamma (N + n)} {N! \\Gamma (n)} p^n (1 - p)^N$$
    where *n* is the number of successes, *p* is the probability of success on each trial, *N + n* is the number of trials, and *Γ()* is the gamma function.
    When *n* is an integer,
    $$\\frac {\\Gamma (N + n)} {N! \\Gamma (n)} = \\binom {N + n - 1} {n - 1}$$
    which is the more common form of this term.
    """

    @nb.njit(nogil=True, cache=True)
    def _negative_binomial():
        return np.int32(np.random.negative_binomial(n, p))

    return _negative_binomial


# normal(loc=0.0, scale=1.0, size=None)
def normal(loc, scale):
    r"""
    Normal (Gaussian) distribution.
    $$p(x) = \\frac {1} {\\sqrt {2 \\pi \\sigma^2}} e^{- \\frac {(x - \\mu)^2} {2 \\sigma^2}}$$
    where *μ* is the mean and *σ* is the standard deviation.
    """

    @nb.njit(nogil=True, cache=True)
    def _normal():
        return np.float32(np.random.normal(loc, scale))

    return _normal


# poisson(lam=1.0, size=None)
def poisson(lam):
    r"""
    Poisson distribution.
    $$f( k ; \\lambda ) = \\frac {\\lambda^k e^{- \\lambda}} {k!}$$
    where *λ* is the expected number of events in the given interval.
    """

    @nb.njit(nogil=True, cache=True)
    def _poisson():
        return np.int32(np.random.poisson(lam))

    return _poisson


# uniform(low=0.0, high=1.0, size=None)
def uniform(low, high):
    r"""
    Uniform distribution.
    $$p(x) = \\frac {1} {b - a}$$
    where *a* is the lower bound and *b* is the upper bound, [*a*, *b*).
    """

    @nb.njit(nogil=True, cache=True)
    def _uniform():
        return np.float32(np.random.uniform(low, high))

    return _uniform


# weibull(a, size=None)
def weibull(a, lam):
    r"""
    Weibull distribution.
    $$X = \\lambda (- \\ln ( U ))^{1 / a}$$
    where *a* is the shape parameter and *λ* is the scale parameter.
    """

    @nb.njit(nogil=True, cache=True)
    def _weibull():
        return np.float32(lam * np.random.weibull(a))

    return _weibull


# Shared Numba sampling functions
@nb.njit(parallel=True, nogil=True, cache=True)
def sample_floats(fn, dest):
    """
    Fill an array with floating point values sampled from a Numba-wrapped distribution function.

    Parameters
    ----------
    fn : function
        Numba-wrapped distribution function returning float32 values.
    dest : np.ndarray
        Pre-allocated destination float32 array to store samples.

    Returns
    -------
    np.ndarray
        The destination array filled with sampled values.
    """
    count = dest.shape[0]
    for i in nb.prange(count):
        dest[i] = fn()
    return dest


@nb.njit(parallel=True, nogil=True, cache=True)
def sample_ints(fn, dest):
    """
    Fill an array with integer values sampled from a Numba-wrapped distribution function.

    Parameters
    ----------
    fn : function
        Numba-wrapped distribution function returning int32 values.
    dest : np.ndarray
        Pre-allocated destination int32 array to store samples.

    Returns
    -------
    np.ndarray
        The destination array filled with sampled values.
    """
    count = dest.shape[0]
    for i in nb.prange(count):
        dest[i] = fn()
    return dest
