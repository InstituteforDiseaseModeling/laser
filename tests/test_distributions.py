"""
Unit tests for laser_core.distributions module.

Note that this is not intended to test NumPy, Numba, or SciPy themselves, but rather to ensure that the
distributions implemented in laser_core.distributions have been "wired up" correctly.
"""

import unittest
from itertools import product
from time import perf_counter_ns

import numpy as np
from scipy.stats import beta as beta_ref
from scipy.stats import binom
from scipy.stats import expon
from scipy.stats import gamma as gamma_ref
from scipy.stats import ks_2samp
from scipy.stats import logistic as logistic_ref
from scipy.stats import lognorm
from scipy.stats import norm
from scipy.stats import poisson
from scipy.stats import uniform as uniform_ref
from scipy.stats import weibull_min

import laser_core.distributions as dists

NSAMPLES = 100_000
KS_THRESHOLD = 0.02  # Acceptable KS statistic for similarity


class TestDistributions(unittest.TestCase):
    def test_beta(self):
        params = [(0.5, 0.5), (5.0, 1.0), (1.0, 3.0), (2.0, 2.0), (2.0, 5.0)]
        for a, b in params:
            fn = dists.beta(a, b)
            samples = dists.sample_floats(fn, NSAMPLES, np.zeros(NSAMPLES, dtype=np.float32))
            ref_samples = beta_ref.rvs(a, b, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Beta({a},{b}) KS={stat}"

    def test_binomial(self):
        params = [(20, 0.5), (20, 0.7), (40, 0.5)]
        for n, p in params:
            fn = dists.binomial(n, p)
            samples = dists.sample_ints(fn, NSAMPLES, np.zeros(NSAMPLES, dtype=np.int32))
            ref_samples = binom.rvs(n, p, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Binomial({n},{p}) KS={stat}"

    def test_constant_float(self):
        values = [0.0, 1.0, -1.0, 3.14159, 2.71828]
        for value in values:
            fn = dists.constant_float(value)
            samples = dists.sample_floats(fn, NSAMPLES, np.zeros(NSAMPLES, dtype=np.float32))
            assert np.all(samples == np.float32(value)), f"Constant Float({value}) failed"

    def test_constant_int(self):
        values = [0, 1, -1, 42, 100]
        for value in values:
            fn = dists.constant_int(value)
            samples = dists.sample_ints(fn, NSAMPLES, np.zeros(NSAMPLES, dtype=np.int32))
            assert np.all(samples == np.int32(value)), f"Constant Int({value}) failed"

    def test_exponential(self):
        params = [0.5, 1.0, 1.5]
        for lam in params:
            scale = 1 / lam
            fn = dists.exponential(scale)
            samples = dists.sample_floats(fn, NSAMPLES, np.zeros(NSAMPLES, dtype=np.float32))
            ref_samples = expon.rvs(scale=scale, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Exponential({scale}) KS={stat}"

    def test_gamma(self):
        params = [(1.0, 2.0), (2.0, 2.0), (3.0, 2.0), (5.0, 1.0), (9.0, 0.5), (7.5, 1.0), (0.5, 1.0)]
        for shape, scale in params:
            fn = dists.gamma(shape, scale)
            samples = dists.sample_floats(fn, NSAMPLES, np.zeros(NSAMPLES, dtype=np.float32))
            ref_samples = gamma_ref.rvs(shape, scale=scale, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Gamma({shape},{scale}) KS={stat}"

    def test_logistic(self):
        params = [(5, 2), (9, 3), (9, 4), (6, 2), (2, 1)]
        for loc, scale in params:
            fn = dists.logistic(loc, scale)
            samples = dists.sample_floats(fn, NSAMPLES, np.zeros(NSAMPLES, dtype=np.float32))
            ref_samples = logistic_ref.rvs(loc=loc, scale=scale, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Logistic({loc},{scale}) KS={stat}"

    def test_lognormal(self):
        params = [(0, 1), (0, 0.5), (0, 0.25)]
        for mean, sigma in params:
            fn = dists.lognormal(mean, sigma)
            samples = dists.sample_floats(fn, NSAMPLES, np.zeros(NSAMPLES, dtype=np.float32))
            ref_samples = lognorm.rvs(sigma, scale=np.exp(mean), size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Lognormal({mean},{sigma}) KS={stat}"

    def test_negative_binomial(self):
        params = product([1, 2, 3, 4, 5], [1 / 2, 1 / 3, 1 / 4, 1 / 5])
        for r, p in params:
            fn = dists.negative_binomial(r, p)
            samples = dists.sample_ints(fn, NSAMPLES, np.zeros(NSAMPLES, dtype=np.int32))
            ref_samples = np.random.negative_binomial(r, p, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Negative Binomial({r},{p}) KS={stat}"

    def test_normal(self):
        params = [(0, 0.2), (0, 1.0), (0, 5.0), (-2, 0.5)]
        for mu, sigmasq in params:
            sigma = np.sqrt(sigmasq)
            fn = dists.normal(mu, sigma)
            samples = dists.sample_floats(fn, NSAMPLES, np.zeros(NSAMPLES, dtype=np.float32))
            ref_samples = norm.rvs(loc=mu, scale=sigma, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Normal({mu},{sigma}) KS={stat}"

    def test_poisson(self):
        params = [1, 4, 10]
        for lam in params:
            fn = dists.poisson(lam)
            samples = dists.sample_ints(fn, NSAMPLES, np.zeros(NSAMPLES, dtype=np.int32))
            ref_samples = poisson.rvs(mu=lam, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Poisson({lam}) KS={stat}"

    def test_uniform(self):
        params = [(0.0, 1.0), (0.25, 1.25), (0.0, 2.0), (-1.0, 1.0), (2.71828, 3.14159), (1.30, 4.20)]
        for low, high in params:
            fn = dists.uniform(low, high)
            samples = dists.sample_floats(fn, NSAMPLES, np.zeros(NSAMPLES, dtype=np.float32))
            ref_samples = uniform_ref.rvs(loc=low, scale=high - low, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Uniform({low},{high}) KS={stat}"

    def test_weibull(self):
        params = [(0.5, 1.0), (1.0, 1.0), (1.5, 1.0), (5.0, 1.0)]
        for a, lam in params:
            fn = dists.weibull(a, lam)
            samples = dists.sample_floats(fn, NSAMPLES, np.zeros(NSAMPLES, dtype=np.float32))
            ref_samples = weibull_min.rvs(a, scale=lam, size=NSAMPLES)
            stat, _ = ks_2samp(samples, ref_samples)
            assert stat < KS_THRESHOLD, f"Weibull({a},{lam}) KS={stat}"

    def test_perf(self):
        NSAMPLES = 1_000_000
        rng = np.random.default_rng(42)

        t0 = perf_counter_ns()
        _npsamples = rng.normal(loc=7.0, scale=0.875, size=NSAMPLES).astype(np.float32)
        t1 = perf_counter_ns()
        # print(f"NumPy normal: {(t1 - t0) / 1_000_000:.2f} ms")
        # print(f"{_npsamples.mean():.4f} ± {_npsamples.std():.4f}")

        gaussian = dists.normal(loc=7.0, scale=0.875)
        _ = dists.sample_floats(gaussian, 1000, np.zeros(1000, dtype=np.float32))  # warmup
        t2 = perf_counter_ns()
        _nbsamples = dists.sample_floats(gaussian, NSAMPLES, np.zeros(NSAMPLES, dtype=np.float32))
        t3 = perf_counter_ns()
        # print(f"LASER normal: {(t3 - t2) / 1_000_000:.2f} ms")
        # print(f"{_nbsamples.mean():.4f} ± {_nbsamples.std():.4f}")

        assert (t3 - t2) < (t1 - t0), "Numba-compatible distribution slower than NumPy"


if __name__ == "__main__":
    unittest.main()
