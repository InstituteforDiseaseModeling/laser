import numpy as np
import pytest

from laser_core.demographics.spatialpops import distribute_population_skewed
from laser_core.demographics.spatialpops import distribute_population_tapered


def test_distribute_population_skewed_basic():
    np.random.seed(42)
    result = distribute_population_skewed(1000, 5, 0.3)
    assert sum(result) == 1000
    assert len(result) == 5
    assert abs(sum(result[:2]) / 1000 - 0.8) <= 0.1


def test_distribute_population_skewed_alternative():
    np.random.seed(42)
    result = distribute_population_skewed(500, 3, 0.4)
    assert sum(result) == 500
    assert len(result) == 3
    assert abs(sum(result[:1]) / 500 - 0.6) <= 0.1


def test_distribute_population_skewed_zero_nodes():
    with pytest.raises(ValueError, match="Number of nodes must be greater than 0"):
        distribute_population_skewed(1000, 0, 0.3)


def test_distribute_population_skewed_zero_population():
    with pytest.raises(ValueError, match="Total population must be greater than 0."):
        distribute_population_skewed(0, 5, 0.3)


def test_distribute_population_skewed_invalid_fraction():
    with pytest.raises(ValueError, match="Fraction of rural population must be between 0 and 1."):
        distribute_population_skewed(1000, 5, -0.1)
    with pytest.raises(ValueError, match="Fraction of rural population must be between 0 and 1."):
        distribute_population_skewed(1000, 5, 1.5)


def test_distribute_population_tapered_basic():
    result = distribute_population_tapered(1000, 5)
    assert sum(result) == 1000
    assert len(result) == 5
    assert result[0] > result[1]
    assert result[1] > result[2]


def test_distribute_population_tapered_small_population():
    result = distribute_population_tapered(10, 4)
    assert sum(result) == 10
    assert len(result) == 4


def test_distribute_population_tapered_equal_distribution():
    result = distribute_population_tapered(10, 10)
    assert sum(result) == 10
    assert len(result) == 10
    assert 0 in result


def test_distribute_population_tapered_large_nodes():
    result = distribute_population_tapered(100, 50)
    assert sum(result) == 100
    assert len(result) == 50
    assert result[0] > result[-1]


def test_distribute_population_tapered_zero_nodes():
    with pytest.raises(ValueError, match="Both total_population and num_nodes must be greater than 0."):
        distribute_population_tapered(1000, 0)


def test_distribute_population_tapered_zero_population():
    with pytest.raises(ValueError, match="Both total_population and num_nodes must be greater than 0."):
        distribute_population_tapered(0, 5)


def test_distribute_population_tapered_adjustment():
    result = distribute_population_tapered(1200, 3)
    assert sum(result) == 1200
    assert len(result) == 3
