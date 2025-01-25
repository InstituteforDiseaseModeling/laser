import numpy as np
import pytest

from laser_core.demographics.spatialpops import distribute_population_skewed
from laser_core.demographics.spatialpops import distribute_population_tapered


# Test the basic functionality of distribute_population_skewed
# Ensures the function correctly divides a population of 1000 among 5 nodes, with 30% in rural areas.
def test_distribute_population_skewed_basic():
    np.random.seed(42)  # Set seed for reproducibility
    result = distribute_population_skewed(tot_pop=1000, num_nodes=5, frac_rural=0.3)
    assert sum(result) == 1000  # Check total population remains constant
    assert len(result) == 5  # Verify the correct number of nodes
    assert abs(sum(result[:2]) / 1000 - 0.8) <= 0.1  # Verify rural fraction compliance


# Test an alternative scenario with distribute_population_skewed
# Verifies the function correctly handles a population of 500 and a rural fraction of 40%.
def test_distribute_population_skewed_alternative():
    np.random.seed(42)  # Set seed for reproducibility
    result = distribute_population_skewed(500, 3, 0.4)
    assert sum(result) == 500  # Ensure total population matches input
    assert len(result) == 3  # Verify the correct number of nodes
    assert abs(sum(result[:1]) / 500 - 0.6) <= 0.1  # Verify rural fraction compliance


# Test edge case where number of nodes is zero
# Ensures the function raises a ValueError for invalid input.
def test_distribute_population_skewed_zero_nodes():
    with pytest.raises(ValueError, match="Number of nodes must be greater than 0"):
        distribute_population_skewed(1000, 0, 0.3)


# Test edge case where total population is zero
# Ensures the function raises a ValueError for invalid input.
def test_distribute_population_skewed_zero_population():
    with pytest.raises(ValueError, match="Total population must be greater than 0."):
        distribute_population_skewed(0, 5, 0.3)


# Test invalid rural fraction inputs for distribute_population_skewed
# Verifies the function raises errors for fractions outside the range [0, 1].
def test_distribute_population_skewed_invalid_fraction():
    with pytest.raises(ValueError, match="Fraction of rural population must be between 0 and 1."):
        distribute_population_skewed(1000, 5, -0.1)
    with pytest.raises(ValueError, match="Fraction of rural population must be between 0 and 1."):
        distribute_population_skewed(1000, 5, 1.5)


# Test the basic functionality of distribute_population_tapered
# Verifies that the function correctly handles a population of 1000 across 5 nodes.
def test_distribute_population_tapered_basic():
    result = distribute_population_tapered(total_population=1000, num_nodes=5)
    assert sum(result) == 1000  # Ensure total population matches input
    assert len(result) == 5  # Verify the correct number of nodes
    assert result[0] > result[1]  # Check tapering pattern
    assert result[1] > result[2]


# Test edge case with small total population for distribute_population_tapered
# Ensures the function can handle small numbers without errors.
def test_distribute_population_tapered_small_population():
    result = distribute_population_tapered(10, 4)
    assert sum(result) == 10  # Check total population matches input
    assert len(result) == 4  # Verify the correct number of nodes


# Test case where total population is evenly distributed among nodes
# Ensures the function can handle equal division correctly.
def test_distribute_population_tapered_equal_distribution():
    result = distribute_population_tapered(10, 10)
    assert sum(result) == 10  # Ensure total population matches input
    assert len(result) == 10  # Verify the correct number of nodes
    assert 0 in result  # Verify some nodes may have zero population


# Test case with a large number of nodes relative to population
# Ensures the function handles large node counts gracefully.
def test_distribute_population_tapered_large_nodes():
    result = distribute_population_tapered(100, 50)
    assert sum(result) == 100  # Ensure total population matches input
    assert len(result) == 50  # Verify the correct number of nodes
    assert result[0] > result[-1]  # Check tapering pattern


# Test edge case where number of nodes is zero
# Ensures the function raises a ValueError for invalid input.
def test_distribute_population_tapered_zero_nodes():
    with pytest.raises(ValueError, match="Both total_population and num_nodes must be greater than 0."):
        distribute_population_tapered(1000, 0)


# Test edge case where total population is zero
# Ensures the function raises a ValueError for invalid input.
def test_distribute_population_tapered_zero_population():
    with pytest.raises(ValueError, match="Both total_population and num_nodes must be greater than 0."):
        distribute_population_tapered(0, 5)


# Test adjustment logic in distribute_population_tapered
# Verifies that the function correctly adjusts the population to match input.
def test_distribute_population_tapered_adjustment():
    result = distribute_population_tapered(1200, 3)
    assert sum(result) == 1200  # Ensure total population matches input
    assert len(result) == 3  # Verify the correct number of nodes
