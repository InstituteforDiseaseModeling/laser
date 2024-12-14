"""
Unit tests for the AliasedDistribution class and the load_pyramid_csv function from the laser_core.demographics module.

This module contains the following test cases:
- TestAliasedDistribution: Tests for the AliasedDistribution class.
    - test_aliased_distribution: Tests the sampling method of AliasedDistribution with a large number of samples.
    - test_aliased_distribution_single: Tests the sampling method of AliasedDistribution with single samples in a loop.
    - test_load_pyramid_csv: Tests the load_pyramid_csv function for correct data loading.
    - test_catch_bad_header: Tests load_pyramid_csv for handling incorrect header format.
    - test_catch_wrong_data_format: Tests load_pyramid_csv for handling incorrect data format.
    - test_catch_wrong_max_format: Tests load_pyramid_csv for handling incorrect max age format.
    - test_catch_descending_start_age: Tests load_pyramid_csv for handling non-ascending start ages.
    - test_catch_descending_end_age: Tests load_pyramid_csv for handling non-ascending end ages.
    - test_catch_negative_male_count: Tests load_pyramid_csv for handling negative male counts.
    - test_catch_negative_female_count: Tests load_pyramid_csv for handling negative female counts.

Each test case uses the unittest framework and pytest for exception handling.
"""

import re
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import kstest

from laser_core.demographics import AliasedDistribution
from laser_core.demographics import load_pyramid_csv


class TestAliasedDistribution(unittest.TestCase):
    def test_aliased_distribution(self):
        pyramid = load_pyramid_csv(Path(__file__).parent / "data" / "us-pyramid-2023.csv")
        both = pyramid[:, 2] + pyramid[:, 3]
        ad = AliasedDistribution(both)  # males and females
        nsamples = 100_000
        samples = ad.sample(nsamples)
        counts = np.zeros(samples.max() + 1, dtype=np.int32)
        np.add.at(counts, samples, 1)
        f_of_x = counts.cumsum()
        g_of_x = both.cumsum()
        total = both.sum()
        g_of_x = (g_of_x / (total / nsamples)).astype(g_of_x.dtype)
        test = kstest(f_of_x, g_of_x)
        assert test.pvalue >= 0.999999, f"Kolmogorov-Smirnov test failed {test.pvalue=}"
        return

    def test_aliased_distribution_single(self):
        pyramid = load_pyramid_csv(Path(__file__).parent / "data" / "us-pyramid-2023.csv")
        both = pyramid[:, 2] + pyramid[:, 3]
        ad = AliasedDistribution(both)  # males and females
        nsamples = 100_000
        samples = np.zeros(nsamples, dtype=np.int32)
        for i in range(nsamples):
            samples[i] = ad.sample()
        counts = np.zeros(samples.max() + 1, dtype=np.int32)
        np.add.at(counts, samples, 1)
        f_of_x = counts.cumsum()
        g_of_x = both.cumsum()
        total = both.sum()
        g_of_x = (g_of_x / (total / nsamples)).astype(g_of_x.dtype)
        test = kstest(f_of_x, g_of_x)
        assert test.pvalue >= 0.999999, f"Kolmogorov-Smirnov test failed {test.pvalue=}"
        return

    def test_load_pyramid_csv(self):
        pyramid = load_pyramid_csv(Path(__file__).parent / "data" / "us-pyramid-2023.csv")
        assert pyramid.shape == (21, 4), f"Expected pyramid shape == (21, 4) got {pyramid.shape=}"
        assert pyramid[0, 0] == 0, f"Expected pyramid[0, 0] == 0 got {pyramid[0, 0]=}"
        assert pyramid[0, 1] == 4, f"Expected pyramid[0, 1] == 4 got {pyramid[0, 1]=}"
        assert pyramid[-1, 0] == 100, f"Expected pyramid[-1, 0] == 100 got {pyramid[-1, 0]=}"
        assert pyramid[-1, 1] == 100, f"Expected pyramid[-1, 1] == 100 got {pyramid[-1, 1]=}"
        return

    def test_load_pyramid_csv_string(self):
        file = Path(__file__).parent / "data" / "us-pyramid-2023.csv"
        pyramid = load_pyramid_csv(str(file))
        assert pyramid.shape == (21, 4), f"Expected pyramid shape == (21, 4) got {pyramid.shape=}"
        assert pyramid[0, 0] == 0, f"Expected pyramid[0, 0] == 0 got {pyramid[0, 0]=}"
        assert pyramid[0, 1] == 4, f"Expected pyramid[0, 1] == 4 got {pyramid[0, 1]=}"
        assert pyramid[-1, 0] == 100, f"Expected pyramid[-1, 0] == 100 got {pyramid[-1, 0]=}"
        assert pyramid[-1, 1] == 100, f"Expected pyramid[-1, 1] == 100 got {pyramid[-1, 1]=}"
        return

    def test_catch_bad_header(self):
        # Missing header line
        text = """0-4,9596708,9175309
5-9,10361680,9904126
10-14,10781688,10274310
15-19,11448281,10950664
20-24,11384263,10964564
25-29,11438191,11078541
30-34,12048644,11797245
35-39,11541070,11299124
40-44,11160804,11028013
45-49,10160722,10185712
50-54,10578142,10641874
55-59,10334788,10678099
60-64,10387785,10997888
65-69,9233967,10097028
70-74,7104835,8189102
75-79,5119582,6295285
80-84,3030378,3983607
85-89,1626571,2440362
90-94,757034,1281854
95-99,172530,361883
100+,27665,76635
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as file:
            filename = Path(file.name)
            file.write(text)
            file.close()
            with pytest.raises(ValueError, match="Header line is not 'Age,M,F'."):
                load_pyramid_csv(filename)
        return

    def test_catch_wrong_data_format(self):
        # Semicolon instead of comma
        text = """Age,M,F
0-4;9596708;9175309
5-9;10361680;9904126
10-14;10781688;10274310
15-19;11448281;10950664
20-24;11384263;10964564
25-29;11438191;11078541
30-34;12048644;11797245
35-39;11541070;11299124
40-44;11160804;11028013
45-49;10160722;10185712
50-54;10578142;10641874
55-59;10334788;10678099
60-64;10387785;10997888
65-69;9233967;10097028
70-74;7104835;8189102
75-79;5119582;6295285
80-84;3030378;3983607
85-89;1626571;2440362
90-94;757034;1281854
95-99;172530;361883
100+;27665;76635
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as file:
            filename = Path(file.name)
            file.write(text)
            file.close()
            with pytest.raises(ValueError, match="Data lines are not in the expected format 'low-high,males,females'."):
                load_pyramid_csv(filename)
        return

    def test_catch_wrong_max_format(self):
        # Missing plus sign
        text = """Age,M,F
0-4,9596708,9175309
5-9,10361680,9904126
10-14,10781688,10274310
15-19,11448281,10950664
20-24,11384263,10964564
25-29,11438191,11078541
30-34,12048644,11797245
35-39,11541070,11299124
40-44,11160804,11028013
45-49,10160722,10185712
50-54,10578142,10641874
55-59,10334788,10678099
60-64,10387785,10997888
65-69,9233967,10097028
70-74,7104835,8189102
75-79,5119582,6295285
80-84,3030378,3983607
85-89,1626571,2440362
90-94,757034,1281854
95-99,172530,361883
100,27665,76635
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as file:
            filename = Path(file.name)
            file.write(text)
            file.close()
            with pytest.raises(ValueError, match=re.escape("Last data line is not in the expected format 'max+,males,females'.")):
                load_pyramid_csv(filename)
        return

    def test_catch_descending_start_age(self):
        # 30-34 is before 25-29
        text = """Age,M,F
0-4,9596708,9175309
5-9,10361680,9904126
10-14,10781688,10274310
15-19,11448281,10950664
20-24,11384263,10964564
30-34,12048644,11797245
25-29,11438191,11078541
35-39,11541070,11299124
40-44,11160804,11028013
45-49,10160722,10185712
50-54,10578142,10641874
55-59,10334788,10678099
60-64,10387785,10997888
65-69,9233967,10097028
70-74,7104835,8189102
75-79,5119582,6295285
80-84,3030378,3983607
85-89,1626571,2440362
90-94,757034,1281854
95-99,172530,361883
100+,27665,76635
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as file:
            filename = Path(file.name)
            file.write(text)
            file.close()
            with pytest.raises(ValueError, match="Starting ages are not in ascending order."):
                load_pyramid_csv(filename)
        return

    def test_catch_descending_end_age(self):
        # Error in 60-75
        text = """Age,M,F
0-4,9596708,9175309
5-9,10361680,9904126
10-14,10781688,10274310
15-19,11448281,10950664
20-24,11384263,10964564
25-29,11438191,11078541
30-34,12048644,11797245
35-39,11541070,11299124
40-44,11160804,11028013
45-49,10160722,10185712
50-54,10578142,10641874
55-59,10334788,10678099
60-75,10387785,10997888
65-69,9233967,10097028
70-74,7104835,8189102
75-79,5119582,6295285
80-84,3030378,3983607
85-89,1626571,2440362
90-94,757034,1281854
95-99,172530,361883
100+,27665,76635
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as file:
            filename = Path(file.name)
            file.write(text)
            file.close()
            with pytest.raises(ValueError, match="Ending ages are not in ascending order."):
                load_pyramid_csv(filename)
        return

    def test_catch_negative_bin_start_age(self):
        # -10-14 is not a valid age range
        text = """Age,M,F
0-4,9596708,9175309
5-9,10361680,9904126
-10-14,10781688,10274310
15-19,11448281,10950664
20-24,11384263,10964564
25-29,11438191,11078541
30-34,12048644,11797245
35-39,11541070,11299124
40-44,11160804,11028013
45-49,10160722,10185712
50-54,10578142,10641874
55-59,10334788,10678099
60-64,10387785,10997888
65-69,9233967,10097028
70-74,7104835,8189102
75-79,5119582,6295285
80-84,3030378,3983607
85-89,1626571,2440362
90-94,757034,1281854
95-99,172530,361883
100+,27665,76635
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as file:
            filename = Path(file.name)
            file.write(text)
            file.close()
            with pytest.raises(ValueError, match="Data lines are not in the expected format 'low-high,males,females'."):
                load_pyramid_csv(filename)
        return

    def test_catch_negative_bin_end_age(self):
        # 10 - -14 is not a valid age range
        text = """Age,M,F
0-4,9596708,9175309
5-9,10361680,9904126
10--14,10781688,10274310
15-19,11448281,10950664
20-24,11384263,10964564
25-29,11438191,11078541
30-34,12048644,11797245
35-39,11541070,11299124
40-44,11160804,11028013
45-49,10160722,10185712
50-54,10578142,10641874
55-59,10334788,10678099
60-64,10387785,10997888
65-69,9233967,10097028
70-74,7104835,8189102
75-79,5119582,6295285
80-84,3030378,3983607
85-89,1626571,2440362
90-94,757034,1281854
95-99,172530,361883
100+,27665,76635
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as file:
            filename = Path(file.name)
            file.write(text)
            file.close()
            with pytest.raises(ValueError, match="Data lines are not in the expected format 'low-high,males,females'."):
                load_pyramid_csv(filename)
        return

    def test_catch_negative_male_count(self):
        # Negative male count in 10-14 bin
        text = """Age,M,F
0-4,9596708,9175309
5-9,10361680,9904126
10-14,-10781688,10274310
15-19,11448281,10950664
20-24,11384263,10964564
25-29,11438191,11078541
30-34,12048644,11797245
35-39,11541070,11299124
40-44,11160804,11028013
45-49,10160722,10185712
50-54,10578142,10641874
55-59,10334788,10678099
60-64,10387785,10997888
65-69,9233967,10097028
70-74,7104835,8189102
75-79,5119582,6295285
80-84,3030378,3983607
85-89,1626571,2440362
90-94,757034,1281854
95-99,172530,361883
100+,27665,76635
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as file:
            filename = Path(file.name)
            file.write(text)
            file.close()
            with pytest.raises(ValueError, match="Data lines are not in the expected format 'low-high,males,females'."):
                load_pyramid_csv(filename)
        return

    def test_negative_female_count(self):
        # Negative female count in 10-14 bin
        text = """Age,M,F
0-4,9596708,9175309
5-9,10361680,9904126
10-14,10781688,-10274310
15-19,11448281,10950664
20-24,11384263,10964564
25-29,11438191,11078541
30-34,12048644,11797245
35-39,11541070,11299124
40-44,11160804,11028013
45-49,10160722,10185712
50-54,10578142,10641874
55-59,10334788,10678099
60-64,10387785,10997888
65-69,9233967,10097028
70-74,7104835,8189102
75-79,5119582,6295285
80-84,3030378,3983607
85-89,1626571,2440362
90-94,757034,1281854
95-99,172530,361883
100+,27665,76635
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as file:
            filename = Path(file.name)
            file.write(text)
            file.close()
            with pytest.raises(ValueError, match="Data lines are not in the expected format 'low-high,males,females'."):
                load_pyramid_csv(filename)
        return


if __name__ == "__main__":
    unittest.main()
