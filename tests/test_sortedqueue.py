"""Tests for the SortedQueue class."""

import unittest
from typing import ClassVar

import numpy as np
import pytest

from laser_core.sortedqueue import SortedQueue


class TestSortedQueue(unittest.TestCase):
    """Tests for the SortedQueue class."""

    messages: ClassVar = []

    # Called before each test
    def setUp(self):
        # 31 41 59 26 53 58 97
        self.sq = SortedQueue(7, np.array([31, 41, 59, 26, 53, 58, 97], dtype=np.int32))

    # Called once after all tests
    @classmethod
    def tearDownClass(cls):
        print()
        for message in cls.messages:
            print(message)

    def test_push_pop(self):
        """Test pushing and popping elements from the sorted queue."""
        self.sq.push(0)
        self.sq.push(1)
        self.sq.push(2)
        self.sq.push(3)
        self.sq.push(4)
        self.sq.push(5)
        self.sq.push(6)

        assert self.sq.popiv() == (3, 26)
        assert self.sq.popiv() == (0, 31)
        assert self.sq.popiv() == (1, 41)
        assert self.sq.popiv() == (4, 53)
        assert self.sq.popiv() == (5, 58)
        assert self.sq.popiv() == (2, 59)
        assert self.sq.popiv() == (6, 97)

    def test_push_pop_random(self):
        """Test pushing and popping random values from the sorted queue."""
        values = np.random.randint(0, 100, 1024, dtype=np.int32)
        self.sq = SortedQueue(len(values), values)
        for i in range(len(values)):
            self.sq.push(i)
        minimum = 0
        while len(self.sq) > 0:
            value = self.sq.popv()
            assert value >= minimum
            minimum = value

    def test_peek(self):
        """Test peeking at the top element of the sorted queue."""
        self.sq.push(0)
        self.sq.push(1)
        self.sq.push(2)
        self.sq.push(3)
        self.sq.push(4)
        self.sq.push(5)
        self.sq.push(6)

        assert self.sq.peekiv() == (3, 26)
        assert self.sq.peekiv() == (3, 26)  # Do it again to prove we didn't modify the queue.

    def test_empty_peek(self):
        """Test peeking at the top element of an empty sorted queue. Should raise an IndexError."""
        with pytest.raises(IndexError):
            _ = self.sq.peekiv()

    def test_peek_random(self):
        """Test peeking at the top element of the sorted queue with random values."""
        values = np.random.randint(0, 100, 1024, dtype=np.int32)
        self.sq = SortedQueue(len(values), values)
        for i in range(len(values)):
            self.sq.push(i)
        minimum = values.min()
        assert self.sq.peekv() == minimum
        assert self.sq.peekv() == minimum  # Do it again to prove we didn't modify the queue.

    def test_empty_pop(self):
        """Test popping from an empty sorted queue. Should raise an IndexError."""
        with pytest.raises(IndexError):
            self.sq._SortedQueue__pop()

    def test_full_push(self):
        """Test pushing to a full sorted queue. Should raise an IndexError."""
        self.sq.push(0)
        self.sq.push(1)
        self.sq.push(2)
        self.sq.push(3)
        self.sq.push(4)
        self.sq.push(5)
        self.sq.push(6)

        with pytest.raises(IndexError):
            self.sq.push(7)

    def test_push_timing(self):
        """Test the timing of the push method."""
        import timeit

        np.random.seed(20240701)
        count = 1 << 20
        values = np.random.randint(0, 100, count, dtype=np.int32)
        self.sq = SortedQueue(len(values), values)
        elapsed = timeit.timeit("for i in range(len(values)): self.sq.push(i)", globals={"values": values, "self": self}, number=1)
        self.messages.append(
            f"SortedQueue.push() timing: {elapsed:0.4f} seconds for {count:9,} elements = {int(round(count / elapsed)):11,} elements/second"
        )

    def test_pop_timing(self):
        """Test the timing of the pop method."""
        import timeit

        np.random.seed(20240701)
        count = 1 << 20
        values = np.random.randint(0, 100, count, dtype=np.int32)
        self.sq = SortedQueue(len(values), values)
        for i in range(len(values)):
            self.sq.push(i)

        elapsed = timeit.timeit("while len(self.sq): self.sq.popv()", globals={"self": self}, number=1)
        self.messages.append(
            f"SortedQueue.popv() timing: {elapsed:0.4f} seconds for {count:9,} elements = {int(round(count / elapsed)):11,} elements/second"
        )

    # Test for peeki()
    def test_peeki(self):
        """Test peeking at the top index of the sorted queue."""
        self.sq.push(0)
        self.sq.push(1)
        self.sq.push(2)
        self.sq.push(3)
        self.sq.push(4)
        self.sq.push(5)
        self.sq.push(6)

        assert self.sq.peeki() == 3
        assert self.sq.peeki() == 3  # Do it again to prove we didn't modify the queue.

    # Test for peeki() on empty sorted queue
    def test_empty_peeki(self):
        """Test peeking at the top index of an empty sorted queue. Should raise an IndexError."""
        with pytest.raises(IndexError):
            _ = self.sq.peeki()

    # Test for popi()
    def test_popi(self):
        """Test popping the top index of the sorted queue."""
        self.sq.push(0)
        self.sq.push(1)
        self.sq.push(2)
        self.sq.push(3)
        self.sq.push(4)
        self.sq.push(5)
        self.sq.push(6)

        assert self.sq.popi() == 3
        assert self.sq.popi() == 0  # Pop the next one to prove we updated the queue.

    # Test for peekv() on empty sorted queue
    def test_empty_peekv(self):
        """Test peeking at the top value of an empty sorted queue. Should raise an IndexError."""
        with pytest.raises(IndexError):
            _ = self.sq.peekv()

    def test_push_pop_random_type(self):
        """Test pushing and popping random values from the sorted queue."""
        values = np.random.randint(-128, 128, 1024, dtype=np.int64)
        for dtype in [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64]:
            self.sq = SortedQueue(len(values), values.astype(dtype))
            for i in range(len(values)):
                self.sq.push(i)
            minimum = values.min()
            while len(self.sq) > 0:
                value = self.sq.popv()
                assert value >= minimum
                minimum = value
        values += 128  # Shift values to positive range
        for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
            self.sq = SortedQueue(len(values), values.astype(dtype))
            for i in range(len(values)):
                self.sq.push(i)
            minimum = values.min()
            while len(self.sq) > 0:
                value = self.sq.popv()
                assert value >= minimum
                minimum = value


if __name__ == "__main__":
    unittest.main(exit=False)
