"""Tests for the PriorityQueue class."""

import unittest

import numpy as np
import pytest

from laser_core.priorityqueue import PriorityQueue

messages = []


class TestPriorityQueue(unittest.TestCase):
    """Tests for the PriorityQueue class."""

    def setUp(self):
        # 31 41 59 26 53 58 97
        self.pq = PriorityQueue(7, np.array([31, 41, 59, 26, 53, 58, 97], dtype=np.int32))

    def test_push_pop(self):
        """Test pushing and popping elements from the priority queue."""
        self.pq.push(0)
        self.pq.push(1)
        self.pq.push(2)
        self.pq.push(3)
        self.pq.push(4)
        self.pq.push(5)
        self.pq.push(6)

        assert self.pq.popiv() == (3, 26)
        assert self.pq.popiv() == (0, 31)
        assert self.pq.popiv() == (1, 41)
        assert self.pq.popiv() == (4, 53)
        assert self.pq.popiv() == (5, 58)
        assert self.pq.popiv() == (2, 59)
        assert self.pq.popiv() == (6, 97)

    def test_push_pop_random(self):
        """Test pushing and popping random values from the priority queue."""
        values = np.random.randint(0, 100, 1024, dtype=np.int32)
        self.pq = PriorityQueue(len(values), values)
        for i in range(len(values)):
            self.pq.push(i)
        minimum = 0
        while len(self.pq) > 0:
            value = self.pq.popv()
            assert value >= minimum
            minimum = value

    def test_peek(self):
        """Test peeking at the top element of the priority queue."""
        self.pq.push(0)
        self.pq.push(1)
        self.pq.push(2)
        self.pq.push(3)
        self.pq.push(4)
        self.pq.push(5)
        self.pq.push(6)

        assert self.pq.peekiv() == (3, 26)

    def test_empty_peek(self):
        """Test peeking at the top element of an empty priority queue. Should raise an IndexError."""
        with pytest.raises(IndexError):
            _ = self.pq.peekiv()

    def test_peek_random(self):
        """Test peeking at the top element of the priority queue with random values."""
        values = np.random.randint(0, 100, 1024, dtype=np.int32)
        self.pq = PriorityQueue(len(values), values)
        for i in range(len(values)):
            self.pq.push(i)
        minimum = values.min()
        assert self.pq.peekv() == minimum

    def test_empty_pop(self):
        """Test popping from an empty priority queue. Should raise an IndexError."""
        with pytest.raises(IndexError):
            self.pq.pop()

    def test_full_push(self):
        """Test pushing to a full priority queue. Should raise an IndexError."""
        self.pq.push(0)
        self.pq.push(1)
        self.pq.push(2)
        self.pq.push(3)
        self.pq.push(4)
        self.pq.push(5)
        self.pq.push(6)

        with pytest.raises(IndexError):
            self.pq.push(7)

    def test_push_timing(self):
        """Test the timing of the push method."""
        import timeit

        np.random.seed(20240701)
        count = 1 << 20
        values = np.random.randint(0, 100, count, dtype=np.int32)
        self.pq = PriorityQueue(len(values), values)
        elapsed = timeit.timeit("for i in range(len(values)): self.pq.push(i)", globals={"values": values, "self": self}, number=1)
        messages.append(
            f"PriorityQueue.push() timing: {elapsed:0.4f} seconds for {count:9,} elements = {int(round(count / elapsed)):11,} elements/second"
        )

    def test_pop_timing(self):
        """Test the timing of the pop method."""
        import timeit

        np.random.seed(20240701)
        count = 1 << 20
        values = np.random.randint(0, 100, count, dtype=np.int32)
        self.pq = PriorityQueue(len(values), values)
        for i in range(len(values)):
            self.pq.push(i)

        elapsed = timeit.timeit("while len(self.pq): self.pq.popv()", globals={"self": self}, number=1)
        messages.append(
            f"PriorityQueue.popv() timing: {elapsed:0.4f} seconds for {count:9,} elements = {int(round(count / elapsed)):11,} elements/second"
        )

    # Test for peeki()
    def test_peeki(self):
        """Test peeking at the top index of the priority queue."""
        self.pq.push(0)
        self.pq.push(1)
        self.pq.push(2)
        self.pq.push(3)
        self.pq.push(4)
        self.pq.push(5)
        self.pq.push(6)

        assert self.pq.peeki() == 3

    # Test for peeki() on empty priority queue
    def test_empty_peeki(self):
        """Test peeking at the top index of an empty priority queue. Should raise an IndexError."""
        with pytest.raises(IndexError):
            _ = self.pq.peeki()

    # Test for popi()
    def test_popi(self):
        """Test popping the top index of the priority queue."""
        self.pq.push(0)
        self.pq.push(1)
        self.pq.push(2)
        self.pq.push(3)
        self.pq.push(4)
        self.pq.push(5)
        self.pq.push(6)

        assert self.pq.popi() == 3

    # Test for peekv() on empty priority queue
    def test_empty_peekv(self):
        """Test peeking at the top value of an empty priority queue. Should raise an IndexError."""
        with pytest.raises(IndexError):
            _ = self.pq.peekv()


if __name__ == "__main__":
    unittest.main(exit=False)
    for message in messages:
        print(message)
