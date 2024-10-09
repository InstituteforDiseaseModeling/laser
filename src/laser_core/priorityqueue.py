"""PriorityQueue implementation using NumPy and Numba."""

from typing import Any

import numba as nb
import numpy as np


class PriorityQueue:
    """
    A priority queue implemented using NumPy arrays and sped-up with Numba.
    Using the algorithm from the Python heapq module.
    __init__ with an existing array of priority values
    __push__ with an index into priority values
    __pop__ returns the index of the highest priority value and its value
    """

    # https://github.com/python/cpython/blob/5592399313c963c110280a7c98de974889e1d353/Modules/_heapqmodule.c
    # https://github.com/python/cpython/blob/5592399313c963c110280a7c98de974889e1d353/Lib/heapq.py

    def __init__(self, capacity: int, values: np.ndarray):
        self.indices = np.zeros(capacity, dtype=np.uint32)
        self.values = values
        self.size = 0

        return

    def push(self, index) -> None:
        """
        Insert an element into the priority queue.
        This method adds an element at the end of the priority queue and then
        ensures the heap property is maintained by sifting the element down
        to its correct position.
        Args:
            index (int): The index of the element to be added to the priority queue.
        Raises:
            IndexError: If the priority queue is full.
        """

        if self.size >= len(self.indices):
            raise IndexError("Priority queue is full")
        self.indices[self.size] = index
        _siftdown(self.indices, self.values, 0, self.size)
        self.size += 1
        return

    def peeki(self) -> np.uint32:
        """
        Returns the index of the highest priority element in the priority queue without removing it.
        Raises:
            IndexError: If the priority queue is empty.
        Returns:
            np.uint32: The index of the highest priority element.
        """

        if self.size == 0:
            raise IndexError("Priority queue is empty")
        return self.indices[0]

    def peekv(self) -> Any:
        """
        Return the highest priority value from the priority queue without removing it.
        Raises:
            IndexError: If the priority queue is empty.
        Returns:
            Any: The value with the highest priority in the priority queue.
        """

        if self.size == 0:
            raise IndexError("Priority queue is empty")
        return self.values[self.indices[0]]

    def peekiv(self) -> tuple[np.uint32, Any]:
        """
        Returns the index and value of the highest priority element in the priority queue without removing it.
        Returns:
            tuple[np.uint32, Any]: A tuple containing the index and value of the highest priority element.
        Raises:
            IndexError: If the priority queue is empty.
        """

        if self.size == 0:
            raise IndexError("Priority queue is empty")
        return (self.indices[0], self.values[self.indices[0]])

    def popi(self) -> np.uint32:
        """
        Removes and returns the index of the highest priority element in the priority queue.
        This method first retrieves the index of the highest priority element using `peeki()`,
        then removes the element from the queue using `pop()`, and finally returns the index.
        Returns:
            np.uint32: The index of the highest priority element in the priority queue.
        """

        index = self.peeki()
        self.pop()

        return index

    def popv(self) -> Any:
        """
        Removes and returns the value at the front of the priority queue.
        This method first retrieves the value at the front of the queue without
        removing it by calling `peekv()`, and then removes the front element
        from the queue by calling `pop()`. The retrieved value is then returned.
        Returns:
            Any: The value at the front of the priority queue.
        """

        value = self.peekv()
        self.pop()

        return value

    def popiv(self) -> tuple[np.uint32, Any]:
        """
        Removes and returns the index and value of the highest priority element in the priority queue.
        This method first retrieves the index and value of the highest priority element using `peekiv()`,
        then removes the element from the queue using `pop()`, and finally returns the index and value.
        Returns:
            tuple[np.uint32, Any]: A tuple containing the index and value of the highest priority element.
        """
        ivtuple = self.peekiv()
        self.pop()

        return ivtuple

    def pop(self) -> None:
        """
        Removes the highest priority element from the priority queue.
        Raises:
            IndexError: If the priority queue is empty.
        Side Effects:
            Decreases the size of the priority queue by one.
            Reorganizes the internal structure of the priority queue to maintain the heap property.
        """

        if self.size == 0:
            raise IndexError("Priority queue is empty")
        self.size -= 1
        self.indices[0] = self.indices[self.size]
        _siftup(self.indices, self.values, 0, self.size)
        return

    def __len__(self):
        """
        Return the number of elements in the priority queue.
        Returns:
            int: The number of elements in the priority queue.
        """

        return self.size


@nb.njit((nb.uint32[:], nb.int32[:], nb.uint32, nb.uint32), nogil=True)
def _siftdown(indices, values, startpos, pos):  # pragma: no cover
    inewitem = indices[pos]
    vnewitem = values[inewitem]
    # Follow the path to the root, moving parents down until finding a place newitem fits.
    while pos > startpos:
        parentpos = (pos - 1) >> 1
        iparent = indices[parentpos]
        vparent = values[iparent]
        if vnewitem < vparent:
            indices[pos] = iparent
            pos = parentpos
            continue
        break
    indices[pos] = inewitem

    return


@nb.njit((nb.uint32[:], nb.int32[:], nb.uint32, nb.uint32), nogil=True)
def _siftup(indices, values, pos, size):  # pragma: no cover
    endpos = size
    startpos = pos
    inewitem = indices[pos]
    # Bubble up the smaller child until hitting a leaf.
    childpos = 2 * pos + 1  # leftmost child position
    while childpos < endpos:
        # Set childpos to index of smaller child.
        rightpos = childpos + 1
        if rightpos < endpos and not values[indices[childpos]] < values[indices[rightpos]]:
            childpos = rightpos
        # Move the smaller child up.
        indices[pos] = indices[childpos]
        pos = childpos
        childpos = 2 * pos + 1
    # The leaf at pos is empty now.  Put newitem there, and bubble it up
    # to its final resting place (by sifting its parents down).
    indices[pos] = inewitem
    _siftdown(indices, values, startpos, pos)
    return
