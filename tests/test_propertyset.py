"""Tests for the PropertySet class."""

import tempfile
import unittest
from pathlib import Path

from laser_core.propertyset import PropertySet

messages = []


class TestPropertySet(unittest.TestCase):
    """Tests for the PropertySet class."""

    def test_empty_property_set(self):
        """Test an empty PropertySet."""
        # assert that the grab bag is empty
        gb = PropertySet()
        assert len(gb) == 0

    def test_single_dict_property_set(self):
        """Test initialization from a single dictionary."""
        # assert that the grab bag is initialized with a single dictionary
        gb = PropertySet({"a": 1, "b": 2})
        assert gb.a == 1
        assert gb.b == 2

    def test_single_property_set(self):
        """Test initialization from a single PropertySet."""
        # assert that the grab bag is initialized with a single PropertySet
        gb = PropertySet(PropertySet({"a": 1, "b": 2}))
        assert gb.a == 1
        assert gb.b == 2

    def test_multiple_dict_property_set(self):
        """Test initialization from multiple dictionaries."""
        # assert that the grab bag is initialized with multiple dictionaries
        gb = PropertySet({"a": 1, "b": 2}, {"c": 3, "d": 4})
        assert gb.a == 1
        assert gb.b == 2
        assert gb.c == 3
        assert gb.d == 4

    def test_multiple_property_set(self):
        """Test initialization from multiple PropertySets."""
        # assert that the grab bag is initialized with multiple PropertySets
        gb = PropertySet(PropertySet({"a": 1, "b": 2}), PropertySet({"c": 3, "d": 4}))
        assert gb.a == 1
        assert gb.b == 2
        assert gb.c == 3
        assert gb.d == 4

    def test_mixed_property_set(self):
        """Test initialization from a mix of dictionaries and PropertySets."""
        # assert that the grab bag is initialized with a mix of dictionaries and PropertySets
        gb = PropertySet({"a": 1, "b": 2}, PropertySet({"c": 3, "d": 4}))
        assert gb.a == 1
        assert gb.b == 2
        assert gb.c == 3
        assert gb.d == 4

    def test_add_dict_empty_property_set(self):
        """Test adding a dictionary to an empty PropertySet."""
        # assert that a dictionary can be added to an empty grab bag
        gb = PropertySet()
        gb += {"a": 1, "b": 2}
        assert gb.a == 1
        assert gb.b == 2

    def test_add_property_set(self):
        """Test adding a PropertySet to an existing PropertySet."""
        # assert that a PropertySet can be added to an existing PropertySet
        gb = PropertySet({"a": 1, "b": 2})
        gb += PropertySet({"c": 3, "d": 4})
        assert gb.a == 1
        assert gb.b == 2
        assert gb.c == 3
        assert gb.d == 4

    def test_add_dict_override(self):
        """Test that adding a subsequent dictionary to a PropertySet overrides existing values."""
        # assert that adding a subsequent dictionary to a PropertySet overrides existing values
        gb = PropertySet({"a": 1, "b": 2})
        gb += {"b": 3, "c": 4}
        assert gb.a == 1
        assert gb.b == 3
        assert gb.c == 4

    def test_add_property_set_override(self):
        """Test that adding a subsequent PropertySet to a PropertySet overrides existing values."""
        # assert that adding a subsequent PropertySet to a PropertySet overrides existing values
        gb = PropertySet({"a": 1, "b": 2})
        gb += PropertySet({"b": 3, "c": 4})
        assert gb.a == 1
        assert gb.b == 3
        assert gb.c == 4

    def test_add_property_set_new(self):
        """Test that PropertySet + PropertySet creates a new PropertySet _and_ does not alter the existing PropertySets"""
        # assert that PropertySet + PropertySet creates a new grab bag _and_ does not alter the existing grab bags
        gb1 = PropertySet({"a": 1, "b": 2})
        gb2 = PropertySet({"b": 3, "c": 4})
        gb3 = gb1 + gb2
        assert gb1.a == 1
        assert gb1.b == 2
        assert gb2.b == 3
        assert gb2.c == 4
        assert gb3.a == 1
        assert gb3.b == 3
        assert gb3.c == 4

    def test_str(self):
        """Test the __str__ method of the PropertySet class."""
        # assert that the __str__ method returns the expected string
        gb = PropertySet({"a": 1, "b": 2}, {"c": 3, "d": 4})
        assert str(gb) == str({"a": 1, "b": 2, "c": 3, "d": 4})

    def test_repr(self):
        """Test the __repr__ method of the PropertySet class."""
        # assert that the __repr__ method returns the expected string
        gb = PropertySet({"a": 1, "b": 2}, {"c": 3, "d": 4})
        assert repr(gb) == f"PropertySet({ {'a': 1, 'b': 2, 'c': 3, 'd': 4}!s})"

    def test_contains(self):
        """Test the __contains__ method of the PropertySet class."""
        # assert that the __contains__ method returns the expected results
        gb = PropertySet({"a": 1, "b": 2}, {"c": 3, "d": 4})
        assert "a" in gb
        assert "b" in gb
        assert "c" in gb
        assert "d" in gb
        assert "e" not in gb

    def test_to_dict(self):
        """Test the to_dict method of the PropertySet class."""
        # assert that the to_dict method returns the expected dictionary
        gb = PropertySet({"a": 1, "b": 2}, {"c": 3, "d": 4})
        assert gb.to_dict() == {"a": 1, "b": 2, "c": 3, "d": 4}

    def test_to_dict_nested(self):
        """Test the to_dict method of the PropertySet class with nested PropertySets."""
        # assert that the to_dict method returns the expected dictionary with nested PropertySets
        gb = PropertySet({"a": 1, "b": 2}, {"c": 3, "d": 4, "e": PropertySet({"f": 5, "g": 6})})
        assert gb.to_dict() == {"a": 1, "b": 2, "c": 3, "d": 4, "e": {"f": 5, "g": 6}}

    # Test save() method on temporary file
    def test_save(self):
        """Test the save method of the PropertySet class."""
        # assert that the save method writes the expected string to the file
        gb = PropertySet({"a": 1, "b": 2}, {"c": 3, "d": 4})
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as file:
            filename = Path(file.name)
            gb.save(filename)
            assert filename.read_text() == str(gb)


if __name__ == "__main__":
    unittest.main(exit=False)
    for message in messages:
        print(message)
