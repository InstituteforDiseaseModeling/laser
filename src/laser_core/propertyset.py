"""Implements a PropertySet class that can be used to store properties in a dictionary-like object."""

import json
from pathlib import Path


class PropertySet:
    """A class that can be used to store properties in a dictionary-like object with `.property` access to properties."""

    def __init__(self, *bags):
        for bag in bags:
            assert isinstance(bag, (type(self), dict))
            for key, value in (bag.__dict__ if isinstance(bag, type(self)) else bag).items():
                setattr(self, key, value)

    def to_dict(self):
        """Convert the PropertySet to a dictionary."""
        result = {}

        for key, value in self.__dict__.items():
            if isinstance(value, PropertySet):
                result[key] = value.to_dict()
            else:
                result[key] = value

        return result

    def save(self, filename):
        """
        Save the PropertySet to a specified file.

        Parameters:

            filename (str): The path to the file where the PropertySet will be saved.

        Returns:

            None
        """
        file = Path(filename)
        with file.open("w") as file:
            file.write(str(self))

        return

    def __getitem__(self, key):
        return getattr(self, key)

    def __add__(self, other):
        """
        Add another PropertySet to this PropertySet.

        This method allows the use of the ``+`` operator to combine two PropertySet instances.

        Parameters:

            other (PropertySet): The other PropertySet instance to add.

        Returns:

            PropertySet: A new PropertySet instance that combines the properties of both instances.
        """

        return PropertySet(self, other)

    def __iadd__(self, other):
        """
        Implements the in-place addition (``+=``) operator for the class.

        This method allows the instance to be updated with attributes from another
        instance of the same class or from a dictionary. If `other` is an instance
        of the same class, its attributes are copied to the current instance. If
        `other` is a dictionary, its key-value pairs are added as attributes to
        the current instance.

        Parameters:

            other (Union[type(self), dict]): The object or dictionary to add to the current instance.

        Returns:

            self: The updated instance with the new attributes.

        Raises:

            AssertionError: If `other` is neither an instance of the same class nor a dictionary.
        """

        assert isinstance(other, (type(self), dict))
        for key, value in (other.__dict__ if isinstance(other, type(self)) else other).items():
            setattr(self, key, value)
        return self

    def __len__(self):
        """
        Return the number of attributes in the instance.

        This method returns the number of attributes stored in the instance's
        __dict__ attribute, which represents the instance's namespace.

        Returns:

            int: The number of attributes in the instance.
        """

        return len(self.__dict__)

    def __str__(self) -> str:
        """
        Returns a string representation of the object's dictionary.

        This method is used to provide a human-readable string representation
        of the object, which includes all the attributes stored in the object's
        `__dict__`.

        Returns:

            str: A string representation of the object's dictionary.
        """

        return json.dumps(self.to_dict(), indent=4)

    def __repr__(self) -> str:
        """
        Return a string representation of the PropertySet instance.

        The string representation includes the class name and the dictionary of
        the instance's attributes.

        Returns:

            str: A string representation of the PropertySet instance.
        """

        return f"PropertySet({self.to_dict()!s})"

    def __contains__(self, key):
        """
        Check if a key is in the property set.

        Parameters:

            key (str): The key to check for existence in the property set.

        Returns:

            bool: True if the key exists in the property set, False otherwise.
        """

        return key in self.__dict__

    def __eq__(self, other):
        """
        Check if two PropertySet instances are equal.

        Parameters:

            other (PropertySet): The other PropertySet instance to compare.

        Returns:

            bool: True if the two instances are equal, False otherwise.
        """

        return self.to_dict() == other.to_dict()

    @staticmethod
    def load(filename):
        """
        Load a PropertySet from a specified file.

        Parameters:

            filename (str): The path to the file where the PropertySet is saved.

        Returns:

            PropertySet: The PropertySet instance loaded from the file.
        """
        with Path(filename).open("r") as file:
            data = json.load(file)

        return PropertySet(data)
