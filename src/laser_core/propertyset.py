"""Implements a PropertySet class that can be used to store properties in a dictionary-like object."""

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

        Args:
            filename (str): The path to the file where the PropertySet will be saved.

        Returns:
            None
        """
        """Save the PropertySet to a file."""
        file = Path(filename)
        with file.open("w") as file:
            file.write(str(self))

        return

    def __add__(self, other):
        """
        Add another PropertySet to this PropertySet.
        This method allows the use of the `+` operator to combine two PropertySet instances.
        Args:
            other (PropertySet): The other PropertySet instance to add.
        Returns:
            PropertySet: A new PropertySet instance that combines the properties of both instances.
        """

        return PropertySet(self, other)

    def __iadd__(self, other):
        """
        Implements the in-place addition (+=) operator for the class.
        This method allows the instance to be updated with attributes from another
        instance of the same class or from a dictionary. If `other` is an instance
        of the same class, its attributes are copied to the current instance. If
        `other` is a dictionary, its key-value pairs are added as attributes to
        the current instance.
        Args:
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

        return str(self.__dict__)

    def __repr__(self) -> str:
        """
        Return a string representation of the PropertySet instance.
        The string representation includes the class name and the dictionary of
        the instance's attributes.
        Returns:
            str: A string representation of the PropertySet instance.
        """

        return f"PropertySet({self.__dict__!s})"

    def __contains__(self, key):
        """
        Check if a key is in the property set.
        Args:
            key (str): The key to check for existence in the property set.
        Returns:
            bool: True if the key exists in the property set, False otherwise.
        """

        return key in self.__dict__
