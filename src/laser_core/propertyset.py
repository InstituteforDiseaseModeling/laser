"""Implements a PropertySet class that can be used to store properties in a dictionary-like object."""

import json
from pathlib import Path


class PropertySet:
    """A class that can be used to store properties in a dictionary-like object with `.property` access to properties.

    Examples
    --------
    Basic Initialization:
        >>> from laser_core import PropertySet
        >>> ps = PropertySet()
        >>> ps['infection_status'] = 'infected'
        >>> ps['age'] = 35
        >>> print(ps.infection_status)  # Outputs: 'infected'
        >>> print(ps['age'])            # Outputs: 35

    Combining Two PropertySets:
        >>> ps1 = PropertySet({'immunity': 'high', 'region': 'north'})
        >>> ps2 = PropertySet({'infectivity': 0.7})
        >>> combined_ps = ps1 + ps2
        >>> print(combined_ps.to_dict())
        {'immunity': 'high', 'region': 'north', 'infectivity': 0.7}

    Creating a PropertySet from a Dictionary:
        >>> ps = PropertySet({'mything': 0.4, 'that_other_thing': 42})
        >>> print(ps.mything)            # Outputs: 0.4
        >>> print(ps.that_other_thing)   # Outputs: 42
        >>> print(ps.to_dict())
        {'mything': 0.4, 'that_other_thing': 42}

    Save and Load:
        >>> ps.save('properties.json')
        >>> loaded_ps = PropertySet.load('properties.json')
        >>> print(loaded_ps.to_dict())  # Outputs the saved properties

    Property Access and Length:
        >>> ps['status'] = 'susceptible'
        >>> ps['exposure_timer'] = 5
        >>> print(ps['status'])          # Outputs: 'susceptible'
        >>> print(len(ps))               # Outputs: 4

    In-Place Addition:
        >>> ps += {'new_timer': 10, 'infectivity': 0.8}
        >>> print(ps.to_dict())
        {'mything': 0.4, 'that_other_thing': 42, 'status': 'susceptible', 'exposure_timer': 5, 'new_timer': 10, 'infectivity': 0.8}
    """

    def __init__(self, *bags):
        """
        Initialize a PropertySet to manage properties in a dictionary-like structure.

        Parameters
        ----------
        *bags : iterable, optional
            A sequence of key-value pairs (e.g., lists, tuples, dictionaries) to initialize
            the PropertySet. Keys must be strings, and values can be any type.
        """

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
        """
        Retrieve the attribute of the object with the given key (e.g., ``ps[key]``).

        Parameters:

            key (str): The name of the attribute to retrieve.

        Returns:

            Any: The value of the attribute with the specified key.

        Raises:

            AttributeError: If the attribute with the specified key does not exist.
        """

        return getattr(self, key)

    def __setitem__(self, key, value):
        """
        Set the value of an attribute.
        This method allows setting an attribute of the instance using the
        dictionary-like syntax (e.g., ``ps[key] = value``).

        Parameters:
            key (str): The name of the attribute to set.
            value (any): The value to set for the attribute.

        Returns:
            None
        """

        setattr(self, key, value)

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
