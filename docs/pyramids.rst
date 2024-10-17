Population Pyramids
===================

The `AliasedDistribution` class provides a way to sample from a set of options
with unequal probabilities, e.g., a population pyramid.

The input to the `AliasedDistribution` constructor is an array of counts by bin
as we would naturally get from a population pyramid (# of people in each age bin).

`AliasedDistribution.sample()` returns *bin indices* so it is up to the user to
convert the values returned from `sample()` to actual ages.

Expected format of the population pyramid CSV file for `load_pyramid_csv()`::

    Header: Age,M,F
    start-end,#males,#females
    start-end,#males,#females
    start-end,#males,#females
    …
    start-end,#males,#females
    max+,#males,#females

E.g.::

    Age,M,F
    0-4,9596708,9175309
    5-9,10361680,9904126
    10-14,10781688,10274310
    15-19,11448281,10950664
    …
    90-94,757034,1281854
    95-99,172530,361883
    100+,27665,76635


`load_pyramid_csv()` returns a 5 column NumPy array with the following columns::

    0 - Lower bound of age bin, inclusive
    1 - Upper bound of age bin, inclusive
    2 - number of males in the age bin
    3 - number of females in the age bin
    4 - total number of people in the age bin
