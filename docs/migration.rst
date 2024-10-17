=========
Migration
=========

Some things to note:

These models all take in some parameters, along with a vector of populations and distances between nodes, and spit out a matrix defining the connection between nodes.

- When comparing to other models, it's important to consider how implementation of the migration / connection model affects interpretation of the parameters.  For example, a migration matrix could be implemented as a per-capita rate of travel from :math:`i` to :math:`j`, or as a total flux of people from :math:`i` to :math:`j`.  If your migration model has a term that scales like :math:`p_i^a`, using a per-capita rate introduces an implicit :math:`+1` into the exponent.  Or, depending on how infectivity / mixing is handled locally, if the introduced infectivity is normalized to local population, that might introduce an ambiguity in interpreting the exponent on the destination population, is it :math:`b` or :math:`b-1`, effectively?  In the end, these are terms that could be calibrated away, but useful to keep this in mind for interpretation and comparison against other models - we aim to support users defining their own migration models and implementations, and this sort of ambiguity is important to keep in mind.

- I'm aiming to do most of this with element-by-element numpy functions when I can, though loops would probably translate more obviously to numba/c.  I don't expect that computation of these matrices will be a substantial part of overall computational spend on a model either way.

- I have not tested nor written code to enforce conditions on the inputs.  This should be done - e.g., populations coming in as integers can present wrapping issues when we start exponentiating and multiplying them (signed integers in particular can be a problem because you may wrap into negative numbers).  So some input checking and such needs to be done.

- It's also worth investigating whether using large floats makes sense when computing these formulas, or whether we should put operations in a specific order.  This is because depending on the choice of someone's metapop network and spatial model parameters, and the order of computations, we can end up multiplying, dividing, summing over numbers that can be across really different scales, so weirdness might happen with loss of precision?  The integer issue above is more concerning and one that I have run into in the past.

- Distances on the diagonal of the distance matrix should always be 0.  We should check for 0s elsewhere and throw an error.  It's also nice to be able to use numpy element-by-element math without constant div-by-zero errors for the diagonal elements, so maybe each function should start by adding epsilon to the diagonal of the distance matrix?  We're going to zero out those terms in the network anyway...

Gravity model
=============

Functional form:
:math:`M_{i,j} = k \frac{p_i^a p_j^b}{d_{i,j}^c}`

Special cases of the gravity model (as noted above, both population exponents are subject to :math:`\pm1` ambiguity depending on implementation of spatial connectivity and local mixing):

- Xia's model: :math:`a = 0`
- mean-field model: :math:`c = 0, a = 1, b = 1`
- spatial diffusion: :math:`a = 0, b = 0`

Capping the total fraction of population that can migrate / infectivity that can be exported on a given timestep
================================================================================================================

Because the inputs to spatial models (populations, distances) can vary over many orders of magnitude, we can run into situations where a a small number of nodes, often those closest to but distinct from large population centers, will end up with huge outflows. The below illustrates an easy way to implement a standard gravity/radiation/etc. model, but cap the total amount of migration/infectivity outflow from any single metapopulation.

The Competing Destinations model
================================

There are many models that aim to account for the impact of competition or synergy between potential destinations. Some aim to account for some "screening" effect of travel to distant destinations due to competition from attractive destinations closer to the origin :math:`i`. This model, in contrast, (Fotheringham AS. Spatial flows and spatial patterns. Environment and Planning A. 1984;16(4):529–543) aims to account for effects from other attractive destinations near destination :math:`j`; notably, this effect could be synergistic or antagonistic, depending on the sign of the exponent :math:`\delta`.

For example, in a "synergistic" version, perhaps migratory flow from Boston to Baltimore is higher than flow between two comparator cities of similar population and at similar distance, because the proximity of Washington, D.C. to Baltimore makes travel to Baltimore more attractive to Bostonians – this would be accounted for by a positive value of :math:`\delta`. On the other hand, this term may also be "antagonistic", if Washington is such an attractive destination that Bostonians eschew travel to Baltimore entirely; this would indicate a negative value of :math:`\delta`.

Mathematical Formulation:
:math:`M_{i,j} = k \frac{p_i^a p_j^b}{d_{i,j}^c} \left(\sum_{k \ne i,j} \frac{p_k^b}{d_{jk}^c}\right)^\delta`

Stouffer's rank model
=====================

Stouffer (Stouffer SA. Intervening opportunities: a theory relating mobility and distance. American Sociological Review. 1940;5(6):845–867) argued that human mobility patterns do not respond to absolute distance directly, but only indirectly through the accumulation of intervening opportunities for destinations. Stouffer thus proposed a model with no distance-dependence at all, rather only a term that accounts for all potential destinations closer than destination :math:`j`; thus, longer-distance travel depends on the density of attractive destinations at shorter distances.

Mathematical formulation:

Define :math:`\Omega(i,j)` to be the set of all locations :math:`k` such that :math:`D_{i,k} \le D_{i,j}`

:math:`M_{i,j} = k p_i^a \sum_j \left(\frac{p_j}{\sum_{k \in \Omega(i,j)} p_k}\right)^b`

This presents us with the choice of whether or not the origin population :math:`i` is included in :math:`\Omega` – i.e., does the same "gravity" that brings others to visit a community reduce the propensity of that community's members to travel to other communities?

The Stouffer model does not include impact from the local community:
:math:`\Omega(i,j) = \left(k: 0 < D_{i,k} \le D_{i,j}\right)`.

The Stouffer variant model does include the impact of the local community:
:math:`\Omega(i,j) = \left(k: 0 \le D_{i,k} \le D_{i,j}\right)`.

Rather than implementing twice, this implementation of the Stouffer model will include a parameter "include_home."

Radiation model
===============

The radiation model (Simini F, González MC, Maritan A, Barabási AL. A universal model for mobility and migration patterns. Nature. 2012;484(7392):96–100.) is a parameter-free model (up to an overall scaling constant for total migration flux), derived from arguments around job-related commuting but essentially capturing a situation in which outbound migration flux from origin to destination is enhanced by destination population and absorbed by the density of nearer destinations.

Mathematical formulation:
With :math:`\Omega` defined as above in the Stouffer model,

:math:`M_{i,j} = k \frac{p_i p_j}{\left(p_i + \sum_{k \in \Omega(i,j)} p_k\right)\left(p_i + p_j + \sum_{k \in \Omega(i,j)} p_k\right)}`

We again use the parameter "include_home" to determine whether or not location :math:`i` is to be included in :math:`\Omega(i,j)`.
