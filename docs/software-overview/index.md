# Software Overview

LASER is a modeling framework that includes a variety of ways for users to implement the code to model infectious diseases. At the root of the framework is LASER-core, a suite of components that can be assembled or customized to fit specific modeling needs. The LASER development team is also in the process of producing pre-built disease models crafted from LASER-core components, which are tailored to address specific public health modeling questions. You can learn more about [creating custom models](../get-started/custom.md) using LASER-core or running models using pre-built disease models in the [Get Started Modeling](../get-started/index.md) section.


## Design principles

<!-- Can include relevant software principles, or design choices. Included topics should be things that are unique to laser, such that modelers would need to know what this is in order to utilize laser properly (i.e. don't include general modeling principles, assume that the user already knows those) -->

The philosophy driving the development of LASER was to create a framework that was flexible, powerful, and fast, able to tackle a variety of complex modeling scenarios without sacrificing performance. But complexity often slows performance, and not every modeling question requires a full suite of model features. To solve this problem, LASER was designed as a set of core components, each with fundamental features that could be added--or not--to build working models. Users can optimize performance by creating models tailored to their research needs, only using components necessary for their modeling question. This building-block framework enables parsimony in model design, but also facilitates the building of powerful models with bespoke, complex dynamics.

## Software architecture

<!-- Framework of how laser works: insert diagram! -->


## Input and output files

<!-- All info on the input files and output files. If there are built-in reports, include those. Any type of data requirements should also be included here.

Even if there are no "required" files, there still needs to be guidelines on formats, basic information needs, example files, etc. Better to provide some guidelines and let users know they're flexible than to say "anything goes" with out any starting point -->
