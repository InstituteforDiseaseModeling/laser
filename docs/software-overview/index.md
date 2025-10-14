# Software overview

LASER is a modeling framework that includes a variety of ways for users to implement the code to model infectious diseases. At the root of the framework is `laser-core`, a suite of components that can be assembled or customized to fit specific modeling needs. The LASER development team is also in the process of producing pre-built disease models crafted from `laser-core` components, which are tailored to address specific public health modeling questions. You can learn more about [creating custom models](../get-started/custom.md) using `laser-core` or running models using pre-built disease models in the [Get started modeling](../get-started/index.md) section.


## Design principles

<!-- Can include relevant software principles, or design choices. Included topics should be things that are unique to laser, such that modelers would need to know what this is in order to utilize laser properly (i.e. don't include general modeling principles, assume that the user already knows those). This can also include the high-level features of LASER, what makes it special. -->

The philosophy driving the development of LASER was to create a framework that was flexible, powerful, and fast, able to tackle a variety of complex modeling scenarios without sacrificing performance. But complexity often slows performance, and not every modeling question requires a full suite of model features. To solve this problem, LASER was designed as a set of core components, each with fundamental features that could be added--or not--to build working models. Users can optimize performance by creating models tailored to their research needs, only using components necessary for their modeling question. This building-block framework enables parsimony in model design, but also facilitates the building of powerful models with bespoke, complex dynamics.

## Software architecture

<!-- Framework of how laser works: insert diagram! -->
<!-- should also include explanations of what core is vs generic or other disease models -->

### Input and output files

<!-- All info on the input files and output files. If there are built-in reports, include those. Any type of data requirements should also be included here. Even if it's just that data needs to have a specific structure, include that here.

Even if there are no "required" files, there still needs to be guidelines on formats, basic information needs, example files, etc. Better to provide some guidelines and let users know they're flexible than to say "anything goes" with out any starting point -->


### Software components

Components are modular units of functionality within the simulation, responsible for performing specific updates or computations on the agent population or node-level data. Each component is implemented as a class with an initialization function to set up any required state and a step function to execute the component’s logic during each timestep.

<!-- [Deep dive into components and how they work, how they comprise laser functionality. Each "type" of component will have a topic section as needed]

Make it clear that this is not a comprehensive list, but a call-out for the various functions the user can play with (link to API docs for full listing of laser functions)


Need to make sure we explain all of the relevant/important parts! Eg, the classes used in the SIR tutorial should be all explained. -->


