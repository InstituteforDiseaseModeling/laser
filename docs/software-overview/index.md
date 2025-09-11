# Software Overview

introduction to how LASER pieces fit together, such as core, generic, etc. Also talk about how core is comprised of various components, and mention that the architecture will be discussed in detail on the architecture page.

Can include relevant software principles, or design choices. Included topics should be things that are unique to laser, such that modelers would need to know what this is in order to utilize laser properly (i.e. don't include general modeling principles, assume that the user already knows those)

## Software Architecture:

This is where the design elements of laser need to be clearly described, so the user can map all of the terminology. The sub-topic pages can delve into what each of those pieces are. Had this section as a sub-page, but it can probably live on the landing page (like this) unless there is a lot of content to be added for principles or ..?

For example:

- components
    - list component types (if there are categories or other specifics)
- laser scripts?
- objects?
- libraries?
- modules?
- models?
- other "things" that are used by laser?

This is not intended to be a full "parameter list" or set of API docs, this is a framework of how laser is put together


Will want a diagram to describe how it fits together, something like:

- laser-core contains components. Those components include:
    - laser frame
    - demographics
    - sorted queue
    - property set
    - migration module (??)
    - visualization
- Then use scripts to build components into a working model. These scripts can be:
    - custom scripts (link to custom section)
    - pre-built disease "scripts" (generic, measles, etc, link to those)


And then you model objects, which include: [need to make it clear how this relates to the model framework; separate entity? Builds into a structure?

- agent population
- nodes, which includes:
    - spatial information
    - population counts at node locations

Question: do you have to write up code to add into the model, eg for things like migration??

Will also need to show how various classes and functions slot into this, where in the hierarchy they fit. Eg, classes make up functions, functions make up components, components make up modules? Then what are model classes?

Info from the "applicaiton layer" section seems better suited to how you set up and run custom models. Also avoid saying it contains the following "components" in the list of tasks, as that can be confused with the laser-core components.