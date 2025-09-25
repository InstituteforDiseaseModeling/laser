# Development and Contributing to Code

<!-- Here's where all the relevant info for submitting tickets for bugs or feature requests, how to contribute to code, etc should go. No need to include persona information, users will navigate the docs based on what tasks they want to do. [If needed, this topic can be broken up with subtopic pages]. -->

## Bugs and improvements

LASER is an open-source model, and we welcome code contributions, feature requests, documentation improvement requests, and identification of bugs. If you need to file a ticket (or submit a pull request), please submit your issue at <https://github.com/InstituteforDiseaseModeling/laser/issues>.

When reporting a bug, please include:
- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

If you are proposing a new feature or providing other feedback, please:
- Explain in detail how the feature would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that code contributions are welcome!


## Contributing to code

Code contributions are welcome! If you develop new features for LASER-core, fix bugs you've found in the model, or want to contribute a new disease type to the LASER framework, we would love to add your code.

### Setting up your local environment

1. Fork [LASER-core](https://github.com/InstituteforDiseaseModeling/laser) using the "Fork" button at the top right of the window.

2. Clone your fork locally:

    `git clone git@github.com:YOURGITHUBNAME/laser.git`

3. Create a branch for local development:

    `git checkout -b <name-of-your-bugfix-or-feature>`
    Now you can make your changes locally.

4. Run all the checks and docs builder after completing your changes:

    `tox`

5. Commit your changes and push your branch to GitHub:

    `git add .`

    `git commit -m "Your detailed description of your changes."`

    `git push origin <name-of-your-bugfix-or-feature>`

6. Submit a pull request through [GitHub](https://github.com/InstituteforDiseaseModeling/laser/pulls), following our pull request guidelines:
    - Include passing tests (run `tox`)
    - Update the documentation for new API, functionality, etc
    - Add a note to `CHANGELOG.md` about the changes
    - Add yourself to `AUTHORS.md`



### GitHub best practices

<!-- how to engage with IDM on GH, what's required for PRs etc (not bug tickets but how to fork & submit PRs) -->

### Development best practices

<!-- other help that's not related to unit tests or code optimization, relevant info from the "iterative development cycle" can go here (only if it's actual workflow steps; don't include if that's just a generalized workflow) -->

### Running unit tests

<!-- needs more information -->

To run a subset of tests:

`tox -e envname -- pytest -k test_myfeature`

To run all the test environments in parallel:

`tox -p auto`

### Optimizing code

<!-- Current optimization info is in the "getting started, optimization" section; if we decide it's more relevant here, it should move; also should add higher-level (eg dev-focused) optimization here, so may be worth moving the Numba, NumPy, C and OpenMP sections here. -->


