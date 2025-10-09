# Contributing

LASER is an open-source model, and we welcome code contributions, feature requests, documentation improvement requests, and identification of bugs. Every little bit helps, and credit will always be given.

## Bug reports 

When [reporting a bug](https://github.com/InstituteforDiseaseModeling/laser/issues) please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

## Documentation improvements

LASER could always use more documentation, whether as part of the official LASER docs, in docstrings, or even on the web in blog posts, articles, and such.

## Feature requests and feedback

The best way to send feedback is to file an issue in the [LASER GitHub repo](https://github.com/InstituteforDiseaseModeling/laser/issues).

If you are proposing a feature:

- Explain in detail how the feature would work.
- Keep the scope as narrow as possible to make it easier to implement.
- Remember that this is a volunteer-driven project, and that code contributions are welcome! 

## Development

If you develop new features for `laser`, fix bugs you've found in the model, or want to contribute a new disease type to the LASER framework, we would love to add your code.

### Set up your local environment

1. Fork [LASER-core](https://github.com/InstituteforDiseaseModeling/laser) using the "Fork" button at the top right of the window.

2. Clone your fork locally:

    `git clone git@github.com:YOURGITHUBNAME/laser.git`

3. Install [`uv`](https://github.com/astral-sh/uv?tab=readme-ov-file#installation) _in your system [Python]_, i.e. _before_ creating and activating a virtual environment.
   
4. Install `tox` as a tool in `uv` with the `tox-uv` plugin with
    `uv tool install tox --with tox-uv`
5. Change to the `laser-core` directory with
    `cd laser-core`
6. Create a virtual environment for development with
    `uv venv`
7. Activate the virtual environment with

    * Mac or Linux:
      `source .venv/bin/activate`

    * Windows:
      `.venv\bin\Activate`

8. Create a branch for local development:

    `git checkout -b <name-of-your-bugfix-or-feature>`
    Now you can make your changes locally.
    
9. Run all the checks and docs builder after completing your changes:

    `tox`

10. Commit your changes and push your branch to GitHub:

    `git add .`

    `git commit -m "Your detailed description of your changes."`

    `git push origin <name-of-your-bugfix-or-feature>`

11. Submit a pull request through [GitHub](https://github.com/InstituteforDiseaseModeling/laser/pulls), following our pull request guidelines:
    - Include passing tests (run `tox`)
    - Update the documentation for new API, functionality, etc
    - Add a note to `CHANGELOG.md` about the changes
    - Add yourself to `AUTHORS.md`

### Pull request guidelines

If you need some code review or feedback while you're developing the code just make the pull request.

For merging, you should:

1. Include passing tests (run `tox`).
2. Update documentation when there's new API, functionality etc.
3. Add a note to `CHANGELOG.md` about the changes.
4. Add yourself to `AUTHORS.md`.

### Run tests
Now you can run tests in the `tests` directory or run the entire check+docs+test suite with ```tox```. Running ```tox``` will run several consistency checks, build documentation, run tests against the supported versions of Python, and create a code coverage report based on the test suite. Note that the first run of ```tox``` may take a few minutes (~5). Subsequent runs should be quicker depending on the speed of your machine and the test suite (~2 minutes). You can use ```tox``` to run tests against a single version of Python with, for example, ```tox -e py310```.

To run a subset of tests:

```sh
tox -e envname -- pytest -k test_myfeature
```

To run all the test environments in *parallel*:

```sh
tox -p auto
```
Note, to combine the coverage data from all the tox environments run:

*   For Windows
    ```sh
    set PYTEST_ADDOPTS=--cov-append
    tox
    ```
*   For other operating systems
    ```sh
    PYTEST_ADDOPTS=--cov-append tox
    ```

