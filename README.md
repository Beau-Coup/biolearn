# BioLearn
A simple library in Python/[JAX](https://github.com/jax-ml/jax) for testing
different learning methods with STL for biomolecular models.

## Installing
This project uses `uv` to take care of dependencies and project management.
Please see the [uv](https://docs.astral.sh/uv/) project for instructions on installation for your platform.

After installing `uv`, from the repo root:

```bash
uv sync
```

## Usage
Run the current entrypoint:

```bash
uv run main.py
```

As the library grows, the recommended usage will be via a package entrypoint
under `biolearn` and example scripts under `examples/`.

## Scripts
Run scripts as modules from the repo root:

```bash
uv run python -m scripts.ss_classification
```

For other scripts, replace `ss_classification` with the module name.

## Tests
If tests are present, run them from the repo root:

```bash
uv run pytest
```

## Project Structure

The code is meant to be modular with the following use cases in mind.
1. Testing and implementation different models for biological systems.
2. Testing of different loss functions.
3. Use with various STL specifications.
4. Verification of learned models.

With this in mind, the project should follow a structure similar to the following
(suggested; folders can be introduced as needed):

```
├── biolearn
│   ├── losses
│   ├── models
│   ├── specifications
│   └── utils
├── data
├── scripts
├── pyproject.toml
└── README.md
```

Notes:
- `models/` holds biomolecular system models and parameterizations.
- `losses/` captures loss functions and objective definitions.
- `stl/` provides Signal Temporal Logic parsers and evaluators.
- `verification/` contains routines to validate learned models against STL specs.
- `examples/` and `scripts/` are for runnable experiments and utilities.


## Contributing
Contributions are welcome.
- Open an issue or discussion before large changes.
- Keep changes scoped and add tests when introducing new behavior.
- Use clear, descriptive docstrings and type hints where helpful.

## License
MIT. See [`LICENSE`](LICENSE).
