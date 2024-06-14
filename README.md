# CompRAG

A multilingual rag pipeline with benchmarking, and a test ui application included.

## Getting started
- Clone the project locally.
- install poetry, the build tool for this project, follow the [guide](https://python-poetry.org/docs/#installing-with-the-official-installer)
- To install the dependencies run `poetry install`
- Install and upgrade pre-commit hooks
```bash
poetry install
poetry run pre-commit install
```

### Code formatting, linting
#### Sorting imports
We are sorting the imports to minimize diffs when merging  
To sort the imports in the whole project manually use: `poetry run isort .`

#### Formatting files
We are formatting the codebase to ensure consistent coding style  
To format the codebase use: `poetry run black .`

#### Lint the files
To check copatibilty with the PEP8 standard we lint the code  
To lint the whole projet manually use: `poetry run flake8 .`

### Scripts
- Run `poetry run format` to sort the imports and format the files
- Run `poetry run lint` to lint all files
- Run `poetry run type-check` to typecheck all files
- Run `poetry run check` to do all of the above in sequence

### Pre-commit hook
Install using `poetry run pre-commit install`  
After this on every commit your commit will be checked for formatting,  linting and typing, to ensure atomic commits
If you want to avoid the check for some reason you can use th `--no-verify` flag, for example
```bash
git commit -am"Testing no verify" --no-verify
```

### Troubleshooting
- You should run `poetry install` if a change involves `pyproject.toml`
- If you see a change in `.pre-commit-config.yaml` you sould reinstall the pre-commit hooks
If someting breaks in your workflow, first make sure that your are up to date. After a `git pull` you should run
```bash
poetry install
poetry run pre-commit install
```

