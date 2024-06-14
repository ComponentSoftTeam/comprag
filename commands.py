# This file contains scripts for the project
import shlex
import subprocess
from pathlib import Path
from typing import List


def get_repo_root() -> Path:
    command = "git rev-parse --show-toplevel"
    repo_root = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        text=True,
        shell=True,
    ).stdout
    return Path(repo_root.strip()).resolve()


def get_git_python_files() -> List[str]:
    root = get_repo_root()
    command = f'git -C "{shlex.quote(str(root))}" ' "ls-files --cached --others --exclude-standard"

    git_files_str = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        text=True,
        shell=True,
    ).stdout

    git_file_paths = [Path(p.strip()) for p in git_files_str.strip().split("\n")]
    python_file_paths = [root.joinpath(p) for p in git_file_paths if p.suffix == ".py"]

    return [shlex.quote(str(p)) for p in python_file_paths]


def fmt() -> None:
    """Custom command to run isort and black."""

    root = get_repo_root()
    python_files = get_git_python_files()
    subprocess.run(
        [
            "python",
            "-m",
            "isort",
        ]
        + python_files,
        cwd=root,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "black",
        ]
        + python_files,
        cwd=root,
    )


def lint() -> None:
    """Custom command to run isort and black."""

    root = get_repo_root()
    python_files = get_git_python_files()
    subprocess.run(
        [
            "python",
            "-m",
            "flake8",
        ]
        + python_files,
        cwd=root,
    )


def type_check() -> None:
    """Custom command to run mypy."""

    root = get_repo_root()
    python_files = get_git_python_files()
    subprocess.run(
        [
            "python",
            "-m",
            "mypy",
            "--explicit-package-bases",
        ]
        + python_files,
        cwd=root,
    )


def check() -> None:
    """Format, lint, typecheck"""

    fmt()
    lint()
    type_check()
