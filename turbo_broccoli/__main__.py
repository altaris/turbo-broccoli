# pylint: disable=import-outside-toplevel

"""Turbo Broccoli CLI"""
__docformat__ = "google"


import sys
from pathlib import Path
from typing import Iterable

try:
    import click
except ModuleNotFoundError:
    print(
        "Turbo Broccoli CLI requires click to be installed. You can install "
        "it by running `pip install click`.",
        file=sys.stderr,
    )
    sys.exit(-1)


@click.group()
def main():
    """Entrypoint."""


@main.command()
@click.argument(
    "file_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
        path_type=Path,
    ),
)
@click.option(
    "-f",
    "--full-path",
    is_flag=True,
    default=False,
    help="Prints the full path of the artifacts",
)
def list_artifacts(file_path: Path, full_path: bool, *_, **__):
    """Lists the artifacts referenced by the JSON file."""
    import json

    from .environment import get_artifact_path
    from .utils import artifacts

    with open(file_path, mode="r", encoding="utf-8") as fp:
        document = json.load(fp)
    a: Iterable = artifacts(document)
    if full_path:
        p = get_artifact_path().absolute()
        a = map(lambda x: p / x, a)
    print(*a, sep="\n")


@main.command()
@click.argument(
    "file_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=True,
        readable=True,
        path_type=Path,
    ),
)
@click.option("-d", "--dry-run", is_flag=True, default=False, help="Dry run")
def rm(file_path: Path, dry_run: bool, *_, **__):
    """
    Removes a json file and all the artifacts it references. Make sure the
    environment variable TB_ARTIFACT_PATH is correctly set.
    """
    import json

    from .environment import get_artifact_path
    from .utils import artifacts

    with open(file_path, mode="r", encoding="utf-8") as fp:
        document = json.load(fp)
    p = get_artifact_path().absolute()
    files = list(map(lambda x: p / x, artifacts(document))) + [
        file_path.absolute()
    ]
    for f in files:
        try:
            if not dry_run:
                f.unlink()
            print(f)
        except FileNotFoundError:
            print(f"rm: {f}: No such artifact", file=sys.stderr)


# pylint: disable=no-value-for-parameter
if __name__ == "__main__":
    main()
