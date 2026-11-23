"""Unified command-line entry point for the project."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from typing import Any

from .core.cli import CompactHelpFormatter
from .core.paths import ProjectPaths


_COMMANDS = (
    "generator",
    "simulation",
    "plot",
    "video",
    "freq",
    "train",
    "evaluate",
)
_COMMAND_ALIASES = {"eval": "evaluate"}
_COMMAND_DESCRIPTIONS = {
    "generator": "Generate parameter configurations",
    "simulation": "Run Monte Carlo simulations",
    "plot": "Create visualizations",
    "video": "Create a zooming mosaic video",
    "freq": "Convert spin grids to the frequency domain",
    "train": "Train a neural network",
    "evaluate": "Evaluate a trained neural network",
    "eval": "Alias for evaluate",
}


def _load_handler(command: str) -> Callable[..., Any]:
    if command == "generator":
        from .generation.generator import main
    elif command == "simulation":
        from .generation.simulation import main
    elif command == "plot":
        from .generation.plot import main
    elif command == "video":
        from .video.cli import main
    elif command == "freq":
        from .generation.utils.frequency_domain import main
    elif command == "train":
        from .neural_network.train import main
    elif command == "evaluate":
        from .neural_network.evaluate import main
    else:
        raise ValueError(f"Unknown command: {command}")
    return main


def _run_command(
    command: str,
    arguments: Sequence[str] | None = None,
) -> int:
    arguments = list(arguments or [])
    command = _COMMAND_ALIASES.get(command, command)
    paths = ProjectPaths()
    handler = _load_handler(command)
    result = handler(arguments=arguments, paths=paths)
    return 0 if result is None else int(result)


def create_parser() -> argparse.ArgumentParser:
    commands = _COMMANDS + tuple(_COMMAND_ALIASES)
    command_width = max(len("Command"), *(len(command) for command in commands))
    command_list = "\n".join(
        f"  {command:<{command_width}}  {_COMMAND_DESCRIPTIONS[command]}"
        for command in commands
    )
    command_table = (
        f"  {'Command':<{command_width}}  Description\n"
        f"{command_list}"
    )
    parser = argparse.ArgumentParser(
        prog="lms",
        description="Command-line entry points for the spin-system project",
        epilog=f"{command_table}",
        formatter_class=CompactHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=commands,
        metavar="COMMAND",
        help="Command to run (see list below)",
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def parse_args(arguments=None):
    return create_parser().parse_args(arguments)


def main(arguments=None) -> int:
    args = parse_args(arguments)
    return _run_command(args.command, args.arguments)


def generator() -> int:
    return _run_command("generator")


def simulation() -> int:
    return _run_command("simulation")


def plot() -> int:
    return _run_command("plot")


def video() -> int:
    return _run_command("video")


def freq() -> int:
    return _run_command("freq")


def train() -> int:
    return _run_command("train")


def eval_command() -> int:
    return _run_command("evaluate")


__all__ = [
    "create_parser",
    "eval_command",
    "freq",
    "generator",
    "main",
    "parse_args",
    "plot",
    "simulation",
    "train",
    "video",
]
