"""Command-line helpers shared by generation and neural-network commands."""

from __future__ import annotations

import argparse
import textwrap


class CompactHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Render option help as a compact table without a usage summary.

    Args:
        None: Uses the inherited formatter constructor.

    Returns:
        CompactHelpFormatter: The argparse formatter class.
    """

    def add_usage(self, usage, actions, groups, prefix=None) -> None:
        """Omit argparse's usage summary.

        Args:
            usage: Explicit usage text, if provided.
            actions: Parser actions.
            groups: Mutually exclusive action groups.
            prefix: Optional usage-line prefix.

        Returns:
            None: Suppresses the usage summary.
        """
        return None

    def _option_columns(self, action: argparse.Action) -> str:
        """Combine an action's option names.

        Args:
            action: Parser action whose option names should be formatted.

        Returns:
            str: Option names.
        """
        short_options = [
            option for option in action.option_strings
            if not option.startswith("--")
        ]
        long_options = [
            option for option in action.option_strings
            if option.startswith("--")
        ]
        return ", ".join(long_options + short_options)

    def _format_action(self, action: argparse.Action) -> str:
        """Format one action as a row in the options table.

        Args:
            action: Parser action to format.

        Returns:
            str: Formatted table row or positional-action text.
        """
        if not action.option_strings:
            return super()._format_action(action)

        if action.help is argparse.SUPPRESS:
            return ""

        option = self._option_columns(action)
        choices = ", ".join(map(str, action.choices)) if action.choices is not None else ""
        if action.default is None or action.default == argparse.SUPPRESS:
            default = ""
        elif isinstance(action.default, (list, tuple)):
            default = ", ".join(map(str, action.default))
        else:
            default = str(action.default)
        indent = " " * self._current_indent
        available = self._width - self._current_indent - 8
        option_width = min(25, max(21, int(available * 0.30)))
        choices_width = min(20, max(13, int(available * 0.19)))
        default_width = min(16, max(12, int(available * 0.17)))
        required_width = len("Required")
        help_width = max(
            12, available - option_width - choices_width - default_width - required_width
        )

        def wrap_cell(text: str, width: int) -> list[str]:
            return textwrap.wrap(text, width, break_on_hyphens=False)

        option_lines = wrap_cell(option, option_width)
        choice_lines = wrap_cell(choices, choices_width)
        default_lines = wrap_cell(default, default_width)
        help_text = (
            self._expand_help(action)
            if action.help
            else ""
        )
        help_lines = self._split_lines(help_text, help_width)
        row_count = max(len(option_lines), len(choice_lines), len(default_lines), len(help_lines), 1)
        rows = []
        for index in range(row_count):
            option_cell = option_lines[index] if index < len(option_lines) else ""
            choice_cell = choice_lines[index] if index < len(choice_lines) else ""
            default_cell = default_lines[index] if index < len(default_lines) else ""
            required_cell = "True" if index == 0 and action.required else ""
            help_cell = help_lines[index] if index < len(help_lines) else ""
            rows.append(
                f"{indent}{option_cell:<{option_width}}  "
                f"{choice_cell:<{choices_width}}  "
                f"{default_cell:<{default_width}}  "
                f"{required_cell:<{required_width}}  {help_cell}\n"
            )
        if not getattr(self, "_option_header_added", False):
            rows.insert(
                0,
                f"{indent}{'Option':<{option_width}}  "
                f"{'Choices':<{choices_width}}  "
                f"{'Default':<{default_width}}  "
                f"{'Required':<{required_width}}  Description\n",
            )
            self._option_header_added = True
        return "".join(rows)

    def format_help(self) -> str:
        """Add a blank line before and after the complete help output.

        Args:
            None: Uses the inherited formatter state.

        Returns:
            str: Formatted help text with surrounding blank lines.
        """
        self._option_header_added = False
        return "\n" + super().format_help().rstrip("\n") + "\n\n"


__all__ = ["CompactHelpFormatter"]