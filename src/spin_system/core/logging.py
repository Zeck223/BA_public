"""Small logging primitives shared by project workflows."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO


class TeeLogger:
    """Write text to a terminal stream and a log file simultaneously.

    Args:
        filepath: Destination log-file path.
        mode: File-open mode used for the log file.
        terminal: Optional terminal stream; defaults to standard output.

    Returns:
        TeeLogger: A tee logger instance.
    """

    def __init__(
        self,
        filepath: str | Path,
        mode: str = "w",
        terminal: TextIO | None = None,
    ) -> None:
        """Open the log file and select the terminal stream.

        Args:
            filepath: Destination log-file path.
            mode: File-open mode.
            terminal: Optional terminal stream.

        Returns:
            None: Initializes the logger streams.
        """
        self.terminal = terminal or sys.stdout
        self.log_file = Path(filepath)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.log_file.open(mode, encoding="utf-8")

    def write(self, message: str) -> int:
        """Write and flush a message to both output streams.

        Args:
            message: Text to write.

        Returns:
            int: Number of characters written.
        """
        self.terminal.write(message)
        self._file.write(message)
        self._file.flush()
        return len(message)

    def flush(self) -> None:
        """Flush both the terminal and log-file streams.

        Args:
            None: Uses the streams stored on this instance.

        Returns:
            None: Flushes both streams.
        """
        self.terminal.flush()
        self._file.flush()

    def close(self) -> None:
        """Close the log-file stream.

        Args:
            None: Uses the log file stored on this instance.

        Returns:
            None: Closes the file.
        """
        self._file.close()

    def __enter__(self) -> "TeeLogger":
        """Return this logger for use as a context manager.

        Args:
            None: Uses the current logger instance.

        Returns:
            TeeLogger: This logger instance.
        """
        return self

    def __exit__(self, exception_type, exception, traceback) -> None:
        """Close the log file when leaving a context manager.

        Args:
            exception_type: Exception class raised in the managed block, if any.
            exception: Exception instance raised in the managed block, if any.
            traceback: Traceback for the exception, if any.

        Returns:
            None: Closes the log file.
        """
        self.close()
