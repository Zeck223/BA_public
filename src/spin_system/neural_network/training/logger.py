"""Training-specific console and file logging."""

import torch

from ...core.logging import TeeLogger


class NNLogger(TeeLogger):
    """Add neural-network training status messages to a tee log.

    Args:
        filepath: Destination log-file path.
        mode: File-open mode used for the log file.
        terminal: Optional terminal output stream.

    Returns:
        NNLogger: A training logger instance.
    """

    def close(self):
        """Close the log and report its output path.

        Args:
            None: Uses the log file stored on this instance.

        Returns:
            None: Closes the log and prints its destination.
        """
        output_file = self.log_file
        super().close()
        print(f"\nOutput saved to: {output_file}")

    def initialization(self, output_file) -> None:
        """Print initial model, output, and device information.

        Args:
            output_file: Path of the log output file.

        Returns:
            None: Prints the training initialization banner.
        """
        print("-" * 50)
        print("Using model: SpinResNet18")
        print(f"Output will be saved to: {output_file}")

        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
        print("-" * 50)

        print("Loading and processing data...", end="\r")