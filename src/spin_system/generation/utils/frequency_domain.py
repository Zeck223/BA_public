"""Convert saved spin grids into log-magnitude frequency spectra."""

import argparse
import sys
from collections.abc import Sequence

import numpy as np
from spin_system import FileManager
from spin_system.core.cli import CompactHelpFormatter
from spin_system.core.paths import ProjectPaths


class FrequencyDomainConverter:
    """Convert spin-grid data to and save frequency-domain spectra.

    Args:
        config_name: Name of the simulation configuration to process.
        file_manager: Manager for loading grids and saving
            spectra.

    Returns:
        FrequencyDomainConverter: A converter instance.
    """

    def __init__(
        self,
        config_name: str,
        file_manager: FileManager | None = None,
    ) -> None:
        """Initialize the converter for a configuration.

        Args:
            config_name: Name of the simulation configuration to process.
            file_manager: Manager for loading and saving arrays.

        Returns:
            None: Initializes the converter.
        """
        self.config_name = config_name
        self.file_manager = file_manager or FileManager()

    def convert_to_frequency_domain(self, sim_name: str) -> np.ndarray:
        """Convert one spin grid into a centered log-magnitude spectrum.

        Args:
            sim_name: Name of the simulation whose spin grid is loaded.

        Returns:
            np.ndarray: Log-magnitude spectrum with one channel per spin component.
        """

        spins = self.file_manager.load_array(self.config_name, sim_name, source="grids")

        # Perform FFT on the spins
        fft_result = np.stack([np.fft.fft2(spins[..., i]) for i in range(3)], axis=-1)

        # Compute the centered log-magnitude spectrum
        magnitude = np.abs(fft_result)
        spins_freq = np.fft.fftshift(
            np.log(np.maximum(magnitude, np.finfo(magnitude.dtype).tiny)),
            axes=(0, 1),
        )

        return spins_freq

    def convert_all_simulations(self) -> int:
        """Convert every saved spin grid for this configuration.

        Args:
            None: The configuration is selected when this converter is created.

        Returns:
            int: Number of converted simulations.
        """

        source_directory = self.file_manager.paths.source_directory(self.config_name, "grids")

        sim_files = list(source_directory.glob("*.npy"))
        sim_count = len(sim_files)
        if sim_count == 0:
            return 0
        progress_interval = max(1, sim_count // 100)
        for sim_num, sim_file in enumerate(sim_files, 1):
            sim_name = sim_file.stem
            spins_freq = self.convert_to_frequency_domain(sim_name)

            # Save the magnitude spectrum
            self.file_manager.save_array(
                self.config_name,
                sim_name,
                spins_freq,
                source="magnitude",
            )

            if sim_num % progress_interval == 0:
                progress = (sim_num / sim_count) * 100
                print(f"Progress: {progress:.0f}%\r", end="")
        return sim_count


def main(
    arguments: Sequence[str] | None = None,
    paths: ProjectPaths | None = None,
) -> int:
    """Run the frequency-domain conversion command-line workflow.

    Args:
        arguments: Optional command-line arguments.
        paths: Paths used by the file manager.

    Returns:
        int: Exit status, where zero indicates successful conversion.
    """

    parser = argparse.ArgumentParser(
        prog="freq",
        description="Convert spin simulations to the frequency domain (FFT).",
        formatter_class=CompactHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        metavar="CONFIG",
        help="Configuration name (without the .yaml extension)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-s", "--sim_name",
        metavar="SIM_NAME",
        help="Name of a single simulation (e.g., sim_000001)",
    )
    group.add_argument(
        "-a", "--all",
        action="store_true",
        help="Convert all simulations in the configuration",
    )

    args = parser.parse_args(arguments)
    converter = FrequencyDomainConverter(
        args.config,
        file_manager=FileManager(paths=paths),
    )

    if args.sim_name:
        spins_freq = converter.convert_to_frequency_domain(args.sim_name)
        out_path = converter.file_manager.save_array(
            args.config,
            args.sim_name,
            spins_freq,
            source="magnitude",
        )
        print(f"Saved: {out_path}")
    elif args.all:
        if not converter.convert_all_simulations():
            print(f"ERROR: No spin grids found for '{args.config}'.", file=sys.stderr)
            return 1
        print(f"Converted all simulations from '{args.config}'.")
    else:
        parser.error("Specify either -s/--sim_name or -a/--all.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())