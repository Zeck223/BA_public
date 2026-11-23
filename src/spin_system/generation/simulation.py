"""Run configured Monte Carlo simulations and save spin-grid outputs."""

import argparse
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from spin_system import FileManager
from spin_system.core.paths import ProjectPaths

from .algorithms.metropolis import monte_carlo_sweep_metropolis
from .utils.cli import CompactHelpFormatter
from .utils.logger import Logger
from .utils.prepare_dipole import DipolePreparation, prepare_dipole_offsets


def _run_simulation_worker(
    sim_name: str,
    slim_config: dict[str, Any],
    grids_path: Path,
    dipole_preparation: DipolePreparation | None = None,
    seed: int | None = None,
) -> None:
    """Run one simulation in a worker process and save its final spins.

    Args:
        sim_name: Name identifying the parameter set and output grid.
        slim_config: Minimal configuration containing this
            simulation's settings.
        grids_path: Directory used for simulation output.
        dipole_preparation: Prepared dipole-neighbour
            arrays, if enabled.
        seed: Seed for reproducible initialization and Monte Carlo steps.

    Returns:
        None: The final grid is saved to disk.
    """
    base_parameters = slim_config['base_parameters']

    L = base_parameters['L']
    T_start = base_parameters['start_temp']
    T_end = base_parameters['end_temp']
    temp_steps = base_parameters['temp_steps']

    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    rng = np.random.default_rng(seed)
    theta = np.arccos(1 - 2 * rng.random((L, L)))
    phi = 2 * np.pi * rng.random((L, L))
    spins = np.empty((L, L, 3), dtype=np.float32)
    sin_theta = np.sin(theta)
    spins[:, :, 0] = sin_theta * np.cos(phi)
    spins[:, :, 1] = sin_theta * np.sin(phi)
    spins[:, :, 2] = np.cos(theta)

    temps = T_start + (T_end - T_start) * np.sqrt(
        np.linspace(0, 1, temp_steps, endpoint=True)
    )

    spins = monte_carlo_sweep_metropolis(
        slim_config, sim_name, spins, temps, dipole_preparation, seed=seed
    )

    FileManager(data_root=grids_path.parents[1]).save_spins(
        slim_config["config_name"], sim_name, spins
    )


class Simulation:
    """Coordinate configuration loading, parallel runs, and result logging.

    Args:
        args: Parsed config name and simulation range.
        file_manager: Manager for configuration and output paths.

    Returns:
        Simulation: A runner instance.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        file_manager: FileManager | None = None,
    ) -> None:
        """Initialize a runner from parsed arguments and a file manager.

        Args:
            args: Parsed config, range, and optional seed.
            file_manager: Manager for configuration and output.

        Returns:
            None: Initializes the runner.
        """
        self.config_name = args.config
        self.file_manager = file_manager or FileManager()
        self.config = self.file_manager.load_config(self.config_name)
        self.first_sim = args.first_sim
        self.last_sim = args.last_sim
        self.seed = getattr(args, "seed", None)
        self._create_save_paths()


    def _create_save_paths(self) -> None:
        """Create the configuration and spin-grid output directories.

        Args:
            None: Creates the output directories.

        Returns:
            None: Creates the output directories.
        """
        self.save_path = self.file_manager.paths.config_directory(self.config_name)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.grids_path = self.save_path / "grids"
        self.grids_path.mkdir(parents=True, exist_ok=True)


    def _create_slim_config(self, sim_name: str) -> dict[str, Any]:
        """
        Create a minimal configuration containing one parameter set.

        Args:
            sim_name: Simulation identifier to retain in the reduced mapping.

        Returns:
            dict[str, Any]: Configuration mapping for one worker process.
        """
        return {
            'base_parameters': self.config['base_parameters'],
            'hamiltonian_terms': self.config['hamiltonian_terms'],
            'config_name': self.config['config_name'],
            'parameter_sets': {sim_name: self.config['parameter_sets'][sim_name]}
        }


    def main(self, start_time: float) -> bool:
        """
        Run the selected simulations in parallel and report their outcomes.

        Args:
            start_time: Wall-clock start time for elapsed-time reporting.

        Returns:
            bool: Whether all selected simulations completed successfully.
        """

        Logger.config_information(self.config, paths=self.file_manager.paths)

        output_dir = self.save_path
        output_dir.mkdir(parents=True, exist_ok=True)

        self.file_manager.save_config(self.config)

        if self.config['hamiltonian_terms']['use_dipole']:
            L = self.config['base_parameters']['L']
            dipole_cutoff = self.config['base_parameters']['dipole_cutoff']
            dipole_preparation = prepare_dipole_offsets(L, dipole_cutoff)
        else:
            dipole_preparation = None

        results = {}

        task_specs = [
            (sim_name, self._create_slim_config(sim_name), dipole_preparation if self.config['hamiltonian_terms']['use_dipole'] else None)
            for sim_name in self.config['parameter_sets'].keys() if self.first_sim <= sim_name <= self.last_sim
        ]
        worker_seeds = np.random.SeedSequence(self.seed).spawn(len(task_specs))
        tasks = [
            (*task, int(worker_seed.generate_state(1, dtype=np.uint32)[0]))
            for task, worker_seed in zip(task_specs, worker_seeds)
        ]

        print(f"\nRunning {len(tasks)} simulations in parallel...\n")

        max_workers = min(os.cpu_count(), len(tasks)) if tasks else 1
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _run_simulation_worker,
                    sim_name,
                    slim_config,
                    self.grids_path,
                    dipole_prep,
                    seed,
                ): sim_name
                for sim_name, slim_config, dipole_prep, seed in tasks
            }

            completed = 0

            for future in as_completed(futures):
                sim_name = futures[future]
                completed += 1

                try:
                    future.result()

                    percentage = completed / len(tasks) * 100
                    elapsed = time.time() - start_time
                    if len(tasks) > 500 and completed % 100 == 0:
                        print(f"[MAIN] SUCCESS: '{sim_name}' completed {percentage:.2f}% TIME: {elapsed:.2f}s", end="\r")
                    elif len(tasks) <= 500:
                        print(f"[MAIN] SUCCESS: '{sim_name}' completed {percentage:.2f}% TIME: {elapsed:.2f}s", end="\r")
                    results[sim_name] = {"success": True}

                except Exception as e:
                    print(f"[MAIN] ERROR in '{sim_name}': {e}")
                    traceback.print_exception(type(e), e, e.__traceback__)

        Logger.simulations_finished(len(tasks), results)
        return len(results) == len(tasks)


def main(
    arguments: Sequence[str] | None = None,
    paths: ProjectPaths | None = None,
) -> int:
    """Run the simulation command-line workflow.

    Args:
        arguments: Optional command-line arguments.
        paths: Paths used by the file manager.

    Returns:
        int: Exit status, where zero indicates that all selected simulations succeeded.
    """

    parser = argparse.ArgumentParser(
        description="Simulation program for spin systems",
        formatter_class=CompactHelpFormatter,
    )

    parser.add_argument(
        "--config",
        "-c",
        help="Name of the configuration (without .yaml extension)",
        required=True
    )
    parser.add_argument(
        "--first_sim",
        "-f",
        help="Name of the first simulation to run (default: sim_000001)",
        default="sim_000001"
    )
    parser.add_argument(
        "--last_sim",
        "-l",
        help="Name of the last simulation to run (default: sim_999999)",
        default="sim_999999"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional seed for reproducible, independent simulations.",
    )

    args = parser.parse_args(arguments)
    start_time = time.time()

    file_manager = FileManager(paths=paths)
    try:
        file_manager.load_config(args.config)
    except FileNotFoundError:
        print(f"ERROR: Config '{args.config}' does not exist.")
        return 1

    simulation = Simulation(args, file_manager=file_manager)
    success = simulation.main(start_time)
    run_time = time.time() - start_time
    print(f"TIME: {run_time}")
    Logger.save_job_log(run_time, paths=file_manager.paths)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())