"""Console and job-log reporting for simulation runs."""

import os
from typing import Any

from spin_system import ProjectPaths

class Logger:
    """Print simulation progress summaries and runtime information.

    Args:
        None.

    Returns:
        Logger: Class providing static logging methods.
    """
    SEPARATOR = "=" * 60

    def config_information(
        config: dict[str, Any],
        paths: ProjectPaths | None = None,
    ) -> None:
        """Print the configuration and output paths before a simulation run.

        Args:
            config: Simulation name and parameter sets.
            paths: Paths used to resolve displayed locations.

        Returns:
            None: Prints the configuration summary.
        """
        paths = paths or ProjectPaths()
        config_name = config["config_name"]

        print(Logger.SEPARATOR)
        print("STARTING SIMULATIONS")
        print(f"Config file: {paths.parameter_config_root / f'{config_name}.yaml'}")
        print(f"Data directory: {paths.config_directory(config_name)}")
        print(f"Total simulations: {len(config['parameter_sets'])}")
        print(Logger.SEPARATOR)


    def simulations_finished(
        total: int,
        results: dict[str, Any],
    ) -> None:
        """Print the number of successful simulations after a run.

        Args:
            total: Number of selected simulations.
            results: Successfully completed simulations.

        Returns:
            None: Prints the completion summary.
        """
        print(Logger.SEPARATOR)
        print("SIMULATIONS FINISHED")
        print(f"Successful: {len(results)}/{total}")
        print(Logger.SEPARATOR)


    def save_job_log(
        run_time: float,
        paths: ProjectPaths | None = None,
    ) -> None:
        """Print machine and elapsed-time information for a completed job.

        Args:
            run_time: Total runtime in seconds.
            paths: Optional project paths accepted for API
                consistency.

        Returns:
            None: Prints machine and runtime information.
        """
        paths = paths or ProjectPaths()
        print(f"Cores: {os.cpu_count()}")
        print(f"Job runtime: {run_time}")
