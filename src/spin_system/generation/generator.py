"""Load generation settings and build simulation parameter configurations."""

import argparse
from collections.abc import Sequence
from typing import Any

import numpy as np

from spin_system import FileManager
from spin_system.core.cli import CompactHelpFormatter
from spin_system.core.config import load_yaml
from spin_system.core.paths import ProjectPaths


def load_simulation_config(
    paths: ProjectPaths | None = None,
) -> dict[str, Any]:
    """Load the simulation-generation settings from YAML.

    Args:
        paths: Optional project paths used to locate the file.

    Returns:
        dict[str, Any]: The loaded simulation configuration.
    """
    file_manager = FileManager(paths=paths)
    return load_yaml(file_manager.paths.simulation_config)


def validate_simulation_config(simulation_config: dict[str, Any]) -> None:
    """Validate fields required by the parameter generator.

    Args:
        simulation_config: Configuration mapping to validate.

    Returns:
        None: This function validates the mapping in place.

    Raise:
        ValueError: If a required field is missing or a generation type is invalid.
    """
    required_fields = {
        "config_name",
        "generation_type",
        "hamiltonian_terms",
        "parameter_ranges",
        "base_parameters",
    }
    missing_fields = sorted(required_fields - simulation_config.keys())
    if missing_fields:
        raise ValueError(
            "Missing simulation configuration fields: "
            + ", ".join(missing_fields)
        )

    generation_type = simulation_config["generation_type"]
    if generation_type not in {"random", "grid"}:
        raise ValueError(f"Invalid generation type: {generation_type}")
    mode_field = "num_simulations" if generation_type == "random" else "values_per_term"
    if mode_field not in simulation_config:
        raise ValueError(f"Missing simulation configuration field: {mode_field}")


def create_config(
    simulation_config: dict[str, Any],
    parameter_sets: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Create a simulation configuration from settings and generated parameters.

    Args:
        simulation_config: Source settings, including Hamiltonian
            flags and ranges.
        parameter_sets: Mapping of simulation names to
            their parameter values.

    Returns:
        dict[str, Any]: Complete configuration for simulation runs.
    """

    hamiltonian_terms = {
        term: simulation_config['hamiltonian_terms'].get(term, False)
        for term in ("use_heisenberg", "use_dmi", "use_dipole", "use_ani")
    }

    checked_parameter_ranges = {}
    if hamiltonian_terms.get('use_heisenberg', False):
        checked_parameter_ranges['J'] = simulation_config['parameter_ranges'].get('J', [0.0, 0.0])
    if hamiltonian_terms.get('use_dmi', False):
        checked_parameter_ranges['DMI'] = simulation_config['parameter_ranges'].get('DMI', [0.0, 0.0])
    if hamiltonian_terms.get('use_dipole', False):
        checked_parameter_ranges['D'] = simulation_config['parameter_ranges'].get('D', [0.0, 0.0])
    if hamiltonian_terms.get('use_ani', False):
        checked_parameter_ranges['K'] = simulation_config['parameter_ranges'].get('K', [0.0, 0.0])

    config = {
        'config_name': simulation_config['config_name'],
        'base_parameters': simulation_config['base_parameters'],
        'hamiltonian_terms': hamiltonian_terms,
        'parameter_ranges': checked_parameter_ranges,
        'parameter_sets': parameter_sets
    }
    return config


def generate_random_parameters(
    simulation_config: dict[str, Any],
    rng: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Generate random parameter values for each requested simulation.

    Args:
        simulation_config: Settings containing active terms,
            ranges, and simulation count.
        rng: Random source with a ``uniform`` method; defaults to
            NumPy's global random module.

    Returns:
        dict[str, dict[str, Any]]: Mapping from simulation names to sampled values.
    """
    rng = rng or np.random
    hamiltonian_terms = simulation_config['hamiltonian_terms']

    use_heisenberg = hamiltonian_terms.get('use_heisenberg', False)
    use_dmi = hamiltonian_terms.get('use_dmi', False)
    use_dipole = hamiltonian_terms.get('use_dipole', False)
    use_ani = hamiltonian_terms.get('use_ani', False)

    j_values = simulation_config['parameter_ranges'].get('J', [0.0, 0.0])
    d_values = simulation_config['parameter_ranges'].get('D', [0.0, 0.0])
    dmi_values = simulation_config['parameter_ranges'].get('DMI', [0.0, 0.0])
    k_values = simulation_config['parameter_ranges'].get('K', [0.0, 0.0])

    parameter_sets = {}
    sim_idx = 1

    for _ in range(simulation_config['num_simulations']):
        name = f"sim_{sim_idx:06d}"
        parameter_sets[name] = {}

        if use_heisenberg:
            j_val = rng.uniform(j_values[0], j_values[1])
            parameter_sets[name]['J'] = j_val

        if use_dipole:
            d_val = rng.uniform(d_values[0], d_values[1])
            parameter_sets[name]['D'] = d_val

        if use_dmi:
            dmi_val = rng.uniform(dmi_values[0], dmi_values[1])
            parameter_sets[name]['DMI'] = dmi_val

        if use_ani:
            k_val = rng.uniform(k_values[0], k_values[1])
            parameter_sets[name]['K'] = k_val

        sim_idx += 1

    return parameter_sets


def generate_grid_parameters(
    simulation_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Generate parameter combinations from evenly spaced values.

    Args:
        simulation_config: Settings containing active terms,
            ranges, and grid size.

    Returns:
        dict[str, dict[str, Any]]: Mapping from simulation names to grid values.
    """
    hamiltonian_terms = simulation_config['hamiltonian_terms']
    values_per_term = simulation_config['values_per_term']

    use_heisenberg = hamiltonian_terms.get('use_heisenberg', False)
    use_dmi = hamiltonian_terms.get('use_dmi', False)
    use_dipole = hamiltonian_terms.get('use_dipole', False)
    use_ani = hamiltonian_terms.get('use_ani', False)

    j_range = simulation_config['parameter_ranges'].get('J', [0.0, 0.0])
    d_range = simulation_config['parameter_ranges'].get('D', [0.0, 0.0])
    dmi_range = simulation_config['parameter_ranges'].get('DMI', [0.0, 0.0])
    k_range = simulation_config['parameter_ranges'].get('K', [0.0, 0.0])

    j_values = np.linspace(j_range[0], j_range[1], values_per_term, endpoint=True).tolist() if use_heisenberg else [0.0]
    d_values = np.linspace(d_range[0], d_range[1], values_per_term, endpoint=True).tolist() if use_dipole else [0.0]
    dmi_values = np.linspace(dmi_range[0], dmi_range[1], values_per_term, endpoint=True).tolist() if use_dmi else [0.0]
    k_values = np.linspace(k_range[0], k_range[1], values_per_term, endpoint=True).tolist() if use_ani else [0.0]

    parameter_sets = {}
    sim_idx = 1

    for _ in range(1):
        for dmi_val in dmi_values:
            for d_val in d_values:
                for k_val in k_values:
                    name = f"sim_{sim_idx:06d}"
                    parameter_sets[name] = {}

                    if use_heisenberg:
                        parameter_sets[name]['J'] = 1

                    if use_dmi:
                        parameter_sets[name]['DMI'] = dmi_val

                    if use_dipole:
                        parameter_sets[name]['D'] = d_val

                    if use_ani:
                        parameter_sets[name]['K'] = k_val

                    sim_idx += 1

    return parameter_sets


def main(
    arguments: Sequence[str] | None = None,
    paths: ProjectPaths | None = None,
) -> int:
    """Run the parameter-generation command-line workflow.

    Args:
        arguments: Optional command-line arguments.
        paths: Paths for loading and saving configuration.

    Returns:
        int: Exit status, where zero indicates success.

    Raise:
        ValueError: If the configured generation type is unsupported.
    """

    parser = argparse.ArgumentParser(
        description="Generate parameter configurations for spin simulations.",
        formatter_class=CompactHelpFormatter,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional seed for reproducible random parameters.",
    )
    args = parser.parse_args(arguments)

    file_manager = FileManager()
    if paths is not None:
        file_manager = FileManager(paths=paths)
    simulation_config = load_simulation_config(paths)
    validate_simulation_config(simulation_config)

    rng = np.random.default_rng(args.seed) if args.seed is not None else np.random

    if simulation_config['generation_type'] == 'random':
        print("\n" + "=" * 60)
        print(f"Generating {simulation_config['num_simulations']} random parameter sets...")
        parameter_sets = generate_random_parameters(simulation_config, rng=rng)
    elif simulation_config['generation_type'] == 'grid':
        print("\n" + "=" * 60)
        print(f"Generating grid parameter sets with {simulation_config['values_per_term']} values per term...")
        parameter_sets = generate_grid_parameters(simulation_config)
    else:
        raise ValueError(f"Invalid generation type: {simulation_config['generation_type']}")

    generated_config = create_config(simulation_config, parameter_sets)


    file_manager.save_parameter_config(generated_config)

    print(f"Parameter configuration '{simulation_config['config_name']}.yaml' created successfully!")
    print("=" * 60 + "\n")
    return 0


if __name__ == "__main__":
    main()


